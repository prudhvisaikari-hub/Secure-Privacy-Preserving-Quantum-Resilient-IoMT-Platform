"""
fl_server.py
============
Federated Learning aggregation server using the Flower framework.
Implements FedAvg with optional Differential Privacy noise injection
at the server aggregation level (global DP).

Architecture:
  - N hospital clients each train locally on their vitals data
  - Server aggregates model updates using FedAvg
  - Optional: Gaussian noise added to aggregated weights (global DP)
  - Optional: Secure aggregation via Shamir secret sharing placeholder

Usage:
    # Start the FL server (listens for clients)
    python federated_learning/fl_server.py --rounds 50 --min-clients 3 --dp-epsilon 1.0

    # Clients connect separately via fl_client.py
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Flower import
# ---------------------------------------------------------------------------
try:
    import flwr as fl
    from flwr.common import (
        FitRes, Parameters, Scalar, ndarrays_to_parameters,
        parameters_to_ndarrays, NDArrays
    )
    from flwr.server.strategy import FedAvg
    from flwr.server.client_proxy import ClientProxy
    _FLOWER_AVAILABLE = True
except ImportError:
    _FLOWER_AVAILABLE = False
    logger.warning("Flower not installed — using local simulation mode.")

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Global DP noise injection
# ---------------------------------------------------------------------------

class DPAggregator:
    """
    Adds calibrated Gaussian noise to aggregated model updates
    to achieve (epsilon, delta)-DP at the global level.

    Based on: Geyer et al., "Differentially Private Federated Learning" (2017)
              McMahan et al., "Learning Differentially Private Recurrent Language Models" (2018)

    Args:
        epsilon:   Privacy budget (smaller = more private, less utility)
        delta:     Failure probability (typically 1/N^1.1 where N = training size)
        sensitivity: L2 sensitivity of clipped model updates (= clip_norm)
        noise_multiplier: σ relative to sensitivity (computed from ε, δ via RDP accounting)
    """

    def __init__(self,
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 sensitivity: float = 1.0,
                 num_rounds: int = 50):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.num_rounds = num_rounds
        self.noise_multiplier = self._calibrate_noise()
        logger.info(
            f"[DP] ε={epsilon}, δ={delta}, sensitivity={sensitivity}, "
            f"σ (noise_multiplier)={self.noise_multiplier:.4f}"
        )

    def _calibrate_noise(self) -> float:
        """
        Approximate noise multiplier σ via the analytic Gaussian mechanism.
        For proper calibration, use autodp or RDP accountant from Opacus.
        This is a closed-form approximation for quick reference.
        """
        import math
        # σ ≈ sqrt(2 * ln(1.25/δ)) / ε  (Gaussian mechanism)
        sigma = math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon
        # Amplification by subsampling over rounds (conservative)
        sigma = sigma * math.sqrt(self.num_rounds)
        return max(sigma, 0.1)  # floor to avoid zero noise on very high epsilon

    def add_noise(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Add calibrated Gaussian noise to each weight array."""
        noisy = []
        for w in weights:
            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_multiplier * self.sensitivity,
                size=w.shape
            ).astype(w.dtype)
            noisy.append(w + noise)
        return noisy

    def clip_update(self, update: List[np.ndarray], clip_norm: float) -> List[np.ndarray]:
        """Clip L2 norm of model update to clip_norm (sensitivity)."""
        flat = np.concatenate([u.flatten() for u in update])
        l2 = np.linalg.norm(flat)
        scale = min(1.0, clip_norm / (l2 + 1e-8))
        return [u * scale for u in update]

    @property
    def privacy_report(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "sensitivity": self.sensitivity,
            "num_rounds": self.num_rounds,
            "mechanism": "Gaussian",
        }


# ---------------------------------------------------------------------------
# Vitals Prediction Model (shared architecture between server and clients)
# ---------------------------------------------------------------------------

class VitalsPredictionModel(nn.Module if _TORCH_AVAILABLE else object):
    """
    Lightweight LSTM model for ICU vitals prediction.
    Predicts deterioration risk (binary) from time-series of:
      heart_rate, spo2, respiratory_rate, systolic_bp, temperature

    Input:  (batch, seq_len=24, n_features=5)
    Output: (batch, 1) — risk score [0, 1]
    """

    def __init__(self, n_features: int = 5, hidden_size: int = 64, num_layers: int = 2):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=4, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)             # (B, T, H)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (B, T, H)
        pooled = attn_out.mean(dim=1)           # (B, H)
        return self.classifier(pooled)          # (B, 1)

    def get_weights(self) -> List[np.ndarray]:
        if not _TORCH_AVAILABLE:
            return []
        return [v.cpu().numpy() for v in self.state_dict().values()]

    def set_weights(self, weights: List[np.ndarray]):
        if not _TORCH_AVAILABLE:
            return
        params = zip(self.state_dict().keys(), weights)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params}
        )
        self.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Custom Flower Strategy with DP
# ---------------------------------------------------------------------------

class DPFedAvg(FedAvg if _FLOWER_AVAILABLE else object):
    """
    FedAvg with server-side Differential Privacy noise injection.
    Extends Flower's built-in FedAvg strategy.
    """

    def __init__(self, dp_aggregator: Optional[DPAggregator] = None, **kwargs):
        if _FLOWER_AVAILABLE:
            super().__init__(**kwargs)
        self.dp = dp_aggregator
        self.round_metrics: List[dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not _FLOWER_AVAILABLE:
            return None, {}

        # Standard FedAvg aggregation
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None and self.dp is not None:
            # Inject DP noise into aggregated weights
            weights = parameters_to_ndarrays(aggregated_params)
            noisy_weights = self.dp.add_noise(weights)
            aggregated_params = ndarrays_to_parameters(noisy_weights)
            logger.info(
                f"[Round {server_round}] DP noise added "
                f"(σ={self.dp.noise_multiplier:.4f}, ε={self.dp.epsilon})"
            )

        # Log round metrics
        num_clients = len(results)
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        self.round_metrics.append({
            "round": server_round,
            "num_clients": num_clients,
            "total_examples": total_examples,
            "dp_epsilon": self.dp.epsilon if self.dp else None,
            "timestamp": time.time(),
        })

        return aggregated_params, metrics

    def save_metrics(self, path: str = "benchmarks/results/fl_server_metrics.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.round_metrics, f, indent=2)
        logger.info(f"Server metrics saved to {path}")


# ---------------------------------------------------------------------------
# Simulation mode (without Flower — for local testing)
# ---------------------------------------------------------------------------

class LocalFLSimulator:
    """
    Simulates N-client federated learning in a single process.
    Useful for rapid prototyping / CI without a Flower server.
    """

    def __init__(self,
                 n_clients: int = 5,
                 n_rounds: int = 10,
                 dp_epsilon: Optional[float] = 1.0,
                 dp_delta: float = 1e-5,
                 clip_norm: float = 1.0):
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.clip_norm = clip_norm
        self.dp = DPAggregator(
            epsilon=dp_epsilon, delta=dp_delta,
            sensitivity=clip_norm, num_rounds=n_rounds
        ) if dp_epsilon is not None else None

        if _TORCH_AVAILABLE:
            self.global_model = VitalsPredictionModel()
        else:
            self.global_model = None

        self.history: List[dict] = []

    def _simulate_client_update(self, client_id: int, global_weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Simulates a client training step: adds small random gradient noise
        to global weights (representing local SGD on local dataset).
        In real deployment, this is replaced by actual training in fl_client.py.
        """
        np.random.seed(client_id)
        updated = []
        for w in global_weights:
            # Simulate gradient update: w ← w - lr * grad (with noise = simulated gradient)
            lr = 0.01
            grad = np.random.randn(*w.shape).astype(w.dtype) * 0.1
            updated.append(w - lr * grad)
        return updated

    def fedavg(self, all_updates: List[List[np.ndarray]], weights: Optional[List[int]] = None) -> List[np.ndarray]:
        """Weighted FedAvg aggregation."""
        if weights is None:
            weights = [1] * len(all_updates)
        total = sum(weights)
        aggregated = [
            sum(w * u[i] for w, u in zip(weights, all_updates)) / total
            for i in range(len(all_updates[0]))
        ]
        return aggregated

    def run(self) -> List[dict]:
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available — using random numpy arrays for simulation.")
            global_weights = [np.random.randn(64, 5).astype(np.float32) for _ in range(4)]
        else:
            global_weights = self.global_model.get_weights()

        logger.info(f"Starting FL simulation: {self.n_clients} clients, {self.n_rounds} rounds")
        if self.dp:
            logger.info(f"DP enabled: {self.dp.privacy_report}")

        for rnd in range(1, self.n_rounds + 1):
            client_updates = []
            for cid in range(self.n_clients):
                update = self._simulate_client_update(cid, global_weights)
                if self.dp:
                    update = self.dp.clip_update(update, self.clip_norm)
                client_updates.append(update)

            aggregated = self.fedavg(client_updates)

            if self.dp:
                aggregated = self.dp.add_noise(aggregated)

            global_weights = aggregated

            # Simulate evaluation metric (decreasing loss)
            sim_loss = 1.0 / (1.0 + 0.1 * rnd) + np.random.uniform(-0.01, 0.01)
            sim_auc  = min(0.95, 0.50 + 0.008 * rnd + np.random.uniform(-0.005, 0.005))

            round_stats = {
                "round": rnd,
                "simulated_loss": round(float(sim_loss), 4),
                "simulated_auc": round(float(sim_auc), 4),
                "dp_epsilon": self.dp.epsilon if self.dp else None,
                "clients": self.n_clients,
            }
            self.history.append(round_stats)
            logger.info(
                f"  Round {rnd:3d}/{self.n_rounds} | Loss: {sim_loss:.4f} | AUC: {sim_auc:.4f}"
            )

        if not _TORCH_AVAILABLE or self.global_model is None:
            pass
        else:
            self.global_model.set_weights(global_weights)

        return self.history

    def save_history(self, path: str = "benchmarks/results/fl_history.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"FL history saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SPQR-IoMT Federated Learning Server")
    parser.add_argument("--rounds",       type=int,   default=50)
    parser.add_argument("--min-clients",  type=int,   default=3)
    parser.add_argument("--dp-epsilon",   type=float, default=1.0, help="Set to 0 to disable DP")
    parser.add_argument("--dp-delta",     type=float, default=1e-5)
    parser.add_argument("--clip-norm",    type=float, default=1.0)
    parser.add_argument("--simulate",     action="store_true", help="Run local simulation without Flower")
    args = parser.parse_args()

    if args.simulate or not _FLOWER_AVAILABLE:
        print("\n[Mode: Local Simulation]\n")
        sim = LocalFLSimulator(
            n_clients=args.min_clients,
            n_rounds=args.rounds,
            dp_epsilon=args.dp_epsilon if args.dp_epsilon > 0 else None,
            dp_delta=args.dp_delta,
            clip_norm=args.clip_norm,
        )
        history = sim.run()
        sim.save_history()
        print(f"\nFinal round AUC: {history[-1]['simulated_auc']}")
        print(f"DP privacy report: {sim.dp.privacy_report if sim.dp else 'DP disabled'}")
    else:
        dp_agg = DPAggregator(
            epsilon=args.dp_epsilon,
            delta=args.dp_delta,
            sensitivity=args.clip_norm,
            num_rounds=args.rounds
        ) if args.dp_epsilon > 0 else None

        strategy = DPFedAvg(
            dp_aggregator=dp_agg,
            min_fit_clients=args.min_clients,
            min_evaluate_clients=args.min_clients,
            min_available_clients=args.min_clients,
        )

        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
        strategy.save_metrics()


if __name__ == "__main__":
    main()
