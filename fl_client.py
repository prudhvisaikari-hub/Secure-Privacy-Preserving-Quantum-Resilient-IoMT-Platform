"""
fl_client.py
============
Federated Learning hospital client.
Performs local training with DP-SGD (via Opacus) and participates in
Flower-based federated rounds.

Each client represents one hospital with local patient vitals data.
Supports:
  - DP-SGD via Opacus (local differential privacy)
  - Gradient clipping before sending updates to server
  - Configurable local epochs per round

Usage:
    # Run a simulated hospital client (connects to fl_server.py)
    python federated_learning/fl_client.py --hospital-id hospital_001 --server 127.0.0.1:8080

    # Or simulate locally (no server needed):
    python federated_learning/fl_client.py --simulate
"""

import argparse
import logging
import json
import time
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — using simulation mode.")

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    _OPACUS_AVAILABLE = True
except ImportError:
    _OPACUS_AVAILABLE = False
    logger.warning("Opacus not installed — DP-SGD disabled (plain SGD will be used).")

try:
    import flwr as fl
    from flwr.common import NDArrays, Scalar
    _FLOWER_AVAILABLE = True
except ImportError:
    _FLOWER_AVAILABLE = False

from federated_learning.fl_server import VitalsPredictionModel


# ---------------------------------------------------------------------------
# Synthetic vitals dataset generator
# ---------------------------------------------------------------------------

class SyntheticVitalsDataset:
    """
    Generates synthetic ICU vitals time-series data for one hospital.
    In production, replace with MIMIC-III/PhysioNet data loader.

    Features (per timestep):
      0: heart_rate      (bpm)     ~ N(75, 15)
      1: spo2            (%)       ~ N(97, 2)
      2: respiratory_rate (bpm)   ~ N(16, 4)
      3: systolic_bp     (mmHg)   ~ N(120, 20)
      4: temperature     (°C)     ~ N(36.8, 0.5)

    Label: deterioration risk (1 = deterioration event within 6h)
    """

    def __init__(self,
                 hospital_id: str,
                 n_patients: int = 500,
                 seq_len: int = 24,
                 n_features: int = 5,
                 positive_rate: float = 0.25,
                 seed: Optional[int] = None):
        self.hospital_id = hospital_id
        self.n_patients = n_patients
        self.seq_len = seq_len
        self.n_features = n_features

        rng = np.random.default_rng(seed or hash(hospital_id) % 2**31)

        # Normal patients
        n_neg = int(n_patients * (1 - positive_rate))
        n_pos = n_patients - n_neg

        means_neg = np.array([75, 97, 16, 120, 36.8])
        stds_neg  = np.array([10,  2,  3,  15,   0.4])
        X_neg = rng.normal(loc=means_neg, scale=stds_neg,
                           size=(n_neg, seq_len, n_features)).astype(np.float32)

        # Deteriorating patients — drifting vitals
        means_pos = np.array([95, 91, 24, 90, 38.5])
        stds_pos  = np.array([15,  4,  5, 20,  0.8])
        X_pos = rng.normal(loc=means_pos, scale=stds_pos,
                           size=(n_pos, seq_len, n_features)).astype(np.float32)

        self.X = np.vstack([X_neg, X_pos])
        self.y = np.array([0]*n_neg + [1]*n_pos, dtype=np.float32).reshape(-1, 1)

        # Normalize features
        self.X = (self.X - means_neg) / (stds_neg + 1e-8)

        # Shuffle
        idx = rng.permutation(n_patients)
        self.X = self.X[idx]
        self.y = self.y[idx]

    def split(self, train_frac: float = 0.8) -> Tuple:
        n = int(len(self.X) * train_frac)
        return (self.X[:n], self.y[:n]), (self.X[n:], self.y[n:])

    def to_dataloader(self, X: np.ndarray, y: np.ndarray,
                      batch_size: int = 32, shuffle: bool = True) -> "DataLoader":
        if not _TORCH_AVAILABLE:
            return None
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Local trainer
# ---------------------------------------------------------------------------

class LocalTrainer:
    """
    Trains VitalsPredictionModel on local hospital data with optional DP-SGD.
    """

    def __init__(self,
                 hospital_id: str,
                 local_epochs: int = 3,
                 batch_size: int = 32,
                 learning_rate: float = 0.01,
                 dp_epsilon: Optional[float] = None,
                 dp_delta: float = 1e-5,
                 clip_norm: float = 1.0,
                 n_patients: int = 500,
                 device: str = "cpu"):
        self.hospital_id = hospital_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.clip_norm = clip_norm
        self.device = device

        self.dataset = SyntheticVitalsDataset(hospital_id, n_patients=n_patients)
        (X_tr, y_tr), (X_val, y_val) = self.dataset.split()
        self.val_data = (X_val, y_val)

        if _TORCH_AVAILABLE:
            self.model = VitalsPredictionModel().to(device)
            self.train_loader = self.dataset.to_dataloader(X_tr, y_tr, batch_size)
            self.val_loader   = self.dataset.to_dataloader(X_val, y_val, batch_size, shuffle=False)
            self.criterion = nn.BCELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self._setup_dp()
        else:
            self.model = None

        self.train_history: List[dict] = []

    def _setup_dp(self):
        """Attach Opacus PrivacyEngine if DP is requested."""
        if not _TORCH_AVAILABLE or not _OPACUS_AVAILABLE or self.dp_epsilon is None:
            if self.dp_epsilon:
                logger.warning("DP-SGD requested but Opacus not installed. Training without DP.")
            return
        # Validate and fix model for Opacus compatibility
        if not ModuleValidator.is_valid(self.model):
            self.model = ModuleValidator.fix(self.model)
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.local_epochs,
            target_epsilon=self.dp_epsilon,
            target_delta=self.dp_delta,
            max_grad_norm=self.clip_norm,
        )
        logger.info(
            f"[{self.hospital_id}] DP-SGD enabled: ε={self.dp_epsilon}, δ={self.dp_delta}, "
            f"clip_norm={self.clip_norm}"
        )

    def set_weights(self, weights: List[np.ndarray]):
        if _TORCH_AVAILABLE and self.model:
            state = OrderedDict(
                {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
            )
            self.model.load_state_dict(state, strict=True)

    def get_weights(self) -> List[np.ndarray]:
        if not _TORCH_AVAILABLE or not self.model:
            return []
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def train_round(self) -> dict:
        """Run local_epochs of local training. Returns metrics dict."""
        if not _TORCH_AVAILABLE or not self.model:
            # Simulation fallback
            sim_loss = 0.4 + np.random.uniform(-0.05, 0.05)
            sim_auc  = 0.72 + np.random.uniform(-0.02, 0.02)
            return {"loss": sim_loss, "auc": sim_auc, "examples": self.dataset.n_patients}

        self.model.train()
        total_loss = 0.0
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss / len(self.train_loader)

        avg_loss = total_loss / self.local_epochs
        auc = self.evaluate()
        metrics = {
            "hospital_id": self.hospital_id,
            "loss": round(avg_loss, 4),
            "auc": round(auc, 4),
            "examples": len(self.dataset.X),
            "dp_epsilon": self.dp_epsilon,
        }
        self.train_history.append(metrics)
        return metrics

    def evaluate(self) -> float:
        """Compute AUC on local validation set."""
        if not _TORCH_AVAILABLE or not self.model:
            return 0.75 + np.random.uniform(-0.05, 0.05)

        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            return 0.0

        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                pred = self.model(X_batch.to(self.device)).cpu().numpy()
                preds.extend(pred.flatten())
                targets.extend(y_batch.numpy().flatten())
        try:
            return float(roc_auc_score(targets, preds))
        except Exception:
            return 0.0

    @property
    def privacy_spent(self) -> Optional[dict]:
        if _OPACUS_AVAILABLE and hasattr(self, "privacy_engine"):
            eps = self.privacy_engine.get_epsilon(self.dp_delta)
            return {"epsilon_spent": eps, "delta": self.dp_delta}
        return None


# ---------------------------------------------------------------------------
# Flower Client wrapper
# ---------------------------------------------------------------------------

class IoMTFlowerClient(fl.client.NumPyClient if _FLOWER_AVAILABLE else object):

    def __init__(self, trainer: LocalTrainer):
        self.trainer = trainer

    def get_parameters(self, config) -> List[np.ndarray]:
        return self.trainer.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.trainer.set_weights(parameters)
        metrics = self.trainer.train_round()
        return self.trainer.get_weights(), metrics["examples"], metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        self.trainer.set_weights(parameters)
        auc = self.trainer.evaluate()
        return float(1 - auc), len(self.trainer.val_data[0]), {"auc": auc}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SPQR-IoMT Federated Learning Client")
    parser.add_argument("--hospital-id",   default="hospital_001")
    parser.add_argument("--server",        default="127.0.0.1:8080")
    parser.add_argument("--dp-epsilon",    type=float, default=1.0)
    parser.add_argument("--local-epochs",  type=int,   default=3)
    parser.add_argument("--n-patients",    type=int,   default=500)
    parser.add_argument("--simulate",      action="store_true")
    args = parser.parse_args()

    trainer = LocalTrainer(
        hospital_id=args.hospital_id,
        local_epochs=args.local_epochs,
        dp_epsilon=args.dp_epsilon if args.dp_epsilon > 0 else None,
        n_patients=args.n_patients,
    )

    if args.simulate or not _FLOWER_AVAILABLE:
        print(f"\n[{args.hospital_id}] Running local simulation ({args.local_epochs} rounds)...\n")
        for rnd in range(1, 6):
            metrics = trainer.train_round()
            print(f"  Round {rnd}: {metrics}")
        priv = trainer.privacy_spent
        if priv:
            print(f"\n  Privacy spent: {priv}")
    else:
        client = IoMTFlowerClient(trainer)
        fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
