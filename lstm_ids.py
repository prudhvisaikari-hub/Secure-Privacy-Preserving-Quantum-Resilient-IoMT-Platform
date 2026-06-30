"""
lstm_ids.py
===========
LSTM-based Intrusion Detection System (IDS) for IoMT networks.

Detects:
  1. Network-level anomalies (DDoS, MITM, replay attacks, unauthorized access)
  2. Side-channel anomalies (abnormal power traces during crypto operations)
  3. Behavioral deviations (unusual telemetry patterns from sensors)

Architecture:
  - BiLSTM encoder with attention pooling
  - Multi-head classification: normal / intrusion_type
  - Configurable input for network flow features OR power trace features

Datasets supported:
  - UNSW-NB15 (network intrusion)
  - IoT-23 (IoT-specific attacks)
  - Custom testbed telemetry + power traces

Usage:
    trainer = IDSTrainer(model_type="network")
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    trainer.save("intrusion_detection/models/lstm_ids.pt")
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
    logger.warning("PyTorch not available — IDS will use sklearn fallback.")

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (roc_auc_score, classification_report,
                                  confusion_matrix, roc_curve)
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Attack categories
# ---------------------------------------------------------------------------

NETWORK_ATTACK_CLASSES = {
    0: "Normal",
    1: "DoS/DDoS",
    2: "Reconnaissance",
    3: "Backdoor",
    4: "Fuzzing",
    5: "MITM",
    6: "Injection",
    7: "Password",
}

SIDE_CHANNEL_CLASSES = {
    0: "Normal_Crypto",
    1: "Power_Glitch",
    2: "Timing_Attack",
    3: "Fault_Injection",
}


# ---------------------------------------------------------------------------
# Synthetic dataset generators
# ---------------------------------------------------------------------------

class NetworkFlowGenerator:
    """
    Generates synthetic network flow features mimicking UNSW-NB15 / IoT-23.

    Features (15-dim):
      dur, proto_tcp, proto_udp, proto_icmp,
      sbytes, dbytes, rate, sload, dload,
      spkts, dpkts, sttl, dttl, sinpkt, dinpkt
    """
    N_FEATURES = 15
    N_CLASSES = len(NETWORK_ATTACK_CLASSES)

    def __init__(self, seq_len: int = 20, seed: int = 42):
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def _normal_flow(self) -> np.ndarray:
        """Normal medical device telemetry flow."""
        features = self.rng.exponential(scale=1.0, size=(self.seq_len, self.N_FEATURES))
        features[:, 0] *= 0.5    # short duration
        features[:, 4:6] *= 100  # moderate byte counts
        features[:, 6] = self.rng.normal(10, 2, self.seq_len)  # stable rate
        return features.astype(np.float32)

    def _dos_flow(self) -> np.ndarray:
        """DoS: high packet rate, small payload, many source pkts."""
        f = self._normal_flow()
        f[:, 6]  = self.rng.normal(500, 50, self.seq_len)   # very high rate
        f[:, 9]  = self.rng.normal(1000, 100, self.seq_len)  # high spkts
        f[:, 10] = self.rng.normal(5, 2, self.seq_len)       # low dpkts
        f[:, 4]  = self.rng.normal(50, 10, self.seq_len)     # small sbytes
        return f

    def _mitm_flow(self) -> np.ndarray:
        """MITM: bidirectional anomaly, unusual TTL."""
        f = self._normal_flow()
        f[:, 11] = self.rng.normal(200, 5, self.seq_len)   # unusual sttl
        f[:, 12] = self.rng.normal(200, 5, self.seq_len)   # unusual dttl
        f[:, 4]  *= 2.5  # inflated byte count (interception overhead)
        return f

    def _recon_flow(self) -> np.ndarray:
        """Reconnaissance: many short connections, low payload."""
        f = self._normal_flow()
        f[:, 0] = self.rng.exponential(0.01, self.seq_len)   # very short duration
        f[:, 4] = self.rng.normal(30, 5, self.seq_len)       # tiny sbytes
        f[:, 5] = self.rng.normal(10, 3, self.seq_len)       # tiny dbytes
        return f

    def generate(self, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        generators = {
            0: self._normal_flow,
            1: self._dos_flow,
            2: self._recon_flow,
            5: self._mitm_flow,
        }
        # Imbalanced: 60% normal, 40% attacks
        n_normal = int(n_samples * 0.60)
        n_attack = n_samples - n_normal

        X, y = [], []
        for _ in range(n_normal):
            X.append(self._normal_flow())
            y.append(0)

        attack_classes = [1, 2, 5]
        for i in range(n_attack):
            cls = attack_classes[i % len(attack_classes)]
            X.append(generators[cls]())
            y.append(cls)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        # Normalize
        flat = X.reshape(-1, self.N_FEATURES)
        self.scaler_mean = flat.mean(axis=0)
        self.scaler_std  = flat.std(axis=0) + 1e-8
        X = (X - self.scaler_mean) / self.scaler_std

        # Shuffle
        idx = self.rng.permutation(n_samples)
        return X[idx], y[idx]


class PowerTraceGenerator:
    """
    Generates synthetic power traces for side-channel analysis.

    Simulates power consumption of a microcontroller (e.g., STM32) during
    AES/Kyber operations. Normal traces vs attack-induced anomalies.
    """
    N_FEATURES = 1   # raw power sample (scalar per timestep)

    def __init__(self, trace_len: int = 256, seed: int = 42):
        self.trace_len = trace_len
        self.rng = np.random.default_rng(seed)

    def _normal_trace(self) -> np.ndarray:
        """Typical Kyber keygen power trace."""
        t = np.linspace(0, 4*np.pi, self.trace_len)
        trace = (
            0.8 * np.sin(t) + 0.4 * np.sin(3*t)
            + self.rng.normal(0, 0.05, self.trace_len)
        )
        return trace.astype(np.float32)

    def _power_glitch(self) -> np.ndarray:
        """Voltage glitch attack — sharp spike at random position."""
        trace = self._normal_trace()
        glitch_pos = self.rng.integers(10, self.trace_len - 10)
        trace[glitch_pos:glitch_pos+3] += self.rng.uniform(2.0, 5.0)
        return trace

    def _timing_attack(self) -> np.ndarray:
        """Timing attack — shifted phase in power trace."""
        trace = self._normal_trace()
        shift = self.rng.integers(5, 20)
        return np.roll(trace, shift)

    def _fault_injection(self) -> np.ndarray:
        """Fault injection — zero-out segment of trace."""
        trace = self._normal_trace()
        start = self.rng.integers(50, self.trace_len - 30)
        trace[start:start+15] = 0.0
        return trace

    def generate(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        generators = {
            0: self._normal_trace,
            1: self._power_glitch,
            2: self._timing_attack,
            3: self._fault_injection,
        }
        n_normal = int(n_samples * 0.70)
        n_attack = n_samples - n_normal

        X, y = [], []
        for _ in range(n_normal):
            X.append(self._normal_trace().reshape(-1, 1))
            y.append(0)
        for i in range(n_attack):
            cls = (i % 3) + 1
            X.append(generators[cls]().reshape(-1, 1))
            y.append(cls)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        idx = np.random.permutation(n_samples)
        return X[idx], y[idx]


# ---------------------------------------------------------------------------
# BiLSTM-Attention IDS Model
# ---------------------------------------------------------------------------

class BiLSTMAttentionIDS(nn.Module if _TORCH_AVAILABLE else object):
    """
    Bidirectional LSTM with multi-head self-attention for IDS.

    Input:  (batch, seq_len, n_features)
    Output: (batch, n_classes) — logits
    """

    def __init__(self, n_features: int = 15, n_classes: int = 8,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=8, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        lstm_out, _ = self.lstm(x)                          # (B, T, 2H)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (B, T, 2H)
        out = self.norm(attn_out + lstm_out)                # residual + norm
        pooled = out.mean(dim=1)                            # (B, 2H) global avg pool
        return self.classifier(pooled)                      # (B, n_classes)


# ---------------------------------------------------------------------------
# IDS Trainer
# ---------------------------------------------------------------------------

class IDSTrainer:

    def __init__(self,
                 model_type: str = "network",  # "network" or "sidechannel"
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 20,
                 device: str = "cpu"):
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        if model_type == "network":
            n_features = NetworkFlowGenerator.N_FEATURES
            n_classes  = len(NETWORK_ATTACK_CLASSES)
            self.class_names = NETWORK_ATTACK_CLASSES
        else:
            n_features = 1
            n_classes  = len(SIDE_CHANNEL_CLASSES)
            self.class_names = SIDE_CHANNEL_CLASSES

        if _TORCH_AVAILABLE:
            self.model = BiLSTMAttentionIDS(
                n_features=n_features, n_classes=n_classes,
                hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
            ).to(device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.model = None
            if _SKLEARN_AVAILABLE:
                self.sklearn_model = RandomForestClassifier(n_estimators=100, random_state=42)

        self.history: List[dict] = []

    def train(self, X: np.ndarray, y: np.ndarray) -> List[dict]:
        if not _TORCH_AVAILABLE:
            return self._train_sklearn(X, y)

        # Create DataLoader
        X_t = torch.tensor(X).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0
            t_start = time.perf_counter()

            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

            self.scheduler.step()
            epoch_time = time.perf_counter() - t_start
            acc = correct / total
            avg_loss = epoch_loss / len(loader)

            stats = {
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                "accuracy": round(acc, 4),
                "epoch_time_s": round(epoch_time, 2),
            }
            self.history.append(stats)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"  Epoch {epoch:3d}/{self.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

        return self.history

    def _train_sklearn(self, X: np.ndarray, y: np.ndarray) -> List[dict]:
        """Fallback to sklearn RandomForest when PyTorch unavailable."""
        X_flat = X.reshape(len(X), -1)
        self.sklearn_model.fit(X_flat, y)
        acc = self.sklearn_model.score(X_flat, y)
        stats = [{"epoch": 1, "accuracy": round(acc, 4), "model": "RandomForest"}]
        self.history = stats
        return stats

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        if _TORCH_AVAILABLE and self.model is not None:
            return self._evaluate_torch(X, y)
        elif _SKLEARN_AVAILABLE:
            return self._evaluate_sklearn(X, y)
        return {"error": "No ML framework available"}

    def _evaluate_torch(self, X: np.ndarray, y: np.ndarray) -> dict:
        self.model.eval()
        X_t = torch.tensor(X).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)

        all_preds, all_probs, all_labels = [], [], []
        t0 = time.perf_counter()

        with torch.no_grad():
            for X_b, y_b in loader:
                logits = self.model(X_b)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_b.cpu().numpy())

        eval_time = time.perf_counter() - t0
        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)

        # Per-class accuracy
        correct = (all_preds == all_labels).mean()

        # Detection-specific: normal=0, attack=1+
        is_attack_true = (all_labels > 0).astype(int)
        is_attack_pred = (all_preds  > 0).astype(int)
        attack_prob    = 1 - all_probs[:, 0]  # P(attack)

        TP = ((is_attack_pred == 1) & (is_attack_true == 1)).sum()
        FP = ((is_attack_pred == 1) & (is_attack_true == 0)).sum()
        FN = ((is_attack_pred == 0) & (is_attack_true == 1)).sum()
        TN = ((is_attack_pred == 0) & (is_attack_true == 0)).sum()

        tpr = TP / (TP + FN + 1e-8)
        fpr = FP / (FP + TN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        f1 = 2 * precision * tpr / (precision + tpr + 1e-8)

        try:
            auc = roc_auc_score(is_attack_true, attack_prob)
        except Exception:
            auc = 0.0

        metrics = {
            "overall_accuracy": round(float(correct), 4),
            "tpr_detection_rate": round(float(tpr), 4),
            "fpr": round(float(fpr), 4),
            "precision": round(float(precision), 4),
            "f1_score": round(float(f1), 4),
            "auc_roc": round(float(auc), 4),
            "eval_time_s": round(eval_time, 3),
            "n_samples": len(all_labels),
            "confusion_matrix": {
                "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN)
            }
        }
        logger.info(
            f"[IDS Eval] Acc={correct:.4f} | TPR={tpr:.4f} | FPR={fpr:.4f} | AUC={auc:.4f}"
        )
        return metrics

    def _evaluate_sklearn(self, X: np.ndarray, y: np.ndarray) -> dict:
        X_flat = X.reshape(len(X), -1)
        preds = self.sklearn_model.predict(X_flat)
        acc = float((preds == y).mean())
        is_attack_true = (y > 0).astype(int)
        is_attack_pred = (preds > 0).astype(int)
        TP = int(((is_attack_pred == 1) & (is_attack_true == 1)).sum())
        FP = int(((is_attack_pred == 1) & (is_attack_true == 0)).sum())
        FN = int(((is_attack_pred == 0) & (is_attack_true == 1)).sum())
        TN = int(((is_attack_pred == 0) & (is_attack_true == 0)).sum())
        return {
            "overall_accuracy": round(acc, 4),
            "confusion_matrix": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
            "model": "RandomForest",
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if _TORCH_AVAILABLE and self.model:
            torch.save({
                "model_state": self.model.state_dict(),
                "history": self.history,
                "model_type": self.model_type,
                "class_names": self.class_names,
            }, path)
            logger.info(f"Model saved to {path}")

    def load(self, path: str):
        if _TORCH_AVAILABLE:
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.history = ckpt.get("history", [])
            logger.info(f"Model loaded from {path}")

    def real_time_detect(self, flow: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Real-time detection for a single network flow window.
        Returns detection result with confidence.
        """
        if not _TORCH_AVAILABLE or self.model is None:
            return {"attack_detected": False, "confidence": 0.0}

        self.model.eval()
        x = torch.tensor(flow[np.newaxis, :, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class = int(probs.argmax())
        attack_prob = float(1 - probs[0])

        return {
            "attack_detected": attack_prob >= threshold,
            "predicted_class": self.class_names.get(pred_class, str(pred_class)),
            "attack_probability": round(attack_prob, 4),
            "class_probabilities": {
                self.class_names.get(i, str(i)): round(float(p), 4)
                for i, p in enumerate(probs)
            },
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== SPQR-IoMT Intrusion Detection System Demo ===\n")

    # Network IDS
    print("[1] Network IDS (BiLSTM-Attention)")
    gen = NetworkFlowGenerator(seq_len=20)
    X, y = gen.generate(n_samples=2000)
    n = int(len(X) * 0.8)
    trainer = IDSTrainer(model_type="network", epochs=10 if _TORCH_AVAILABLE else 1)
    trainer.train(X[:n], y[:n])
    metrics = trainer.evaluate(X[n:], y[n:])
    print(f"  Metrics: {metrics}\n")

    # Side-channel IDS
    print("[2] Side-Channel IDS (power traces)")
    sc_gen = PowerTraceGenerator(trace_len=256)
    X_sc, y_sc = sc_gen.generate(n_samples=1000)
    n_sc = int(len(X_sc) * 0.8)
    sc_trainer = IDSTrainer(model_type="sidechannel", epochs=5 if _TORCH_AVAILABLE else 1)
    sc_trainer.train(X_sc[:n_sc], y_sc[:n_sc])
    sc_metrics = sc_trainer.evaluate(X_sc[n_sc:], y_sc[n_sc:])
    print(f"  Metrics: {sc_metrics}\n")

    # Real-time detection demo
    print("[3] Real-time detection demo")
    sample_dos = gen._dos_flow()
    result = trainer.real_time_detect(sample_dos)
    print(f"  DoS sample → {result}")
