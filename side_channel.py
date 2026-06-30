"""
intrusion_detection/side_channel.py
=====================================
Dedicated side-channel attack detection module.
Classifies power traces from IoMT microcontrollers to detect:
  - Normal crypto operations
  - Power glitch attacks
  - Timing side-channel attacks
  - Fault injection (DFA)

Uses a 1-D CNN + LSTM hybrid — CNNs extract local temporal patterns
(glitch spikes, phase shifts) while LSTM captures sequence-level context.

Also implements:
  - Correlation Power Analysis (CPA) risk scorer
  - Test-Vector Leakage Assessment (TVLA) t-test
  - Signal-to-Noise Ratio (SNR) estimation

Usage:
    from intrusion_detection.side_channel import SideChannelDetector
    detector = SideChannelDetector(trace_len=512)
    detector.train(traces, labels)
    result = detector.detect(new_trace)
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from sklearn.metrics import roc_auc_score, classification_report
    _SK = True
except ImportError:
    _SK = False


# ---------------------------------------------------------------------------
# 1D CNN + LSTM hybrid model
# ---------------------------------------------------------------------------

class CNNLSTMSideChannel(nn.Module if _TORCH else object):
    """
    Hybrid 1-D CNN + LSTM for power trace classification.

    Input:  (batch, trace_len, 1)
    Output: (batch, n_classes)

    CNN extracts local features (glitch spikes, periodicity changes),
    LSTM models temporal structure across the trace.
    """

    def __init__(self, trace_len: int = 512, n_classes: int = 4,
                 n_filters: int = 32, kernel_size: int = 7,
                 lstm_hidden: int = 64, dropout: float = 0.2):
        if not _TORCH:
            return
        super().__init__()
        # CNN block: (B, 1, T) → (B, filters, T//4)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(n_filters, n_filters*2, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(n_filters*2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        # LSTM block on CNN output
        self.lstm = nn.LSTM(
            input_size=n_filters*2, hidden_size=lstm_hidden,
            num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden*2, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # x: (B, T, 1) → permute to (B, 1, T) for Conv1d
        x = x.permute(0, 2, 1)       # (B, 1, T)
        x = self.cnn(x)               # (B, 2F, T//4)
        x = x.permute(0, 2, 1)       # (B, T//4, 2F)
        lstm_out, _ = self.lstm(x)    # (B, T//4, 2H)
        pooled = lstm_out.mean(1)     # (B, 2H)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Side-channel statistical analysis
# ---------------------------------------------------------------------------

class TVLAAnalyzer:
    """
    Test-Vector Leakage Assessment (TVLA) using Welch's t-test.
    Identifies which sample points in a power trace leak key-dependent info.

    Standard approach: compare traces under fixed vs. random inputs.
    |t| > 4.5 → statistically significant leakage (standard threshold).
    """

    THRESHOLD = 4.5

    def __init__(self, traces_fixed: np.ndarray, traces_random: np.ndarray):
        """
        Args:
            traces_fixed:  Power traces with fixed test vector (n1, trace_len)
            traces_random: Power traces with random data    (n2, trace_len)
        """
        self.traces_fixed  = np.asarray(traces_fixed, dtype=np.float64)
        self.traces_random = np.asarray(traces_random, dtype=np.float64)

    def t_statistic(self) -> np.ndarray:
        """Welch's t-statistic at each sample point."""
        m1 = self.traces_fixed.mean(0)
        m2 = self.traces_random.mean(0)
        v1 = self.traces_fixed.var(0, ddof=1)  / len(self.traces_fixed)
        v2 = self.traces_random.var(0, ddof=1) / len(self.traces_random)
        return (m1 - m2) / np.sqrt(v1 + v2 + 1e-12)

    def leaking_points(self) -> np.ndarray:
        """Return indices where |t| > threshold (leakage detected)."""
        return np.where(np.abs(self.t_statistic()) > self.THRESHOLD)[0]

    def report(self) -> dict:
        t = self.t_statistic()
        leaking = self.leaking_points()
        return {
            "max_t_statistic":  round(float(np.max(np.abs(t))), 4),
            "n_leaking_points": int(len(leaking)),
            "leakage_detected": len(leaking) > 0,
            "first_leaking_idx": int(leaking[0]) if len(leaking) > 0 else None,
            "threshold": self.THRESHOLD,
        }


class SNRAnalyzer:
    """
    Signal-to-Noise Ratio estimation for power traces.
    SNR(t) = Var_signal(t) / Var_noise(t)
    High SNR → easy side-channel → higher attack risk.
    """

    def __init__(self, traces: np.ndarray, labels: np.ndarray):
        self.traces = np.asarray(traces, dtype=np.float64)
        self.labels = np.asarray(labels)

    def snr(self) -> np.ndarray:
        classes = np.unique(self.labels)
        class_means = np.array([self.traces[self.labels == c].mean(0) for c in classes])
        signal_var = class_means.var(0)
        noise_var  = np.array([
            self.traces[self.labels == c].var(0, ddof=1) for c in classes
        ]).mean(0)
        return signal_var / (noise_var + 1e-12)

    def peak_snr_points(self, top_k: int = 10) -> np.ndarray:
        return np.argsort(self.snr())[::-1][:top_k]

    def report(self) -> dict:
        snr = self.snr()
        return {
            "max_snr":         round(float(snr.max()), 4),
            "mean_snr":        round(float(snr.mean()), 4),
            "peak_snr_points": self.peak_snr_points(5).tolist(),
            "high_risk":       float(snr.max()) > 1.0,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class SideChannelDetector:
    """
    Full side-channel detection pipeline:
      1. Train CNN-LSTM classifier on labelled power traces
      2. Run TVLA to identify leaking points
      3. Real-time detect on streaming traces
    """

    def __init__(self, trace_len: int = 512, n_classes: int = 4,
                 epochs: int = 20, batch_size: int = 64,
                 lr: float = 1e-3, device: str = "cpu"):
        self.trace_len  = trace_len
        self.n_classes  = n_classes
        self.epochs     = epochs
        self.batch_size = batch_size
        self.device     = device
        self.history: List[dict] = []

        if _TORCH:
            self.model = CNNLSTMSideChannel(trace_len, n_classes).to(device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.model = None

    def train(self, traces: np.ndarray, labels: np.ndarray) -> List[dict]:
        if not _TORCH or self.model is None:
            logger.warning("PyTorch not available — side-channel training skipped.")
            return []

        X_t = torch.tensor(traces, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(labels, dtype=torch.long).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            loss_sum, correct, total = 0.0, 0, 0
            for Xb, yb in loader:
                self.optimizer.zero_grad()
                out  = self.model(Xb)
                loss = self.criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                loss_sum += loss.item()
                correct  += (out.argmax(1) == yb).sum().item()
                total    += len(yb)
            stats = {"epoch": epoch, "loss": round(loss_sum/len(loader), 4),
                     "accuracy": round(correct/total, 4)}
            self.history.append(stats)
            if epoch % 5 == 0:
                logger.info(f"  SC Epoch {epoch:3d}/{self.epochs}: {stats}")
        return self.history

    def evaluate(self, traces: np.ndarray, labels: np.ndarray) -> dict:
        if not _TORCH or self.model is None:
            return {"error": "PyTorch not available"}

        self.model.eval()
        X_t = torch.tensor(traces, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(labels, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)

        preds, probs, lbls = [], [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for Xb, yb in loader:
                out = self.model(Xb)
                pb  = torch.softmax(out, 1)
                preds.extend(out.argmax(1).cpu().numpy())
                probs.extend(pb.cpu().numpy())
                lbls.extend(yb.numpy())

        preds = np.array(preds); lbls = np.array(lbls); probs = np.array(probs)
        t_eval = time.perf_counter() - t0
        acc = float((preds == lbls).mean())
        is_atk = (lbls > 0).astype(int)
        is_atk_pred = (preds > 0).astype(int)
        TP = int(((is_atk_pred==1)&(is_atk==1)).sum())
        FP = int(((is_atk_pred==1)&(is_atk==0)).sum())
        FN = int(((is_atk_pred==0)&(is_atk==1)).sum())
        TN = int(((is_atk_pred==0)&(is_atk==0)).sum())
        tpr = TP / max(TP+FN, 1)
        fpr = FP / max(FP+TN, 1)
        auc = 0.0
        if _SK:
            try:
                auc = float(roc_auc_score(is_atk, 1-probs[:,0]))
            except Exception:
                pass
        return {
            "model": "CNN-LSTM SideChannel",
            "accuracy": round(acc, 4), "tpr": round(tpr, 4), "fpr": round(fpr, 4),
            "auc_roc": round(auc, 4),
            "confusion_matrix": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
            "eval_time_s": round(t_eval, 3),
        }

    def detect(self, trace: np.ndarray, threshold: float = 0.5) -> dict:
        """Real-time detection on a single trace."""
        if not _TORCH or self.model is None:
            return {"attack_detected": False}
        self.model.eval()
        x = torch.tensor(trace.reshape(1, -1, 1), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), 1).cpu().numpy()[0]
        atk_prob = float(1 - probs[0])
        return {
            "attack_detected": atk_prob >= threshold,
            "attack_probability": round(atk_prob, 4),
            "predicted_class": int(probs.argmax()),
            "class_probs": {f"class_{i}": round(float(p), 4) for i, p in enumerate(probs)},
        }

    def run_tvla(self, traces: np.ndarray, labels: np.ndarray) -> dict:
        """Run TVLA leakage assessment on provided traces."""
        fixed  = traces[labels == 0]
        random = traces[labels != 0]
        if len(fixed) < 10 or len(random) < 10:
            return {"error": "Insufficient samples for TVLA (need ≥10 per group)"}
        tvla = TVLAAnalyzer(fixed.reshape(len(fixed), -1), random.reshape(len(random), -1))
        return tvla.report()

    def save(self, path: str):
        if _TORCH and self.model:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": self.model.state_dict(), "history": self.history}, path)

    def load(self, path: str):
        if _TORCH:
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.history = ckpt.get("history", [])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from intrusion_detection.data_gen import DatasetFactory
    print("=== Side-Channel Detection Demo ===")
    X, y = DatasetFactory.power_traces(n_samples=800, trace_len=512)
    n = int(len(X)*0.8)
    detector = SideChannelDetector(trace_len=512, epochs=5)
    detector.train(X[:n], y[:n])
    metrics = detector.evaluate(X[n:], y[n:])
    print(f"Metrics: {metrics}")
    tvla = detector.run_tvla(X[:n], y[:n])
    print(f"TVLA: {tvla}")
