"""
transformer_ids.py
==================
Lightweight Transformer-based Intrusion Detection System for IoMT networks.
An alternative to the BiLSTM-Attention IDS — uses a full self-attention
Transformer encoder which can capture longer-range dependencies in network
flow sequences or power traces.

Architecture:
  - Positional encoding (sinusoidal)
  - N × Transformer encoder blocks (multi-head self-attention + FFN)
  - Global average pooling
  - Classification head

Compared to BiLSTM: fewer parameters for short sequences (< 50 steps),
faster inference on CPU, no sequential dependency → parallelisable training.

Usage:
    from intrusion_detection.transformer_ids import TransformerIDS, TransformerIDSTrainer
    model = TransformerIDS(n_features=15, n_classes=8)
    trainer = TransformerIDSTrainer(model_type="network", epochs=15)
    trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
"""

import math
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False
    logger.warning("PyTorch not available — TransformerIDS will be non-functional.")

try:
    from sklearn.metrics import roc_auc_score
    _SK = True
except ImportError:
    _SK = False


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module if _TORCH else object):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        if not _TORCH:
            return
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module if _TORCH else object):
    """Single Transformer encoder layer: MHA + FFN + residual + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        if not _TORCH:
            return
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# Full Transformer IDS Model
# ---------------------------------------------------------------------------

class TransformerIDS(nn.Module if _TORCH else object):
    """
    Lightweight Transformer encoder for sequence classification.

    Input:  (batch, seq_len, n_features)
    Output: (batch, n_classes)

    Default config targets ~120K parameters — feasible on RPi 4 for inference.
    """

    def __init__(self,
                 n_features: int = 15,
                 n_classes:  int = 8,
                 d_model:    int = 64,
                 n_heads:    int = 4,
                 n_layers:   int = 3,
                 d_ff:       int = 128,
                 dropout:    float = 0.1,
                 max_len:    int = 256):
        if not _TORCH:
            return
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len, dropout)
        self.layers     = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, n_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)    # (B, T, d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        pooled = x.mean(dim=1)    # global average pool
        return self.classifier(pooled)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TransformerIDSTrainer:
    """Trainer wrapper for TransformerIDS — same API as IDSTrainer."""

    def __init__(self,
                 model_type:  str   = "network",
                 d_model:     int   = 64,
                 n_heads:     int   = 4,
                 n_layers:    int   = 3,
                 lr:          float = 1e-3,
                 batch_size:  int   = 64,
                 epochs:      int   = 20,
                 device:      str   = "cpu"):
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs     = epochs
        self.device     = device

        if model_type == "network":
            n_features, n_classes = 15, 8
        else:
            n_features, n_classes = 1, 4

        if _TORCH:
            self.model = TransformerIDS(
                n_features=n_features, n_classes=n_classes,
                d_model=d_model, n_heads=n_heads, n_layers=n_layers
            ).to(device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=lr, steps_per_epoch=100, epochs=epochs
            )
            self.criterion = nn.CrossEntropyLoss()
            logger.info(f"TransformerIDS: {self.model.n_parameters:,} parameters")
        else:
            self.model = None

        self.history: List[dict] = []

    def train(self, X: np.ndarray, y: np.ndarray) -> List[dict]:
        if not _TORCH or self.model is None:
            return [{"error": "PyTorch not available"}]

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        # Recreate scheduler with correct steps_per_epoch
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=1e-3,
            steps_per_epoch=len(loader), epochs=self.epochs
        )

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss, correct, total = 0.0, 0, 0
            for Xb, yb in loader:
                self.optimizer.zero_grad()
                out  = self.model(Xb)
                loss = self.criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                correct    += (out.argmax(1) == yb).sum().item()
                total      += len(yb)

            stats = {
                "epoch":    epoch,
                "loss":     round(total_loss / len(loader), 4),
                "accuracy": round(correct / total, 4),
            }
            self.history.append(stats)
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"  Epoch {epoch:3d}/{self.epochs} | {stats}")

        return self.history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not _TORCH or self.model is None:
            return {"error": "PyTorch not available"}
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)

        preds, probs, labels = [], [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for Xb, yb in loader:
                logits = self.model(Xb)
                pb = torch.softmax(logits, 1)
                preds.extend(logits.argmax(1).cpu().numpy())
                probs.extend(pb.cpu().numpy())
                labels.extend(yb.cpu().numpy())

        preds  = np.array(preds)
        probs  = np.array(probs)
        labels = np.array(labels)
        t_eval = time.perf_counter() - t0

        is_atk_true = (labels > 0).astype(int)
        is_atk_pred = (preds  > 0).astype(int)
        atk_prob    = 1 - probs[:, 0]

        TP = int(((is_atk_pred==1)&(is_atk_true==1)).sum())
        FP = int(((is_atk_pred==1)&(is_atk_true==0)).sum())
        FN = int(((is_atk_pred==0)&(is_atk_true==1)).sum())
        TN = int(((is_atk_pred==0)&(is_atk_true==0)).sum())

        tpr = TP / (TP + FN + 1e-8)
        fpr = FP / (FP + TN + 1e-8)
        pre = TP / (TP + FP + 1e-8)
        f1  = 2*pre*tpr / (pre + tpr + 1e-8)
        acc = float((preds == labels).mean())
        try:
            auc = float(roc_auc_score(is_atk_true, atk_prob)) if _SK else 0.0
        except Exception:
            auc = 0.0

        return {
            "model": "TransformerIDS",
            "n_parameters": self.model.n_parameters,
            "accuracy":  round(acc, 4),
            "tpr":       round(tpr, 4),
            "fpr":       round(fpr, 4),
            "precision": round(pre, 4),
            "f1_score":  round(f1,  4),
            "auc_roc":   round(auc, 4),
            "eval_time_s": round(t_eval, 3),
            "confusion_matrix": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
        }

    def save(self, path: str):
        if _TORCH and self.model:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": self.model.state_dict(), "history": self.history}, path)
            logger.info(f"Saved TransformerIDS to {path}")

    def compare_with_lstm(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Head-to-head comparison with BiLSTM on same data."""
        from intrusion_detection.lstm_ids import IDSTrainer
        n = int(len(X) * 0.8)
        self.train(X[:n], y[:n])
        tf_metrics = self.evaluate(X[n:], y[n:])

        lstm_trainer = IDSTrainer(model_type=self.model_type, epochs=self.epochs)
        lstm_trainer.train(X[:n], y[:n])
        lstm_metrics = lstm_trainer.evaluate(X[n:], y[n:])

        return {
            "TransformerIDS": tf_metrics,
            "BiLSTM-Attention": lstm_metrics,
            "winner_auc": "TransformerIDS" if tf_metrics.get("auc_roc", 0) > lstm_metrics.get("auc_roc", 0) else "BiLSTM-Attention",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== TransformerIDS Demo ===\n")
    from intrusion_detection.lstm_ids import NetworkFlowGenerator
    gen = NetworkFlowGenerator(seq_len=20)
    X, y = gen.generate(1000)
    n = 800
    trainer = TransformerIDSTrainer(model_type="network", epochs=5 if _TORCH else 1)
    trainer.train(X[:n], y[:n])
    metrics = trainer.evaluate(X[n:], y[n:])
    print(f"Metrics: {metrics}")
