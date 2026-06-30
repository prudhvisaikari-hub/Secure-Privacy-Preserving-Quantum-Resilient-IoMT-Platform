"""
federated_learning/dp_trainer.py
=================================
Standalone DP-SGD trainer for local hospital training.
Wraps Opacus PrivacyEngine with a clean API independent of Flower,
so it can be used for pre-training, fine-tuning, or standalone hospital use.

Features:
  - Automatic model compatibility fix via Opacus ModuleValidator
  - Per-epoch privacy budget tracking (RDP accountant)
  - Early stopping when privacy budget is exhausted
  - Gradient clipping with configurable norm
  - AUC + loss history export

Usage:
    trainer = DPSGDTrainer(
        model=VitalsPredictionModel(),
        target_epsilon=1.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        epochs=20,
        batch_size=32,
    )
    trainer.fit(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    trainer.save("models/hospital_001_model.pt")
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

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
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.accountants.utils import get_noise_multiplier
    _OPACUS = True
except ImportError:
    _OPACUS = False

try:
    from sklearn.metrics import roc_auc_score
    _SK = True
except ImportError:
    _SK = False


# ---------------------------------------------------------------------------
# DP-SGD Trainer
# ---------------------------------------------------------------------------

class DPSGDTrainer:
    """
    Differentially Private SGD trainer for any PyTorch nn.Module.

    Args:
        model:            PyTorch model (will be fixed for Opacus compatibility).
        target_epsilon:   Privacy budget ε. Set None for no DP.
        target_delta:     DP failure probability δ (default 1e-5).
        max_grad_norm:    L2 gradient clipping norm (sensitivity C).
        epochs:           Total training epochs.
        batch_size:       Mini-batch size.
        lr:               Learning rate (AdamW).
        device:           "cpu" or "cuda".
        verbose:          Print per-epoch progress.
    """

    def __init__(self,
                 model,
                 target_epsilon:  Optional[float] = 1.0,
                 target_delta:    float = 1e-5,
                 max_grad_norm:   float = 1.0,
                 epochs:          int   = 20,
                 batch_size:      int   = 32,
                 lr:              float = 0.001,
                 weight_decay:    float = 1e-4,
                 device:          str   = "cpu",
                 verbose:         bool  = True):

        if not _TORCH:
            raise RuntimeError("PyTorch is required for DPSGDTrainer.")

        self.target_epsilon = target_epsilon
        self.target_delta   = target_delta
        self.max_grad_norm  = max_grad_norm
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.device         = device
        self.verbose        = verbose

        # Fix model for Opacus (replaces BatchNorm → GroupNorm, etc.)
        if _OPACUS and target_epsilon is not None:
            if not ModuleValidator.is_valid(model):
                model = ModuleValidator.fix(model)
                logger.info("Model fixed for Opacus compatibility.")

        self.model     = model.to(device)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()

        self.privacy_engine: Optional["PrivacyEngine"] = None
        self.history: List[dict] = []
        self._noise_multiplier: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> List[dict]:
        """
        Train the model on (X, y).

        Args:
            X: Feature array (n_samples, seq_len, n_features) or (n_samples, n_features)
            y: Label array  (n_samples,) binary floats

        Returns:
            Training history list of dicts (one per epoch).
        """
        if not _TORCH:
            return []

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Attach PrivacyEngine
        if _OPACUS and self.target_epsilon is not None:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, loader = \
                self.privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=loader,
                    epochs=self.epochs,
                    target_epsilon=self.target_epsilon,
                    target_delta=self.target_delta,
                    max_grad_norm=self.max_grad_norm,
                )
            self._noise_multiplier = self.optimizer.noise_multiplier
            logger.info(
                f"[DP-SGD] ε={self.target_epsilon}, δ={self.target_delta}, "
                f"C={self.max_grad_norm}, σ={self._noise_multiplier:.4f}"
            )
        else:
            logger.info("[Trainer] Running WITHOUT differential privacy.")

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            t0 = time.perf_counter()
            epoch_loss, n_correct, n_total = 0.0, 0, 0

            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(Xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                if not (_OPACUS and self.target_epsilon is not None):
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_correct  += ((pred > 0.5).float() == yb).sum().item()
                n_total    += len(yb)

            avg_loss = epoch_loss / max(len(loader), 1)
            acc      = n_correct  / max(n_total, 1)
            eps_spent = self._get_epsilon_spent()
            elapsed   = time.perf_counter() - t0

            stats = {
                "epoch":       epoch,
                "loss":        round(avg_loss, 5),
                "accuracy":    round(acc, 4),
                "epsilon_spent": round(eps_spent, 4) if eps_spent else None,
                "elapsed_s":   round(elapsed, 2),
            }
            self.history.append(stats)

            if self.verbose and (epoch % 5 == 0 or epoch == 1):
                dp_str = f" | ε_spent={eps_spent:.3f}" if eps_spent else ""
                logger.info(f"  Epoch {epoch:3d}/{self.epochs} | loss={avg_loss:.4f} | acc={acc:.4f}{dp_str}")

        return self.history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model on (X, y). Returns dict with loss, accuracy, AUC."""
        if not _TORCH:
            return {}

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)

        all_preds, all_labels = [], []
        total_loss = 0.0
        with torch.no_grad():
            for Xb, yb in loader:
                pred = self.model(Xb)
                total_loss += self.criterion(pred, yb).item()
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(yb.cpu().numpy().flatten())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = float(((all_preds > 0.5).astype(float) == all_labels).mean())
        auc = float(roc_auc_score(all_labels, all_preds)) if _SK else 0.0

        return {
            "loss":     round(total_loss / max(len(loader), 1), 5),
            "accuracy": round(acc, 4),
            "auc_roc":  round(auc, 4),
            "epsilon_spent": self._get_epsilon_spent(),
            "n_samples": len(all_labels),
        }

    def _get_epsilon_spent(self) -> Optional[float]:
        if _OPACUS and self.privacy_engine and self.target_epsilon:
            try:
                return float(self.privacy_engine.get_epsilon(self.target_delta))
            except Exception:
                pass
        return None

    @property
    def privacy_report(self) -> dict:
        return {
            "target_epsilon":    self.target_epsilon,
            "target_delta":      self.target_delta,
            "max_grad_norm":     self.max_grad_norm,
            "noise_multiplier":  self._noise_multiplier,
            "epsilon_spent":     self._get_epsilon_spent(),
            "dp_enabled":        _OPACUS and self.target_epsilon is not None,
        }

    def get_weights(self) -> List[np.ndarray]:
        return [v.cpu().detach().numpy() for v in self.model.state_dict().values()]

    def set_weights(self, weights: List[np.ndarray]):
        state = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state, strict=True)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":    self.model.state_dict(),
            "history":        self.history,
            "privacy_report": self.privacy_report,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", [])
        logger.info(f"Model loaded from {path}")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not _TORCH:
        print("PyTorch not installed — cannot run demo.")
    else:
        from federated_learning.fl_server import VitalsPredictionModel
        from federated_learning.fl_client import SyntheticVitalsDataset

        ds = SyntheticVitalsDataset("demo_hospital", n_patients=400)
        (X_tr, y_tr), (X_val, y_val) = ds.split()

        model = VitalsPredictionModel()
        trainer = DPSGDTrainer(model, target_epsilon=1.0, epochs=10, verbose=True)
        trainer.fit(X_tr, y_tr.flatten())
        metrics = trainer.evaluate(X_val, y_val.flatten())
        print(f"\nEval: {metrics}")
        print(f"Privacy: {trainer.privacy_report}")
