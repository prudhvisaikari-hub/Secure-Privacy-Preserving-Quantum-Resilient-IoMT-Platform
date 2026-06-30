"""
utils/metrics.py
================
Shared evaluation metrics used across all SPQR-IoMT experiments.
Provides unified computation of classification, detection, privacy,
and cryptographic performance metrics.

Usage:
    from utils.metrics import ClassificationMetrics, PrivacyMetrics, CryptoMetrics
    cm = ClassificationMetrics(y_true, y_pred, y_prob)
    print(cm.report())
"""

import time
import json
import math
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, confusion_matrix,
        classification_report, roc_curve, precision_recall_curve
    )
    _SK = True
except ImportError:
    _SK = False


# ---------------------------------------------------------------------------
# Classification / Detection Metrics
# ---------------------------------------------------------------------------

class ClassificationMetrics:
    """
    Binary and multi-class classification metrics for IDS / vitals prediction.

    Args:
        y_true:  Ground truth labels (int array)
        y_pred:  Predicted labels (int array)
        y_prob:  Predicted probabilities for positive class or all classes (float array)
        positive_label: Which label counts as "attack" / "positive" (default 1 for binary, >0 for multi)
    """

    def __init__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 y_prob: Optional[np.ndarray] = None,
                 positive_label: int = 1):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_prob = np.asarray(y_prob) if y_prob is not None else None
        self.positive_label = positive_label

        # Binary attack detection (any non-zero class = attack)
        self._is_atk_true = (self.y_true != 0).astype(int)
        self._is_atk_pred = (self.y_pred != 0).astype(int)

    # ---------- Core binary metrics ----------

    @property
    def accuracy(self) -> float:
        return float((self.y_pred == self.y_true).mean())

    @property
    def confusion(self) -> Dict[str, int]:
        TP = int(((self._is_atk_pred == 1) & (self._is_atk_true == 1)).sum())
        FP = int(((self._is_atk_pred == 1) & (self._is_atk_true == 0)).sum())
        FN = int(((self._is_atk_pred == 0) & (self._is_atk_true == 1)).sum())
        TN = int(((self._is_atk_pred == 0) & (self._is_atk_true == 0)).sum())
        return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}

    @property
    def tpr(self) -> float:
        c = self.confusion
        return c["TP"] / max(c["TP"] + c["FN"], 1)

    @property
    def fpr(self) -> float:
        c = self.confusion
        return c["FP"] / max(c["FP"] + c["TN"], 1)

    @property
    def precision(self) -> float:
        c = self.confusion
        return c["TP"] / max(c["TP"] + c["FP"], 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.tpr
        return 2 * p * r / max(p + r, 1e-8)

    @property
    def auc_roc(self) -> float:
        if not _SK or self.y_prob is None:
            return 0.0
        try:
            prob_pos = self.y_prob if self.y_prob.ndim == 1 else 1 - self.y_prob[:, 0]
            return float(roc_auc_score(self._is_atk_true, prob_pos))
        except Exception:
            return 0.0

    @property
    def average_precision(self) -> float:
        if not _SK or self.y_prob is None:
            return 0.0
        try:
            prob_pos = self.y_prob if self.y_prob.ndim == 1 else 1 - self.y_prob[:, 0]
            return float(average_precision_score(self._is_atk_true, prob_pos))
        except Exception:
            return 0.0

    def report(self) -> Dict[str, Any]:
        return {
            "accuracy":          round(self.accuracy, 4),
            "tpr_recall":        round(self.tpr, 4),
            "fpr":               round(self.fpr, 4),
            "precision":         round(self.precision, 4),
            "f1_score":          round(self.f1, 4),
            "auc_roc":           round(self.auc_roc, 4),
            "average_precision": round(self.average_precision, 4),
            "confusion_matrix":  self.confusion,
            "n_samples":         len(self.y_true),
        }

    def roc_points(self, n_thresholds: int = 100) -> Tuple[List[float], List[float]]:
        """Returns (fpr_list, tpr_list) for plotting ROC curve."""
        if not _SK or self.y_prob is None:
            return [0.0, 1.0], [0.0, 1.0]
        prob_pos = self.y_prob if self.y_prob.ndim == 1 else 1 - self.y_prob[:, 0]
        try:
            fpr, tpr, _ = roc_curve(self._is_atk_true, prob_pos)
            return fpr.tolist(), tpr.tolist()
        except Exception:
            return [0.0, 1.0], [0.0, 1.0]

    def optimal_threshold(self) -> float:
        """Youden's J statistic: threshold maximising TPR - FPR."""
        if not _SK or self.y_prob is None:
            return 0.5
        prob_pos = self.y_prob if self.y_prob.ndim == 1 else 1 - self.y_prob[:, 0]
        try:
            fpr, tpr, thresholds = roc_curve(self._is_atk_true, prob_pos)
            j_scores = tpr - fpr
            return float(thresholds[np.argmax(j_scores)])
        except Exception:
            return 0.5


# ---------------------------------------------------------------------------
# Privacy Metrics
# ---------------------------------------------------------------------------

class PrivacyMetrics:
    """
    Metrics for differential privacy and membership inference risk.
    """

    @staticmethod
    def dp_utility_gap(auc_no_dp: float, auc_with_dp: float) -> dict:
        """Utility cost of adding DP."""
        gap = auc_no_dp - auc_with_dp
        rel_gap = gap / max(auc_no_dp, 1e-8)
        return {
            "auc_no_dp":    round(auc_no_dp, 4),
            "auc_with_dp":  round(auc_with_dp, 4),
            "absolute_gap": round(gap, 4),
            "relative_gap": round(rel_gap, 4),
            "acceptable":   gap < 0.05,  # <5% drop is generally acceptable
        }

    @staticmethod
    def membership_inference_risk(
        train_preds: np.ndarray,
        test_preds:  np.ndarray,
        train_labels: np.ndarray,
        test_labels:  np.ndarray,
    ) -> dict:
        """
        Estimate membership inference attack success.
        Higher confidence on training vs. test data indicates memorisation.
        MI risk ~ AUC of a classifier distinguishing train vs. test confidence.

        Args:
            train_preds:  Model confidence on training samples
            test_preds:   Model confidence on test samples (same distribution)
            train_labels: True labels for training samples
            test_labels:  True labels for test samples
        """
        # Loss-based MI: lower loss on training = higher MI risk
        eps = 1e-8
        train_loss = -np.mean(
            train_labels * np.log(train_preds + eps) + (1 - train_labels) * np.log(1 - train_preds + eps)
        )
        test_loss  = -np.mean(
            test_labels  * np.log(test_preds  + eps) + (1 - test_labels)  * np.log(1 - test_preds  + eps)
        )

        # MI advantage: probability attacker correctly identifies member
        # (simplified: threshold-based using train/test confidence scores)
        all_scores  = np.concatenate([train_preds, test_preds])
        all_members = np.concatenate([np.ones(len(train_preds)), np.zeros(len(test_preds))])

        mi_auc = 0.5
        if _SK:
            try:
                mi_auc = float(roc_auc_score(all_members, all_scores))
            except Exception:
                pass

        return {
            "train_loss":     round(float(train_loss), 4),
            "test_loss":      round(float(test_loss), 4),
            "generalisation_gap": round(float(test_loss - train_loss), 4),
            "mi_auc":         round(mi_auc, 4),
            "mi_advantage":   round(2 * abs(mi_auc - 0.5), 4),
            "risk_level":     "Low" if mi_auc < 0.6 else ("Medium" if mi_auc < 0.75 else "High"),
        }

    @staticmethod
    def privacy_utility_pareto(
        epsilon_values: List[float],
        auc_values:     List[float],
    ) -> List[dict]:
        """Return Pareto-efficient (ε, AUC) points."""
        points = sorted(zip(epsilon_values, auc_values), key=lambda x: x[0])
        pareto, best_auc = [], -1.0
        for eps, auc in points:
            if auc > best_auc:
                best_auc = auc
                pareto.append({"epsilon": eps, "auc": auc, "pareto_optimal": True})
            else:
                pareto.append({"epsilon": eps, "auc": auc, "pareto_optimal": False})
        return pareto


# ---------------------------------------------------------------------------
# Crypto Performance Metrics
# ---------------------------------------------------------------------------

class CryptoMetrics:
    """Aggregate and compare cryptographic operation timing results."""

    def __init__(self, results: List[Dict]):
        """
        Args:
            results: List of benchmark dicts with keys:
                     variant, keygen_ms, encaps_ms, decaps_ms, total_ms,
                     pk_bytes, sk_bytes, ct_bytes
        """
        self.results = results

    def speedup_vs(self, baseline_variant: str = "RSA-2048") -> List[dict]:
        """Compute speedup ratio of each scheme vs the baseline."""
        baseline = next((r for r in self.results if r.get("variant") == baseline_variant), None)
        if not baseline:
            return self.results
        out = []
        for r in self.results:
            r2 = dict(r)
            base_total = baseline.get("total_ms", 1) or 1
            r2["speedup_vs_baseline"] = round(base_total / max(r.get("total_ms", 1e-9), 1e-9), 2)
            r2["baseline"] = baseline_variant
            out.append(r2)
        return out

    def energy_estimate_uj(self, current_ma: float = 50.0, voltage_v: float = 3.3) -> List[dict]:
        """
        Estimate energy per operation in µJ.
        E = P × t = (V × I) × t_seconds
        Default: 50 mA active current, 3.3 V supply (Cortex-M4 typical).
        """
        power_mw = current_ma * voltage_v  # mW
        out = []
        for r in self.results:
            r2 = dict(r)
            t_ms = r.get("total_ms", 0)
            r2["energy_uj"] = round(power_mw * t_ms / 1000, 4)  # mW × ms / 1000 = µJ
            out.append(r2)
        return out

    def bandwidth_overhead_pct(self, baseline_variant: str = "RSA-2048") -> List[dict]:
        """Compare ciphertext + public key bytes vs baseline."""
        baseline = next((r for r in self.results if r.get("variant") == baseline_variant), None)
        if not baseline:
            return self.results
        base_wire = (baseline.get("pk_bytes", 0) or 0) + (baseline.get("ct_bytes", 0) or 0)
        out = []
        for r in self.results:
            wire = (r.get("pk_bytes", 0) or 0) + (r.get("ct_bytes", 0) or 0)
            r2 = dict(r)
            r2["wire_bytes"] = wire
            r2["wire_overhead_pct"] = round((wire / max(base_wire, 1) - 1) * 100, 1)
            out.append(r2)
        return out

    def summary_table(self) -> str:
        """Render ASCII table for terminal / paper appendix."""
        cols = ["variant", "keygen_ms", "encaps_ms", "decaps_ms", "pk_bytes", "ct_bytes"]
        header = "  ".join(f"{c:>14}" for c in cols)
        rows   = [header, "-" * len(header)]
        for r in self.results:
            row = "  ".join(f"{str(r.get(c,'N/A')):>14}" for c in cols)
            rows.append(row)
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Timing Context Manager
# ---------------------------------------------------------------------------

class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter_ns() - self._t0) / 1e6

    def __repr__(self):
        return f"Timer({self.label!r}: {self.elapsed_ms:.3f} ms)"


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(data: Any, path: str, overwrite: bool = True):
    """Save experiment results to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use overwrite=True.")
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_results(path: str) -> Any:
    """Load experiment results from JSON."""
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import numpy as np
    # Demo
    np.random.seed(42)
    y_true = np.array([0,0,0,1,1,1,2,2,0,1])
    y_pred = np.array([0,0,1,1,1,0,2,2,0,1])
    y_prob = np.random.rand(10)
    cm = ClassificationMetrics(y_true, y_pred, y_prob)
    print("Classification Report:")
    print(json.dumps(cm.report(), indent=2))

    print("\nPrivacy-Utility Gap:")
    print(PrivacyMetrics.dp_utility_gap(auc_no_dp=0.913, auc_with_dp=0.878))

    with Timer("example") as t:
        _ = sum(range(1_000_000))
    print(f"\nTimer: {t}")
