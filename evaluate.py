"""
intrusion_detection/evaluate.py
================================
Unified evaluation harness for all IDS models.
Produces publication-ready metrics: ROC curves, PR curves,
confusion matrices, per-class F1, and detection latency.

Saves outputs to benchmarks/results/ids_evaluation/

Usage:
    from intrusion_detection.evaluate import IDSEvaluator
    ev = IDSEvaluator()
    ev.run_full_evaluation()
    ev.save_report("benchmarks/results/ids_full_report.json")
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, roc_curve,
        precision_recall_curve, classification_report, confusion_matrix
    )
    _SK = True
except ImportError:
    _SK = False

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def binary_detection_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None) -> dict:
    """
    Full binary detection metrics for any IDS model.
    Positive class = any attack (label > 0).
    """
    is_atk_true = (y_true > 0).astype(int)
    is_atk_pred = (y_pred > 0).astype(int)

    TP = int(((is_atk_pred==1)&(is_atk_true==1)).sum())
    FP = int(((is_atk_pred==1)&(is_atk_true==0)).sum())
    FN = int(((is_atk_pred==0)&(is_atk_true==1)).sum())
    TN = int(((is_atk_pred==0)&(is_atk_true==0)).sum())

    tpr = TP / max(TP+FN, 1)
    fpr = FP / max(FP+TN, 1)
    tnr = TN / max(TN+FP, 1)
    pre = TP / max(TP+FP, 1)
    f1  = 2*pre*tpr / max(pre+tpr, 1e-8)
    mcc_num = TP*TN - FP*FN
    mcc_den = max(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5, 1e-8)
    mcc = mcc_num / mcc_den

    auc = ap = 0.0
    if _SK and y_prob is not None:
        atk_prob = y_prob if y_prob.ndim == 1 else 1 - y_prob[:, 0]
        try:
            auc = float(roc_auc_score(is_atk_true, atk_prob))
            ap  = float(average_precision_score(is_atk_true, atk_prob))
        except Exception:
            pass

    return {
        "accuracy":          round(float((y_pred==y_true).mean()), 4),
        "tpr_recall":        round(tpr, 4),
        "fpr":               round(fpr, 4),
        "tnr_specificity":   round(tnr, 4),
        "precision":         round(pre, 4),
        "f1_score":          round(f1, 4),
        "mcc":               round(float(mcc), 4),
        "auc_roc":           round(auc, 4),
        "average_precision": round(ap, 4),
        "confusion_matrix":  {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
        "n_samples":         int(len(y_true)),
        "attack_prevalence": round(float(is_atk_true.mean()), 3),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: Optional[Dict[int, str]] = None) -> List[dict]:
    """Per-class precision, recall, F1, support."""
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    results = []
    for c in classes:
        TP = int(((y_pred==c)&(y_true==c)).sum())
        FP = int(((y_pred==c)&(y_true!=c)).sum())
        FN = int(((y_pred!=c)&(y_true==c)).sum())
        pre  = TP / max(TP+FP, 1)
        rec  = TP / max(TP+FN, 1)
        f1   = 2*pre*rec / max(pre+rec, 1e-8)
        supp = int((y_true==c).sum())
        results.append({
            "class_id":   int(c),
            "class_name": class_names.get(c, str(c)) if class_names else str(c),
            "precision":  round(pre, 4),
            "recall":     round(rec, 4),
            "f1_score":   round(f1, 4),
            "support":    supp,
        })
    return results


def detection_latency_benchmark(model,
                                 X_sample: np.ndarray,
                                 n_runs: int = 100,
                                 device: str = "cpu") -> dict:
    """
    Measures per-sample inference latency for real-time detection.
    Reports mean, std, p95, p99 in milliseconds.
    """
    if not _TORCH or model is None:
        return {"error": "PyTorch required"}

    model.eval()
    latencies = []
    x = torch.tensor(X_sample[:1], dtype=torch.float32).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    # Timed runs
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter_ns()
            _ = model(x)
            latencies.append((time.perf_counter_ns() - t0) / 1e6)

    latencies = np.array(latencies)
    return {
        "mean_ms":   round(float(latencies.mean()), 4),
        "std_ms":    round(float(latencies.std()),  4),
        "min_ms":    round(float(latencies.min()),  4),
        "p50_ms":    round(float(np.percentile(latencies, 50)), 4),
        "p95_ms":    round(float(np.percentile(latencies, 95)), 4),
        "p99_ms":    round(float(np.percentile(latencies, 99)), 4),
        "max_ms":    round(float(latencies.max()),  4),
        "n_runs":    n_runs,
        "realtime_feasible": float(np.percentile(latencies, 95)) < 50.0,
    }


def threshold_sweep(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    thresholds: Optional[np.ndarray] = None) -> List[dict]:
    """
    Sweep detection threshold and report TPR/FPR/F1 at each point.
    Useful for choosing operational threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    is_atk = (y_true > 0).astype(int)
    atk_prob = y_prob if y_prob.ndim == 1 else 1 - y_prob[:, 0]
    results = []
    for thr in thresholds:
        pred = (atk_prob >= thr).astype(int)
        TP = int(((pred==1)&(is_atk==1)).sum())
        FP = int(((pred==1)&(is_atk==0)).sum())
        FN = int(((pred==0)&(is_atk==1)).sum())
        TN = int(((pred==0)&(is_atk==0)).sum())
        tpr = TP / max(TP+FN, 1)
        fpr = FP / max(FP+TN, 1)
        pre = TP / max(TP+FP, 1)
        f1  = 2*pre*tpr / max(pre+tpr, 1e-8)
        results.append({"threshold": round(float(thr), 2),
                         "tpr": round(tpr, 4), "fpr": round(fpr, 4),
                         "precision": round(pre, 4), "f1": round(f1, 4)})
    return results


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

class IDSEvaluator:
    """
    Runs full evaluation suite across all IDS models on all dataset types.
    Saves a consolidated JSON report.
    """

    OUT_DIR = Path("benchmarks/results/ids_evaluation")

    def __init__(self, quick: bool = False):
        self.quick = quick
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.report: Dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}

    def _train_and_eval(self, model_name: str, trainer, X: np.ndarray, y: np.ndarray,
                        class_names: Dict) -> dict:
        n = int(len(X) * 0.8)
        X_tr, y_tr = X[:n], y[:n]
        X_te, y_te = X[n:], y[n:]

        logger.info(f"  Training {model_name}...")
        history = trainer.train(X_tr, y_tr)
        metrics = trainer.evaluate(X_te, y_te)

        # Per-class
        if _TORCH and hasattr(trainer, 'model') and trainer.model is not None:
            import torch
            trainer.model.eval()
            X_t = torch.tensor(X_te, dtype=torch.float32)
            with torch.no_grad():
                logits = trainer.model(X_t)
                probs  = torch.softmax(logits, 1).numpy()
            preds = probs.argmax(1)
            per_cls = per_class_metrics(y_te, preds, class_names)
            lat = detection_latency_benchmark(trainer.model, X_te, n_runs=50 if not self.quick else 10)
        else:
            per_cls = []
            lat = {}

        return {**metrics, "per_class": per_cls, "latency": lat, "training_epochs": len(history)}

    def run_full_evaluation(self):
        from intrusion_detection.data_gen import DatasetFactory, NETWORK_CLASSES, POWER_CLASSES
        from intrusion_detection.lstm_ids import IDSTrainer
        from intrusion_detection.transformer_ids import TransformerIDSTrainer
        from intrusion_detection.side_channel import SideChannelDetector

        n_net = 1000 if self.quick else 3000
        n_pwr = 600  if self.quick else 2000
        epochs = 5   if self.quick else 20

        # 1. Network IDS — BiLSTM
        logger.info("[1/4] Network IDS (BiLSTM-Attention)...")
        X_net, y_net = DatasetFactory.network_flows(n_net)
        lstm_trainer = IDSTrainer(model_type="network", epochs=epochs)
        self.report["models"]["network_bilstm"] = self._train_and_eval(
            "BiLSTM-Network", lstm_trainer, X_net, y_net, NETWORK_CLASSES)

        # 2. Network IDS — Transformer
        logger.info("[2/4] Network IDS (Transformer)...")
        tf_trainer = TransformerIDSTrainer(model_type="network", epochs=epochs)
        self.report["models"]["network_transformer"] = self._train_and_eval(
            "Transformer-Network", tf_trainer, X_net, y_net, NETWORK_CLASSES)

        # 3. Side-channel — CNN-LSTM
        logger.info("[3/4] Side-Channel IDS (CNN-LSTM)...")
        X_pwr, y_pwr = DatasetFactory.power_traces(n_pwr, trace_len=256)
        sc = SideChannelDetector(trace_len=256, epochs=epochs)
        sc.train(X_pwr[:int(n_pwr*0.8)], y_pwr[:int(n_pwr*0.8)])
        sc_metrics = sc.evaluate(X_pwr[int(n_pwr*0.8):], y_pwr[int(n_pwr*0.8):])
        tvla = sc.run_tvla(X_pwr, y_pwr)
        self.report["models"]["sidechannel_cnn_lstm"] = {**sc_metrics, "tvla": tvla}

        # 4. Vitals anomaly — BiLSTM
        logger.info("[4/4] Vitals Anomaly IDS (BiLSTM)...")
        from intrusion_detection.data_gen import VITALS_CLASSES
        X_vit, y_vit = DatasetFactory.vitals_anomalies(n_net)
        vit_trainer = IDSTrainer(model_type="network", epochs=epochs)
        # Patch n_features=5 for vitals
        if _TORCH:
            from intrusion_detection.lstm_ids import BiLSTMAttentionIDS
            vit_trainer.model = BiLSTMAttentionIDS(n_features=5, n_classes=4)
        self.report["models"]["vitals_anomaly"] = self._train_and_eval(
            "BiLSTM-Vitals", vit_trainer, X_vit, y_vit, VITALS_CLASSES)

        return self.report

    def save_report(self, path: str = "benchmarks/results/ids_full_report.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        logger.info(f"Full IDS evaluation report saved to {path}")

    def print_summary(self):
        print("\n=== IDS Evaluation Summary ===")
        print(f"{'Model':30s} {'AUC':>8} {'TPR':>8} {'FPR':>8} {'F1':>8} {'P95 lat':>10}")
        print("-" * 70)
        for name, m in self.report.get("models", {}).items():
            auc = m.get("auc_roc", m.get("auc", "N/A"))
            tpr = m.get("tpr_detection_rate", m.get("tpr", "N/A"))
            fpr = m.get("fpr", "N/A")
            f1  = m.get("f1_score", "N/A")
            lat = m.get("latency", {}).get("p95_ms", "N/A")
            print(f"  {name:28s} {str(auc):>8} {str(tpr):>8} {str(fpr):>8} {str(f1):>8} {str(lat):>10}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ev = IDSEvaluator(quick=True)
    ev.run_full_evaluation()
    ev.print_summary()
    ev.save_report()
