"""
experiments/exp4_ids_detection.py
===================================
Experiment 4: AI-Driven IDS Detection Performance

Trains and evaluates:
  - BiLSTM-Attention IDS on network flows (UNSW-NB15-style)
  - Transformer IDS on network flows (comparison)
  - CNN-LSTM side-channel classifier on power traces
  - Vitals anomaly detector

Reports: TPR, FPR, AUC-ROC, F1, per-class metrics, inference latency.

Outputs:
  benchmarks/results/exp4_ids_detection.json
  benchmarks/results/exp4_ids_per_class.csv
  intrusion_detection/models/  (saved model weights)

Usage:
    python experiments/exp4_ids_detection.py
    python experiments/exp4_ids_detection.py --quick
"""

import sys
import json
import csv
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
logger = logging.getLogger("exp4_ids_detection")

RESULTS_DIR  = Path("benchmarks/results")
MODELS_DIR   = Path("intrusion_detection/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run(quick: bool = False):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: AI-Driven IDS Detection Performance")
    logger.info("=" * 60)

    from intrusion_detection.data_gen import DatasetFactory, NETWORK_CLASSES, POWER_CLASSES, VITALS_CLASSES
    from intrusion_detection.lstm_ids import IDSTrainer
    from intrusion_detection.transformer_ids import TransformerIDSTrainer
    from intrusion_detection.side_channel import SideChannelDetector
    from intrusion_detection.evaluate import (
        binary_detection_metrics, per_class_metrics, detection_latency_benchmark
    )

    n_net  = 800  if quick else 3000
    n_pwr  = 500  if quick else 2000
    epochs = 5    if quick else 20

    all_results = {}

    # ---- 4A: Network IDS — BiLSTM ----
    logger.info("\n[4A] Network IDS — BiLSTM-Attention")
    X_net, y_net = DatasetFactory.network_flows(n_net, seq_len=20)
    n = int(len(X_net) * 0.8)
    X_tr, y_tr = X_net[:n], y_net[:n]
    X_te, y_te = X_net[n:], y_net[n:]

    lstm_trainer = IDSTrainer(model_type="network", epochs=epochs, hidden_size=128)
    lstm_trainer.train(X_tr, y_tr)
    lstm_metrics = lstm_trainer.evaluate(X_te, y_te)
    lstm_trainer.save(str(MODELS_DIR / "network_bilstm.pt"))

    # Per-class
    try:
        import torch
        lstm_trainer.model.eval()
        X_t = torch.tensor(X_te, dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(lstm_trainer.model(X_t), 1).numpy()
        preds = probs.argmax(1)
        lstm_per_class = per_class_metrics(y_te, preds, NETWORK_CLASSES)
        lstm_latency   = detection_latency_benchmark(lstm_trainer.model, X_te,
                                                     n_runs=20 if quick else 100)
    except Exception:
        lstm_per_class = []
        lstm_latency   = {}

    all_results["network_bilstm"] = {
        **lstm_metrics,
        "per_class": lstm_per_class,
        "latency":   lstm_latency,
    }
    logger.info(f"  AUC={lstm_metrics.get('auc_roc','N/A')} | "
                f"TPR={lstm_metrics.get('tpr_detection_rate','N/A')} | "
                f"FPR={lstm_metrics.get('fpr','N/A')}")

    # ---- 4B: Network IDS — Transformer ----
    logger.info("\n[4B] Network IDS — Transformer")
    tf_trainer = TransformerIDSTrainer(model_type="network", epochs=epochs,
                                        d_model=64, n_heads=4, n_layers=3)
    tf_trainer.train(X_tr, y_tr)
    tf_metrics = tf_trainer.evaluate(X_te, y_te)
    tf_trainer.save(str(MODELS_DIR / "network_transformer.pt"))
    all_results["network_transformer"] = tf_metrics
    logger.info(f"  AUC={tf_metrics.get('auc_roc','N/A')} | "
                f"Params={tf_metrics.get('n_parameters','N/A')}")

    # ---- 4C: Side-channel IDS ----
    logger.info("\n[4C] Side-Channel IDS — CNN-LSTM")
    X_pwr, y_pwr = DatasetFactory.power_traces(n_pwr, trace_len=256)
    n_pwr_tr = int(len(X_pwr) * 0.8)
    sc = SideChannelDetector(trace_len=256, epochs=epochs)
    sc.train(X_pwr[:n_pwr_tr], y_pwr[:n_pwr_tr])
    sc_metrics = sc.evaluate(X_pwr[n_pwr_tr:], y_pwr[n_pwr_tr:])
    tvla       = sc.run_tvla(X_pwr, y_pwr)
    sc.save(str(MODELS_DIR / "sidechannel_cnn_lstm.pt"))
    all_results["sidechannel_cnn_lstm"] = {**sc_metrics, "tvla": tvla}
    logger.info(f"  AUC={sc_metrics.get('auc_roc','N/A')} | TVLA: {tvla}")

    # ---- 4D: Vitals Anomaly IDS ----
    logger.info("\n[4D] Vitals Anomaly Detection")
    X_vit, y_vit = DatasetFactory.vitals_anomalies(n_net, seq_len=24)
    n_vit = int(len(X_vit) * 0.8)
    try:
        import torch
        from intrusion_detection.lstm_ids import BiLSTMAttentionIDS
        vit_trainer = IDSTrainer(model_type="network", epochs=epochs)
        vit_trainer.model = BiLSTMAttentionIDS(n_features=5, n_classes=4)
        vit_trainer.train(X_vit[:n_vit], y_vit[:n_vit])
        vit_metrics = vit_trainer.evaluate(X_vit[n_vit:], y_vit[n_vit:])
        vit_trainer.save(str(MODELS_DIR / "vitals_anomaly.pt"))
        all_results["vitals_anomaly"] = vit_metrics
        logger.info(f"  AUC={vit_metrics.get('auc_roc','N/A')}")
    except Exception as e:
        logger.warning(f"  Vitals IDS skipped: {e}")
        all_results["vitals_anomaly"] = {"error": str(e)}

    # ---- Model comparison ----
    logger.info("\n[4E] BiLSTM vs Transformer comparison")
    comparison = {
        "BiLSTM-Attention": {
            "auc": lstm_metrics.get("auc_roc"),
            "f1":  lstm_metrics.get("f1_score"),
            "p95_ms": lstm_latency.get("p95_ms"),
        },
        "Transformer": {
            "auc": tf_metrics.get("auc_roc"),
            "f1":  tf_metrics.get("f1_score"),
            "n_params": tf_metrics.get("n_parameters"),
        },
    }
    winner = max(comparison, key=lambda m: comparison[m].get("auc") or 0)
    comparison["winner_by_auc"] = winner
    all_results["model_comparison"] = comparison
    logger.info(f"  Winner by AUC: {winner}")

    # ---- Save ----
    output = {
        "experiment": "Exp4: AI-Driven IDS Detection",
        "results":    all_results,
    }
    out_path = RESULTS_DIR / "exp4_ids_detection.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Per-class CSV
    if lstm_per_class:
        csv_path = RESULTS_DIR / "exp4_ids_per_class.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=lstm_per_class[0].keys())
            writer.writeheader()
            writer.writerows(lstm_per_class)
        logger.info(f"Per-class results saved to {csv_path}")

    logger.info(f"\nAll results saved to {out_path}")

    # Summary
    logger.info("\n" + "="*65)
    logger.info("EXPERIMENT 4 SUMMARY — IDS Detection Performance")
    logger.info("="*65)
    logger.info(f"  {'Model':30s} {'AUC':>8} {'TPR':>8} {'FPR':>8} {'F1':>8}")
    logger.info(f"  {'-'*60}")
    for name, m in all_results.items():
        if name == "model_comparison":
            continue
        auc = m.get("auc_roc", m.get("auc", "N/A"))
        tpr = m.get("tpr_detection_rate", m.get("tpr", "N/A"))
        fpr = m.get("fpr", "N/A")
        f1  = m.get("f1_score", "N/A")
        logger.info(f"  {name:30s} {str(auc):>8} {str(tpr):>8} {str(fpr):>8} {str(f1):>8}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp4: IDS Detection")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.quick)
