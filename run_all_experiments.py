"""
run_all_experiments.py
======================
Master runner for all 5 SPQR-IoMT research experiments.
Produces CSV/JSON outputs used in paper figures and tables.

Experiments:
  1. Crypto Overhead      — Kyber vs RSA/ECC timing + key sizes
  2. FL + DP Tradeoff     — Privacy (ε) vs. utility (AUC) curves
  3. HE Inference         — Latency + accuracy of encrypted inference
  4. IDS Detection        — TPR/FPR/AUC for network + side-channel IDS
  5. Quantum Attack Sim   — BB84 Eve detection + QKD vs PQC cost comparison

Usage:
    python run_all_experiments.py                   # run all
    python run_all_experiments.py --exp 1 2 4       # run specific
    python run_all_experiments.py --quick           # fast mode (fewer iterations)
"""

import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("benchmarks/results/experiment_run.log"),
    ]
)
logger = logging.getLogger("SPQR-Experiments")

Path("benchmarks/results").mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Experiment 1: Cryptographic Overhead
# ===========================================================================

def exp1_crypto_overhead(quick: bool = False):
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 1: Cryptographic Overhead Comparison")
    logger.info("="*60)

    from pqc_layer.comparison import run_comparison

    n_iter_kyber = 50 if quick else 200
    n_iter_rsa   = 10 if quick else 50

    results = run_comparison(
        iterations_kyber=n_iter_kyber,
        iterations_rsa=n_iter_rsa,
        output="benchmarks/results/exp1_crypto_overhead.csv"
    )

    with open("benchmarks/results/exp1_crypto_overhead.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary for paper
    logger.info("\n[Exp1 Summary for Paper]")
    kyber_variants = [r for r in results if "Kyber" in r.get("variant", "")]
    classical = [r for r in results if r.get("variant") not in [k["variant"] for k in kyber_variants]]

    if kyber_variants and classical:
        best_kyber = min(kyber_variants, key=lambda r: r.get("total_ms", float("inf")))
        best_classical = min(classical, key=lambda r: r.get("total_ms", float("inf")))
        ratio = best_kyber.get("total_ms", 1) / max(best_classical.get("total_ms", 1), 0.001)
        logger.info(f"  Fastest Kyber variant: {best_kyber.get('variant')} @ {best_kyber.get('total_ms')} ms")
        logger.info(f"  Fastest classical:     {best_classical.get('variant')} @ {best_classical.get('total_ms')} ms")
        logger.info(f"  Overhead ratio (Kyber/Classical): {ratio:.2f}x")

    return results


# ===========================================================================
# Experiment 2: Federated Learning + DP Privacy-Utility Tradeoff
# ===========================================================================

def exp2_fl_dp_tradeoff(quick: bool = False):
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 2: FL + DP Privacy-Utility Tradeoff")
    logger.info("="*60)

    from federated_learning.fl_server import LocalFLSimulator

    epsilon_values = [0.5, 1.0, 3.0, 10.0, None]  # None = no DP
    n_rounds = 10 if quick else 50
    n_clients = 3 if quick else 5

    all_results = []
    for eps in epsilon_values:
        label = f"ε={eps}" if eps else "No DP"
        logger.info(f"  Running FL simulation: {label}, {n_rounds} rounds, {n_clients} clients...")

        sim = LocalFLSimulator(
            n_clients=n_clients,
            n_rounds=n_rounds,
            dp_epsilon=eps,
        )
        history = sim.run()
        final = history[-1]

        result = {
            "epsilon": eps,
            "label": label,
            "final_auc": final["simulated_auc"],
            "final_loss": final["simulated_loss"],
            "n_rounds": n_rounds,
            "n_clients": n_clients,
            "dp_noise_multiplier": sim.dp.noise_multiplier if sim.dp else 0.0,
        }
        all_results.append(result)
        logger.info(f"    → AUC={final['simulated_auc']:.4f}, Loss={final['simulated_loss']:.4f}")

        sim.save_history(f"benchmarks/results/exp2_fl_eps_{str(eps).replace('.', '_')}.json")

    with open("benchmarks/results/exp2_fl_dp_tradeoff.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Privacy-utility curve summary
    logger.info("\n[Exp2 Summary — Privacy-Utility Curve]")
    logger.info(f"  {'Epsilon':>10} | {'AUC':>8} | {'Loss':>8}")
    logger.info(f"  {'-'*32}")
    for r in all_results:
        logger.info(f"  {str(r['epsilon']):>10} | {r['final_auc']:>8.4f} | {r['final_loss']:>8.4f}")

    return all_results


# ===========================================================================
# Experiment 3: Homomorphic Encryption Inference
# ===========================================================================

def exp3_he_inference(quick: bool = False):
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 3: Homomorphic Encryption Inference")
    logger.info("="*60)

    from federated_learning.he_inference import HEInferenceEngine

    configs = [
        {"poly_modulus_degree": 4096,  "label": "CKKS-4096 (fast, less secure)"},
        {"poly_modulus_degree": 8192,  "label": "CKKS-8192 (standard)"},
        {"poly_modulus_degree": 16384, "label": "CKKS-16384 (high precision)"},
    ] if not quick else [
        {"poly_modulus_degree": 4096, "label": "CKKS-4096"}
    ]

    all_results = []
    for cfg in configs:
        logger.info(f"  Testing {cfg['label']}...")
        engine = HEInferenceEngine(n_features=5, poly_modulus_degree=cfg["poly_modulus_degree"])
        engine.train_plaintext_model()

        n_samples = 5 if quick else 20
        bench = engine.benchmark(n_samples=n_samples)
        bench["label"] = cfg["label"]
        bench["poly_modulus_degree"] = cfg["poly_modulus_degree"]

        all_results.append(bench)
        logger.info(
            f"    → Latency: {bench['mean_latency_ms']:.2f}ms ± {bench['std_latency_ms']:.2f}ms | "
            f"Error: {bench['mean_approx_error']:.6f}"
        )

    with open("benchmarks/results/exp3_he_inference.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Real-time feasibility assessment
    logger.info("\n[Exp3 Summary — Real-Time Feasibility]")
    for r in all_results:
        feasible = r["mean_latency_ms"] < 100  # <100ms threshold for real-time
        logger.info(
            f"  {r['label']:40s}: {r['mean_latency_ms']:.1f}ms "
            f"→ {'✓ FEASIBLE' if feasible else '✗ TOO SLOW for real-time'}"
        )

    return all_results


# ===========================================================================
# Experiment 4: IDS Detection Performance
# ===========================================================================

def exp4_ids_detection(quick: bool = False):
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 4: AI-Driven IDS Detection Performance")
    logger.info("="*60)

    from intrusion_detection.lstm_ids import (
        IDSTrainer, NetworkFlowGenerator, PowerTraceGenerator
    )

    results = {}

    # 4A: Network IDS
    logger.info("  [4A] Network IDS (BiLSTM-Attention)...")
    gen = NetworkFlowGenerator(seq_len=20, seed=42)
    n_samples = 500 if quick else 3000
    X, y = gen.generate(n_samples)
    n_train = int(len(X) * 0.8)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]

    trainer_net = IDSTrainer(
        model_type="network",
        epochs=5 if quick else 20,
        hidden_size=64 if quick else 128,
    )
    history_net = trainer_net.train(X_tr, y_tr)
    metrics_net = trainer_net.evaluate(X_te, y_te)
    results["network_ids"] = {**metrics_net, "training_history": history_net}
    logger.info(f"    → AUC={metrics_net.get('auc_roc', 'N/A')} | TPR={metrics_net.get('tpr_detection_rate', 'N/A')}")

    # Save model
    trainer_net.save("intrusion_detection/models/network_ids.pt")

    # 4B: Side-channel IDS
    logger.info("  [4B] Side-Channel IDS (power traces)...")
    sc_gen = PowerTraceGenerator(trace_len=256, seed=42)
    n_sc = 400 if quick else 1500
    X_sc, y_sc = sc_gen.generate(n_sc)
    n_sc_tr = int(len(X_sc) * 0.8)

    trainer_sc = IDSTrainer(
        model_type="sidechannel",
        epochs=5 if quick else 15,
        hidden_size=64 if quick else 128,
    )
    history_sc = trainer_sc.train(X_sc[:n_sc_tr], y_sc[:n_sc_tr])
    metrics_sc = trainer_sc.evaluate(X_sc[n_sc_tr:], y_sc[n_sc_tr:])
    results["sidechannel_ids"] = {**metrics_sc, "training_history": history_sc}
    logger.info(f"    → AUC={metrics_sc.get('auc_roc', 'N/A')} | TPR={metrics_sc.get('tpr_detection_rate', 'N/A')}")

    trainer_sc.save("intrusion_detection/models/sidechannel_ids.pt")

    with open("benchmarks/results/exp4_ids_detection.json", "w") as f:
        # Remove non-serializable history detail
        out = {k: {kk: vv for kk, vv in v.items() if kk != "training_history"} for k, v in results.items()}
        json.dump(out, f, indent=2)

    # Detection thresholds analysis
    logger.info("\n[Exp4 Summary — Detection Performance]")
    for name, metrics in results.items():
        logger.info(
            f"  {name}: AUC={metrics.get('auc_roc', 'N/A')}, "
            f"TPR={metrics.get('tpr_detection_rate', 'N/A')}, "
            f"FPR={metrics.get('fpr', 'N/A')}"
        )

    return results


# ===========================================================================
# Experiment 5: Quantum Attack Simulation + QKD vs PQC
# ===========================================================================

def exp5_quantum_attack_sim(quick: bool = False):
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 5: Quantum Attack Simulation + QKD vs PQC Analysis")
    logger.info("="*60)

    from qkd_comparison.bb84_sim import BB84Simulator, QKDvsPQCAnalysis

    results = {}

    # 5A: BB84 vs Eve
    logger.info("  [5A] BB84 simulation (with and without eavesdropper)...")
    n_qubits = 10_000 if quick else 100_000

    scenarios = [
        {"qber": 0.01, "eve": False, "label": "Low noise, no Eve"},
        {"qber": 0.05, "eve": False, "label": "Medium noise, no Eve"},
        {"qber": 0.02, "eve": True,  "label": "Low noise + Eve (intercept-resend)"},
        {"qber": 0.02, "eve": False, "label": "Low noise, no Eve (baseline)"},
    ]

    bb84_results = []
    for s in scenarios:
        sim = BB84Simulator(
            n_qubits=n_qubits,
            distance_km=10,
            intrinsic_qber=s["qber"],
            eve_present=s["eve"],
        )
        r = sim.run()
        entry = r.summary()
        entry["label"] = s["label"]
        entry["eve_present"] = s["eve"]
        bb84_results.append(entry)
        logger.info(
            f"    {s['label']:45s}: QBER={r.qber_measured:.4f}, "
            f"Key={r.n_final_key_bits} bits, Secure={r.secure}"
        )

    results["bb84"] = bb84_results

    # 5B: Distance sweep
    logger.info("  [5B] BB84 key rate vs. distance...")
    sim_sweep = BB84Simulator(n_qubits=n_qubits, intrinsic_qber=0.02)
    distances = [1, 5, 10, 20, 40, 60, 80]
    distance_results = sim_sweep.sweep_distance(distances)
    results["bb84_distance_sweep"] = distance_results
    for dr in distance_results:
        logger.info(
            f"    d={dr['distance_km']:3d} km: key_rate={dr.get('key_rate_bps', 0):.0f} bps, "
            f"secure={dr.get('secure', False)}"
        )

    # 5C: QKD vs PQC cost analysis
    logger.info("  [5C] QKD vs PQC cost/feasibility analysis...")
    for n_hospitals in [5, 10, 20]:
        analysis = QKDvsPQCAnalysis(n_hospitals=n_hospitals, avg_distance_km=15)
        comp = analysis.comparison_table()
        comp["n_hospitals"] = n_hospitals
        results[f"qkd_vs_pqc_{n_hospitals}_hospitals"] = comp
        logger.info(
            f"    {n_hospitals} hospitals: QKD cost ${comp['qkd']['total_cost_5yr_usd']:,.0f} "
            f"vs PQC ${comp['pqc']['total_cost_5yr_usd']:,.0f} "
            f"(ratio: {comp['cost_ratio_qkd_vs_pqc']:.0f}x)"
        )

    with open("benchmarks/results/exp5_quantum_attack.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ===========================================================================
# Orchestrator
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="SPQR-IoMT Experiment Runner")
    parser.add_argument("--exp", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Experiment numbers to run (1-5)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer iterations for faster runs (CI/testing)")
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    experiments = {
        1: ("Crypto Overhead",        exp1_crypto_overhead),
        2: ("FL + DP Tradeoff",       exp2_fl_dp_tradeoff),
        3: ("HE Inference",           exp3_he_inference),
        4: ("IDS Detection",          exp4_ids_detection),
        5: ("Quantum Attack Sim",     exp5_quantum_attack_sim),
    }

    print("\n" + "█"*60)
    print("  SPQR-IoMT: Experiment Suite")
    print("  Running:", [f"Exp{i}" for i in args.exp])
    print("  Mode:", "Quick" if args.quick else "Full")
    print("█"*60 + "\n")

    t_total = time.perf_counter()
    all_results = {}

    for exp_id in sorted(args.exp):
        if exp_id not in experiments:
            logger.warning(f"Unknown experiment {exp_id}, skipping.")
            continue
        name, fn = experiments[exp_id]
        t0 = time.perf_counter()
        try:
            result = fn(quick=args.quick)
            all_results[f"exp{exp_id}"] = {"status": "success", "result_summary": str(result)[:200]}
        except Exception as e:
            logger.error(f"Experiment {exp_id} ({name}) FAILED: {e}", exc_info=True)
            all_results[f"exp{exp_id}"] = {"status": "failed", "error": str(e)}
        elapsed = time.perf_counter() - t0
        logger.info(f"\n  Exp {exp_id} ({name}) completed in {elapsed:.1f}s\n")

    total_time = time.perf_counter() - t_total
    summary = {
        "experiments_run": args.exp,
        "quick_mode": args.quick,
        "total_time_s": round(total_time, 2),
        "results": all_results,
    }
    with open(f"{args.output_dir}/experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "█"*60)
    print(f"  All experiments complete in {total_time:.1f}s")
    print(f"  Results saved to: {args.output_dir}/")
    print("█"*60 + "\n")
    return summary


if __name__ == "__main__":
    main()
