"""
benchmarks/run_all.py
=====================
Unified benchmark runner that executes all SPQR-IoMT benchmarks
in sequence and produces a consolidated performance report.

Benchmarks run:
  1. Crypto Overhead   — Kyber vs RSA/ECC timing + energy
  2. Secure Channel    — End-to-end handshake + data encryption latency
  3. FL Simulation     — Communication overhead per FL round
  4. HE Inference      — CKKS latency at various poly modulus sizes
  5. IDS Inference     — Per-sample detection latency (TPR/FPR)
  6. QKD Simulation    — Key rate vs distance

Outputs all results to benchmarks/results/ as CSV + JSON.
Generates a Markdown summary report.

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --quick          # fewer iterations
    python benchmarks/run_all.py --skip crypto    # skip specific benchmark
"""

import sys
import json
import time
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmarks.run_all")

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save_json(data: Any, name: str):
    p = RESULTS_DIR / f"{name}.json"
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved {p}")
    return p


def save_csv(rows: List[dict], name: str):
    if not rows:
        return
    p = RESULTS_DIR / f"{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved {p}")
    return p


# ---------------------------------------------------------------------------
# Benchmark 1 — Crypto Overhead
# ---------------------------------------------------------------------------

def bench_crypto(quick: bool = False) -> dict:
    logger.info("[1/6] Crypto Overhead Benchmark")
    from pqc_layer.kyber_wrapper import KyberKEM, RSA_KEM, ECC_KEM, KYBER_VARIANTS
    from benchmarks.energy_meter import EnergyMeter

    n_kyber = 30 if quick else 200
    n_rsa   = 5  if quick else 30
    meter   = EnergyMeter(backend="simulation")
    results = []

    for variant in KYBER_VARIANTS:
        kem = KyberKEM(variant)
        r   = kem.benchmark(n_kyber).summary()
        energy = meter.benchmark_operation(
            f"{variant.lower()}_keygen", lambda v=variant: KyberKEM(v).keygen(), 10 if quick else 50
        )
        r["energy_uj"] = energy["mean_energy_uj"]
        results.append(r)

    for key_size in [2048, 4096]:
        rsa = RSA_KEM(key_size)
        r   = rsa.benchmark(n_rsa)
        energy = meter.benchmark_operation(
            f"rsa{key_size}_keygen", lambda ks=key_size: RSA_KEM(ks).keygen(), 5 if quick else 20
        )
        r["energy_uj"] = energy["mean_energy_uj"]
        results.append(r)

    ecc_r = ECC_KEM().benchmark(n_kyber)
    ecc_r["energy_uj"] = 0.0
    results.append(ecc_r)

    save_json(results, "bench_crypto_overhead")
    save_csv(results, "bench_crypto_overhead")
    meter.save_csv(str(RESULTS_DIR / "bench_energy.csv"))
    return {"status": "ok", "n_schemes": len(results), "results": results}


# ---------------------------------------------------------------------------
# Benchmark 2 — Secure Channel
# ---------------------------------------------------------------------------

def bench_secure_channel(quick: bool = False) -> dict:
    logger.info("[2/6] Secure Channel Benchmark")
    from pqc_layer.secure_channel import SecureServer, SecureClient

    variants = ["Kyber512", "Kyber768", "Kyber1024"]
    n_iter   = 10 if quick else 100
    results  = []

    for variant in variants:
        hs_times, enc_times, dec_times = [], [], []
        for _ in range(n_iter):
            server = SecureServer(variant)
            client = SecureClient("sensor_bench", variant)

            t0 = time.perf_counter_ns()
            h  = client.send_hello()
            sh = server.handle_hello(h)
            ck = client.handle_server_hello(sh)
            server.handle_client_key(client.sensor_id, ck)
            hs_times.append((time.perf_counter_ns() - t0) / 1e6)

            payload = {"hr": 72, "spo2": 98, "temp": 36.5, "ts": time.time()}
            t1  = time.perf_counter_ns()
            msg = client.send_data(payload)
            enc_times.append((time.perf_counter_ns() - t1) / 1e6)

            t2 = time.perf_counter_ns()
            server.receive_data(client.sensor_id, msg)
            dec_times.append((time.perf_counter_ns() - t2) / 1e6)

        results.append({
            "variant":        variant,
            "handshake_ms":   round(sum(hs_times)  / n_iter, 4),
            "encrypt_ms":     round(sum(enc_times)  / n_iter, 4),
            "decrypt_ms":     round(sum(dec_times)  / n_iter, 4),
            "wire_bytes":     len(msg),
            "iterations":     n_iter,
        })
        logger.info(f"  {variant}: handshake={results[-1]['handshake_ms']:.2f}ms")

    save_json(results, "bench_secure_channel")
    save_csv(results, "bench_secure_channel")
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# Benchmark 3 — FL Communication Overhead
# ---------------------------------------------------------------------------

def bench_fl_overhead(quick: bool = False) -> dict:
    logger.info("[3/6] FL Communication Overhead Benchmark")
    try:
        import torch
        from federated_learning.fl_server import VitalsPredictionModel
        model = VitalsPredictionModel()
        n_params = sum(p.numel() for p in model.parameters())
        weight_bytes_fp32 = n_params * 4
        results = {
            "model":               "BiLSTM-Attention (VitalsPrediction)",
            "n_parameters":        n_params,
            "weight_bytes_fp32":   weight_bytes_fp32,
            "weight_kb":           round(weight_bytes_fp32 / 1024, 2),
            "bytes_per_fl_round":  weight_bytes_fp32 * 2,  # upload + download
            "kb_per_fl_round":     round(weight_bytes_fp32 * 2 / 1024, 2),
            "mb_per_50_rounds":    round(weight_bytes_fp32 * 2 * 50 / 1024 / 1024, 3),
        }
    except ImportError:
        results = {"error": "PyTorch not installed", "model": "BiLSTM-Attention"}

    # FL simulation timing
    from federated_learning.fl_server import LocalFLSimulator
    n_rounds = 5 if quick else 20
    sim = LocalFLSimulator(n_clients=3, n_rounds=n_rounds, dp_epsilon=1.0)
    t0 = time.perf_counter()
    history = sim.run()
    elapsed = time.perf_counter() - t0

    results["fl_simulation"] = {
        "n_rounds":        n_rounds,
        "n_clients":       3,
        "total_time_s":    round(elapsed, 2),
        "time_per_round_s": round(elapsed / n_rounds, 3),
        "final_auc":       history[-1]["simulated_auc"],
    }

    save_json(results, "bench_fl_overhead")
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# Benchmark 4 — HE Inference
# ---------------------------------------------------------------------------

def bench_he_inference(quick: bool = False) -> dict:
    logger.info("[4/6] Homomorphic Encryption Inference Benchmark")
    from federated_learning.he_inference import HEInferenceEngine

    configs = [{"poly": 4096}, {"poly": 8192}] if quick else \
              [{"poly": 4096}, {"poly": 8192}, {"poly": 16384}]
    n_samples = 5 if quick else 20
    results = []

    for cfg in configs:
        engine = HEInferenceEngine(n_features=5, poly_modulus_degree=cfg["poly"])
        engine.train_plaintext_model()
        bench = engine.benchmark(n_samples)
        bench["poly_modulus_degree"] = cfg["poly"]
        results.append(bench)
        logger.info(f"  CKKS-{cfg['poly']}: {bench['mean_latency_ms']:.2f} ms "
                    f"(err={bench['mean_approx_error']:.6f})")

    save_json(results, "bench_he_inference")
    save_csv(results, "bench_he_inference")
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# Benchmark 5 — IDS Inference Latency
# ---------------------------------------------------------------------------

def bench_ids_latency(quick: bool = False) -> dict:
    logger.info("[5/6] IDS Inference Latency Benchmark")
    from intrusion_detection.lstm_ids import IDSTrainer, NetworkFlowGenerator
    from intrusion_detection.evaluate import detection_latency_benchmark

    n_samples = 500 if quick else 2000
    epochs    = 3   if quick else 10

    gen = NetworkFlowGenerator(seq_len=20)
    X, y = gen.generate(n_samples)
    n = int(len(X) * 0.8)
    trainer = IDSTrainer(model_type="network", epochs=epochs)
    trainer.train(X[:n], y[:n])
    metrics = trainer.evaluate(X[n:], y[n:])

    lat = {}
    try:
        import torch
        lat = detection_latency_benchmark(trainer.model, X[n:], n_runs=20 if quick else 100)
    except Exception:
        pass

    results = {**metrics, "latency": lat, "n_train": n, "n_test": len(X)-n}
    save_json(results, "bench_ids_latency")
    logger.info(f"  IDS AUC={metrics.get('auc_roc','N/A')} | P95={lat.get('p95_ms','N/A')} ms")
    return {"status": "ok", "results": results}


# ---------------------------------------------------------------------------
# Benchmark 6 — QKD Key Rate vs Distance
# ---------------------------------------------------------------------------

def bench_qkd(quick: bool = False) -> dict:
    logger.info("[6/6] QKD Key Rate Benchmark")
    from qkd_comparison.channel_noise import distance_performance_table
    from qkd_comparison.cost_analysis import CostAnalyzer

    distances = [1, 5, 10, 20, 40, 60, 80] if quick else \
                [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100]

    table = distance_performance_table(distances)
    save_json(table, "bench_qkd_distance")
    save_csv(table,  "bench_qkd_distance")

    ca = CostAnalyzer(n_nodes=10, avg_distance_km=15, years=5)
    cost_report = ca.full_report()
    save_json(cost_report, "bench_qkd_vs_pqc_cost")

    return {"status": "ok", "distance_table": table,
            "cost_ratio": cost_report["comparison"]["cost_ratio_qkd_vs_pqc"]}


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------

def generate_markdown_report(all_results: dict, elapsed_total: float) -> str:
    lines = [
        "# SPQR-IoMT Benchmark Report",
        f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total runtime:** {elapsed_total:.1f}s\n",
        "---\n",
        "## 1. Cryptographic Overhead\n",
    ]

    crypto = all_results.get("crypto", {}).get("results", [])
    if crypto:
        lines += ["| Scheme | Keygen (ms) | Encaps (ms) | Decaps (ms) | PK bytes | CT bytes | Energy (µJ) |",
                  "|--------|------------|------------|------------|----------|----------|-------------|"]
        for r in crypto:
            lines.append(
                f"| {r.get('variant','N/A')} | {r.get('keygen_ms','N/A')} | "
                f"{r.get('encaps_ms','N/A')} | {r.get('decaps_ms','N/A')} | "
                f"{r.get('pk_bytes','N/A')} | {r.get('ct_bytes','N/A')} | "
                f"{r.get('energy_uj','N/A')} |"
            )

    lines += ["\n## 2. Secure Channel\n"]
    sc = all_results.get("channel", {}).get("results", [])
    if sc:
        lines += ["| Variant | Handshake (ms) | Encrypt (ms) | Decrypt (ms) | Wire bytes |",
                  "|---------|---------------|-------------|-------------|------------|"]
        for r in sc:
            lines.append(f"| {r['variant']} | {r['handshake_ms']} | {r['encrypt_ms']} | {r['decrypt_ms']} | {r['wire_bytes']} |")

    lines += ["\n## 3. HE Inference\n"]
    he = all_results.get("he", {}).get("results", [])
    if he:
        lines += ["| CKKS Config | Mean latency (ms) | Max latency (ms) | Approx error |",
                  "|-------------|------------------|-----------------|--------------|"]
        for r in he:
            lines.append(f"| CKKS-{r.get('poly_modulus_degree','?')} | {r.get('mean_latency_ms','N/A')} | {r.get('max_latency_ms','N/A')} | {r.get('mean_approx_error','N/A')} |")

    lines += ["\n## 4. IDS Detection\n"]
    ids = all_results.get("ids", {}).get("results", {})
    if ids:
        lines += [f"- **AUC-ROC:** {ids.get('auc_roc','N/A')}",
                  f"- **TPR:** {ids.get('tpr_detection_rate','N/A')}",
                  f"- **FPR:** {ids.get('fpr','N/A')}",
                  f"- **P95 inference latency:** {ids.get('latency',{}).get('p95_ms','N/A')} ms"]

    lines += ["\n## 5. QKD vs PQC Cost\n"]
    qkd = all_results.get("qkd", {})
    if "cost_ratio" in qkd:
        lines.append(f"- **Cost ratio (QKD/PQC) for 10 nodes over 5 years:** {qkd['cost_ratio']}×")

    lines += ["\n---\n*All results saved to `benchmarks/results/`*"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SPQR-IoMT Unified Benchmark Runner")
    parser.add_argument("--quick", action="store_true", help="Fast mode (fewer iterations)")
    parser.add_argument("--skip", nargs="*", default=[], help="Benchmarks to skip",
                        choices=["crypto","channel","fl","he","ids","qkd"])
    args = parser.parse_args()

    print("\n" + "▓"*60)
    print("  SPQR-IoMT Unified Benchmark Suite")
    print(f"  Mode: {'Quick' if args.quick else 'Full'}")
    if args.skip:
        print(f"  Skipping: {args.skip}")
    print("▓"*60 + "\n")

    t_total = time.perf_counter()
    all_results = {}

    benchmarks = [
        ("crypto",  "Crypto Overhead",        bench_crypto),
        ("channel", "Secure Channel",         bench_secure_channel),
        ("fl",      "FL Communication",       bench_fl_overhead),
        ("he",      "HE Inference",           bench_he_inference),
        ("ids",     "IDS Latency",            bench_ids_latency),
        ("qkd",     "QKD Distance/Cost",      bench_qkd),
    ]

    for key, name, fn in benchmarks:
        if key in args.skip:
            logger.info(f"Skipping: {name}")
            continue
        t0 = time.perf_counter()
        try:
            result = fn(quick=args.quick)
            all_results[key] = result
        except Exception as e:
            logger.error(f"{name} FAILED: {e}", exc_info=True)
            all_results[key] = {"status": "failed", "error": str(e)}
        logger.info(f"  → {name} done in {time.perf_counter()-t0:.1f}s\n")

    elapsed = time.perf_counter() - t_total

    # Save consolidated results
    save_json(all_results, "benchmark_summary")

    # Generate Markdown report
    md = generate_markdown_report(all_results, elapsed)
    md_path = RESULTS_DIR / "benchmark_report.md"
    md_path.write_text(md)
    logger.info(f"Markdown report: {md_path}")

    print("\n" + "▓"*60)
    print(f"  All benchmarks complete in {elapsed:.1f}s")
    print(f"  Results: benchmarks/results/")
    print("▓"*60 + "\n")
    return all_results


if __name__ == "__main__":
    main()
