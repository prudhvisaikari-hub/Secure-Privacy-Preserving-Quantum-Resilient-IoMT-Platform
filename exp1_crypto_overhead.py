"""
experiments/exp1_crypto_overhead.py
=====================================
Experiment 1: Cryptographic Overhead Comparison
Kyber512/768/1024 vs RSA-2048/4096 vs ECC-P256/P384

Metrics:
  - Key generation time (ms)
  - Encapsulation / encryption time (ms)
  - Decapsulation / decryption time (ms)
  - Public key size, secret key size, ciphertext size (bytes)
  - Energy per operation (µJ, simulated or INA219)
  - Memory footprint estimate (KB)
  - End-to-end handshake latency (ms) per variant

Outputs:
  benchmarks/results/exp1_crypto_overhead.csv
  benchmarks/results/exp1_crypto_overhead.json
  benchmarks/results/exp1_energy.csv

Usage:
    python experiments/exp1_crypto_overhead.py
    python experiments/exp1_crypto_overhead.py --iterations 500 --rsa-iterations 50
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
logger = logging.getLogger("exp1_crypto_overhead")

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run(iterations: int = 200, rsa_iterations: int = 30, energy: bool = True):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Cryptographic Overhead Comparison")
    logger.info("=" * 60)

    # --- Crypto benchmarks ---
    from pqc_layer.comparison import run_comparison
    results = run_comparison(
        iterations_kyber=iterations,
        iterations_rsa=rsa_iterations,
        output=str(RESULTS_DIR / "exp1_crypto_overhead.csv"),
    )

    # --- Energy measurement ---
    if energy:
        logger.info("\nRunning energy benchmark (simulation)...")
        from benchmarks.energy_meter import EnergyMeter
        from pqc_layer.kyber_wrapper import KyberKEM, RSA_KEM, ECC_KEM

        meter = EnergyMeter(backend="simulation")
        ops = [
            ("kyber512_keygen",  lambda: KyberKEM("Kyber512").keygen()),
            ("kyber768_keygen",  lambda: KyberKEM("Kyber768").keygen()),
            ("kyber1024_keygen", lambda: KyberKEM("Kyber1024").keygen()),
            ("rsa2048_keygen",   lambda: RSA_KEM(2048).keygen()),
            ("rsa4096_keygen",   lambda: RSA_KEM(4096).keygen()),
            ("ecc_p256_keygen",  lambda: ECC_KEM().benchmark(1)),
        ]
        energy_results = []
        for label, fn in ops:
            r = meter.benchmark_operation(label, fn, iterations=min(50, iterations))
            energy_results.append(r)
            logger.info(f"  {label:25s}: {r['mean_energy_uj']:8.2f} µJ")

        meter.save_csv(str(RESULTS_DIR / "exp1_energy.csv"))
        meter.save_json(str(RESULTS_DIR / "exp1_energy.json"))

        # Merge energy into results
        energy_map = {r["label"]: r["mean_energy_uj"] for r in energy_results}
        for r in results:
            variant = r.get("variant", "")
            key = variant.lower().replace("-", "").replace("kyber", "kyber") + "_keygen"
            r["energy_uj"] = energy_map.get(key, None)

    # --- Handshake latency ---
    logger.info("\nRunning secure channel handshake latency...")
    from pqc_layer.secure_channel import SecureServer, SecureClient

    hs_results = []
    n_hs = min(50, iterations)
    for variant in ["Kyber512", "Kyber768", "Kyber1024"]:
        times = []
        for _ in range(n_hs):
            server = SecureServer(variant)
            client = SecureClient("exp1_sensor", variant)
            t0 = time.perf_counter_ns()
            h  = client.send_hello()
            sh = server.handle_hello(h)
            ck = client.handle_server_hello(sh)
            server.handle_client_key(client.sensor_id, ck)
            times.append((time.perf_counter_ns() - t0) / 1e6)
        hs_results.append({
            "variant":       variant,
            "mean_hs_ms":    round(sum(times) / len(times), 4),
            "min_hs_ms":     round(min(times), 4),
            "max_hs_ms":     round(max(times), 4),
        })
        logger.info(f"  {variant}: handshake = {hs_results[-1]['mean_hs_ms']:.2f} ms")

    # --- Final output ---
    output = {
        "experiment":       "Exp1: Cryptographic Overhead",
        "iterations_kyber": iterations,
        "iterations_rsa":   rsa_iterations,
        "crypto_results":   results,
        "handshake_results": hs_results,
    }
    out_path = RESULTS_DIR / "exp1_crypto_overhead.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nFull results saved to {out_path}")

    # --- Summary ---
    logger.info("\n" + "="*55)
    logger.info("EXPERIMENT 1 SUMMARY")
    logger.info("="*55)
    kyber_schemes  = [r for r in results if "Kyber" in r.get("variant", "")]
    classic_schemes = [r for r in results if r.get("variant") == "ECC-P256"]
    if kyber_schemes and classic_schemes:
        best_k = min(kyber_schemes, key=lambda r: r.get("total_ms", 99999))
        ecc    = classic_schemes[0]
        logger.info(f"  Fastest Kyber: {best_k['variant']} total={best_k['total_ms']} ms")
        logger.info(f"  ECC-P256:      total={ecc.get('total_ms', 'N/A')} ms")
        logger.info(f"  Overhead ratio Kyber/ECC: "
                    f"{round(best_k.get('total_ms',1)/max(ecc.get('total_ms',1),0.001), 2)}×")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp1: Crypto Overhead")
    parser.add_argument("--iterations",     type=int, default=200)
    parser.add_argument("--rsa-iterations", type=int, default=30)
    parser.add_argument("--no-energy",      action="store_true")
    args = parser.parse_args()
    run(args.iterations, args.rsa_iterations, not args.no_energy)
