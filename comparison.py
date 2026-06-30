"""
comparison.py
=============
Side-by-side benchmarking of Kyber512/768/1024 vs RSA-2048/4096 vs ECC-P256/P384.

Metrics collected:
  - Key generation time (ms)
  - Encapsulation / encryption time (ms)
  - Decapsulation / decryption time (ms)
  - Public key size (bytes)
  - Secret key / private key size (bytes)
  - Ciphertext / encrypted size (bytes)

Output: JSON + pretty-printed table to stdout, CSV to benchmarks/results/

Usage:
    python -m pqc_layer.comparison --iterations 100 --output benchmarks/results/crypto_comparison.csv
"""

import csv
import json
import time
import argparse
import logging
import secrets
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import benchmarking subjects
# ---------------------------------------------------------------------------
from pqc_layer.kyber_wrapper import KyberKEM, RSA_KEM, ECC_KEM, KYBER_VARIANTS

try:
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.backends import default_backend
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not available — RSA/ECC results will use stub timings.")


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_kyber(variant: str, iterations: int = 100) -> Dict[str, Any]:
    kem = KyberKEM(variant)
    result = kem.benchmark(iterations)
    return result.summary()


def benchmark_rsa(key_size: int, iterations: int = 50) -> Dict[str, Any]:
    """RSA keygen + OAEP encrypt/decrypt cycle."""
    if not _CRYPTO_AVAILABLE:
        return {"variant": f"RSA-{key_size}", "error": "cryptography not installed"}

    kg_total = enc_total = dec_total = 0.0
    pk_size = sk_size = ct_size = 0

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=key_size, backend=default_backend()
        )
        t1 = time.perf_counter_ns()

        public_key = private_key.public_key()
        message = secrets.token_bytes(32)  # 256-bit symmetric key to wrap
        ciphertext = public_key.encrypt(
            message,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        t2 = time.perf_counter_ns()

        private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        t3 = time.perf_counter_ns()

        kg_total  += t1 - t0
        enc_total += t2 - t1
        dec_total += t3 - t2

        pk_size = len(public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
        sk_size = len(private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
        ct_size = len(ciphertext)

    return {
        "variant": f"RSA-{key_size}",
        "keygen_ms": round(kg_total  / iterations / 1e6, 4),
        "encaps_ms": round(enc_total / iterations / 1e6, 4),
        "decaps_ms": round(dec_total / iterations / 1e6, 4),
        "total_ms":  round((kg_total + enc_total + dec_total) / iterations / 1e6, 4),
        "pk_bytes":  pk_size,
        "sk_bytes":  sk_size,
        "ct_bytes":  ct_size,
        "ss_bytes":  32,
        "iterations": iterations,
    }


def benchmark_ecdh(curve_name: str, iterations: int = 100) -> Dict[str, Any]:
    """ECDH ephemeral key exchange timing."""
    if not _CRYPTO_AVAILABLE:
        return {"variant": f"ECC-{curve_name}", "error": "cryptography not installed"}

    curve_map = {
        "P256": ec.SECP256R1(),
        "P384": ec.SECP384R1(),
        "P521": ec.SECP521R1(),
    }
    curve = curve_map.get(curve_name, ec.SECP256R1())

    kg_total = exc_total = 0.0
    pk_size = sk_size = 0

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        alice_sk = ec.generate_private_key(curve, default_backend())
        t1 = time.perf_counter_ns()
        bob_sk   = ec.generate_private_key(curve, default_backend())

        alice_ss = alice_sk.exchange(ec.ECDH(), bob_sk.public_key())
        t2 = time.perf_counter_ns()

        kg_total  += t1 - t0
        exc_total += t2 - t1

        pk_size = len(alice_sk.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ))
        sk_size = len(alice_sk.private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()
        ))

    return {
        "variant": f"ECC-{curve_name}",
        "keygen_ms": round(kg_total  / iterations / 1e6, 4),
        "encaps_ms": round(exc_total / iterations / 1e6, 4),
        "decaps_ms": 0.0,  # symmetric with ECDH — already measured in encaps
        "total_ms":  round((kg_total + exc_total) / iterations / 1e6, 4),
        "pk_bytes":  pk_size,
        "sk_bytes":  sk_size,
        "ct_bytes":  pk_size,  # ephemeral public key is the "ciphertext"
        "ss_bytes":  32,
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Compute overhead ratios relative to ECC-P256
# ---------------------------------------------------------------------------

def compute_ratios(results: List[Dict], baseline_variant: str = "ECC-P256") -> List[Dict]:
    baseline = next((r for r in results if r["variant"] == baseline_variant), None)
    if baseline is None:
        return results
    enriched = []
    for r in results:
        r2 = dict(r)
        for field in ["keygen_ms", "encaps_ms", "total_ms", "pk_bytes", "ct_bytes"]:
            base_val = baseline.get(field, 1) or 1
            r2[f"{field}_ratio_vs_{baseline_variant}"] = round(r.get(field, 0) / base_val, 3)
        enriched.append(r2)
    return enriched


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_table(results: List[Dict]):
    cols = ["variant", "keygen_ms", "encaps_ms", "decaps_ms", "total_ms",
            "pk_bytes", "sk_bytes", "ct_bytes"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in results)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("  SPQR-IoMT Cryptographic Overhead Comparison")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for r in results:
        row = "  ".join(str(r.get(c, "N/A")).ljust(widths[c]) for c in cols)
        print(row)
    print(sep)


def save_csv(results: List[Dict], path: str):
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(iterations_kyber: int = 100, iterations_rsa: int = 30,
                   output: str = "benchmarks/results/crypto_comparison.csv") -> List[Dict]:
    logging.basicConfig(level=logging.INFO)
    results = []

    print("\n[1/3] Benchmarking Kyber variants...")
    for variant in KYBER_VARIANTS:
        print(f"  → {variant} ({iterations_kyber} iterations)...")
        r = benchmark_kyber(variant, iterations_kyber)
        results.append(r)

    print("\n[2/3] Benchmarking RSA variants...")
    for key_size in [2048, 4096]:
        print(f"  → RSA-{key_size} ({iterations_rsa} iterations)...")
        r = benchmark_rsa(key_size, iterations_rsa)
        results.append(r)

    print("\n[3/3] Benchmarking ECC variants...")
    for curve in ["P256", "P384"]:
        print(f"  → ECC-{curve} ({iterations_kyber} iterations)...")
        r = benchmark_ecdh(curve, iterations_kyber)
        results.append(r)

    results = compute_ratios(results)
    print_table(results)
    save_csv(results, output)
    print("\nJSON summary:")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PQC vs Classical Crypto Benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations for Kyber/ECC")
    parser.add_argument("--rsa-iterations", type=int, default=30, help="Iterations for RSA (slow)")
    parser.add_argument("--output", type=str, default="benchmarks/results/crypto_comparison.csv")
    args = parser.parse_args()
    run_comparison(args.iterations, args.rsa_iterations, args.output)
