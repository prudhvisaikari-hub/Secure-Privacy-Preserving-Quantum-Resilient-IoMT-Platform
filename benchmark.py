"""
benchmark.py
============
Standalone benchmarking harness for PQC operations.
Measures CPU cycles, wall-clock time, memory, and (when INA219 available) energy.

Metrics per operation:
  - Mean / median / std / min / max latency (ms)
  - Estimated CPU cycles (from perf_counter_ns × CPU freq)
  - Peak RSS memory delta (bytes)
  - Energy per operation (µJ) — requires INA219 hardware

Usage:
    python -m pqc_layer.benchmark --variant Kyber512 --iterations 500
    python -m pqc_layer.benchmark --all --output benchmarks/results/pqc_bench.csv
"""

import gc
import csv
import json
import time
import logging
import argparse
import resource
import statistics
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from pqc_layer.kyber_wrapper import KyberKEM, KYBER_VARIANTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory measurement helper
# ---------------------------------------------------------------------------

def _rss_kb() -> int:
    """Return current RSS memory in KB (Linux/macOS)."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# INA219 energy measurement (Raspberry Pi hardware)
# ---------------------------------------------------------------------------

class EnergyMeter:
    """
    Wraps INA219 current/voltage sensor for energy measurement.
    Falls back to a no-op stub when hardware is unavailable.

    Wiring: INA219 on I2C bus 1, address 0x40, 1 mΩ shunt resistor.
    """

    def __init__(self):
        self._available = False
        try:
            import board
            import busio
            from adafruit_ina219 import INA219
            i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = INA219(i2c)
            self._available = True
            logger.info("INA219 energy meter connected.")
        except Exception:
            logger.info("INA219 not available — energy measurements will be estimated.")

    def measure_op(self, fn, *args, **kwargs):
        """
        Run fn(*args, **kwargs) while sampling power at ~1 kHz.
        Returns (result, energy_uj).
        """
        if not self._available:
            t0 = time.perf_counter_ns()
            result = fn(*args, **kwargs)
            elapsed_s = (time.perf_counter_ns() - t0) / 1e9
            # Estimate: assume 50 mW active power for ARM Cortex-A72 core
            energy_uj = elapsed_s * 50e-3 * 1e6
            return result, round(energy_uj, 4)

        samples_v, samples_a = [], []
        t0 = time.perf_counter_ns()

        import threading
        stop_flag = threading.Event()

        def _sample():
            while not stop_flag.is_set():
                try:
                    samples_v.append(self._sensor.bus_voltage)
                    samples_a.append(self._sensor.current / 1000.0)  # mA → A
                except Exception:
                    pass
                time.sleep(0.001)  # 1 ms sample interval

        t = threading.Thread(target=_sample, daemon=True)
        t.start()
        result = fn(*args, **kwargs)
        stop_flag.set()
        t.join(timeout=0.1)

        elapsed_s = (time.perf_counter_ns() - t0) / 1e9
        if samples_v and samples_a:
            avg_v = statistics.mean(samples_v)
            avg_a = statistics.mean(samples_a)
            energy_uj = avg_v * avg_a * elapsed_s * 1e6
        else:
            energy_uj = 0.0

        return result, round(energy_uj, 4)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

class PQCBenchmark:
    """
    Runs N iterations of Kyber KEM operations and reports full statistics.
    """

    def __init__(self, variant: str = "Kyber512", iterations: int = 200,
                 warmup: int = 10, measure_energy: bool = False):
        self.variant    = variant
        self.iterations = iterations
        self.warmup     = warmup
        self.kem        = KyberKEM(variant)
        self.meter      = EnergyMeter() if measure_energy else None

    def run(self) -> Dict:
        logger.info(f"Benchmarking {self.variant} ({self.warmup} warmup + {self.iterations} iterations)...")

        # Warmup
        for _ in range(self.warmup):
            pk, sk = self.kem.keygen()
            ct, ss = self.kem.encapsulate(pk)
            self.kem.decapsulate(sk, ct)
        gc.collect()

        kg_times, enc_times, dec_times = [], [], []
        energy_kg, energy_enc, energy_dec = [], [], []
        mem_before = _rss_kb()

        for i in range(self.iterations):
            if self.meter:
                (pk, sk), e_kg   = self.meter.measure_op(self.kem.keygen)
                (ct, ss), e_enc  = self.meter.measure_op(self.kem.encapsulate, pk)
                _,         e_dec = self.meter.measure_op(self.kem.decapsulate, sk, ct)
                energy_kg.append(e_kg)
                energy_enc.append(e_enc)
                energy_dec.append(e_dec)
            else:
                t0 = time.perf_counter_ns()
                pk, sk = self.kem.keygen()
                t1 = time.perf_counter_ns()
                ct, ss = self.kem.encapsulate(pk)
                t2 = time.perf_counter_ns()
                self.kem.decapsulate(sk, ct)
                t3 = time.perf_counter_ns()
                kg_times.append((t1 - t0) / 1e6)
                enc_times.append((t2 - t1) / 1e6)
                dec_times.append((t3 - t2) / 1e6)

        mem_after = _rss_kb()

        def _stats(data: List[float]) -> Dict:
            if not data:
                return {}
            return {
                "mean_ms":   round(statistics.mean(data), 4),
                "median_ms": round(statistics.median(data), 4),
                "std_ms":    round(statistics.stdev(data) if len(data) > 1 else 0, 4),
                "min_ms":    round(min(data), 4),
                "max_ms":    round(max(data), 4),
                "p95_ms":    round(float(np.percentile(data, 95)), 4),
                "p99_ms":    round(float(np.percentile(data, 99)), 4),
            }

        meta = KYBER_VARIANTS[self.variant]
        result = {
            "variant":       self.variant,
            "security_level": meta["security_level"],
            "iterations":    self.iterations,
            "pk_bytes":      meta["pk_bytes"],
            "sk_bytes":      meta["sk_bytes"],
            "ct_bytes":      meta["ct_bytes"],
            "ss_bytes":      meta["ss_bytes"],
            "memory_delta_kb": mem_after - mem_before,
            "keygen":   _stats(kg_times),
            "encaps":   _stats(enc_times),
            "decaps":   _stats(dec_times),
        }

        total_times = [kg_times[i] + enc_times[i] + dec_times[i]
                       for i in range(len(kg_times))]
        result["total"] = _stats(total_times)

        if energy_kg:
            result["energy_uj"] = {
                "keygen_mean":  round(statistics.mean(energy_kg), 4),
                "encaps_mean":  round(statistics.mean(energy_enc), 4),
                "decaps_mean":  round(statistics.mean(energy_dec), 4),
                "total_mean":   round(statistics.mean(
                    [energy_kg[i]+energy_enc[i]+energy_dec[i] for i in range(len(energy_kg))]
                ), 4),
            }

        logger.info(
            f"  Keygen: {result['keygen'].get('mean_ms','?')} ms | "
            f"Encaps: {result['encaps'].get('mean_ms','?')} ms | "
            f"Decaps: {result['decaps'].get('mean_ms','?')} ms"
        )
        return result

    @staticmethod
    def run_all(iterations: int = 200, warmup: int = 10,
                measure_energy: bool = False,
                output: str = "benchmarks/results/pqc_detailed_bench.json") -> List[Dict]:
        results = []
        for variant in KYBER_VARIANTS:
            b = PQCBenchmark(variant, iterations, warmup, measure_energy)
            results.append(b.run())
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved → {output}")
        return results

    @staticmethod
    def print_table(results: List[Dict]):
        cols = ["variant", "keygen_mean", "encaps_mean", "decaps_mean",
                "total_mean", "pk_bytes", "ct_bytes", "memory_delta_kb"]
        header = f"{'Variant':<14}{'Keygen(ms)':>12}{'Encaps(ms)':>12}{'Decaps(ms)':>12}{'Total(ms)':>11}{'PK(B)':>8}{'CT(B)':>8}{'ΔMem(KB)':>10}"
        print("\n" + "="*len(header))
        print(" SPQR-IoMT — PQC Detailed Benchmark")
        print("="*len(header))
        print(header)
        print("-"*len(header))
        for r in results:
            kg  = r.get("keygen",  {}).get("mean_ms", "—")
            enc = r.get("encaps",  {}).get("mean_ms", "—")
            dec = r.get("decaps",  {}).get("mean_ms", "—")
            tot = r.get("total",   {}).get("mean_ms", "—")
            print(f"{r['variant']:<14}{str(kg):>12}{str(enc):>12}{str(dec):>12}"
                  f"{str(tot):>11}{r['pk_bytes']:>8}{r['ct_bytes']:>8}"
                  f"{r.get('memory_delta_kb',0):>10}")
        print("="*len(header))

    @staticmethod
    def save_csv(results: List[Dict], path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in results:
            row = {
                "variant":         r["variant"],
                "security_level":  r["security_level"],
                "pk_bytes":        r["pk_bytes"],
                "ct_bytes":        r["ct_bytes"],
                "sk_bytes":        r["sk_bytes"],
                "keygen_mean_ms":  r.get("keygen", {}).get("mean_ms"),
                "keygen_p95_ms":   r.get("keygen", {}).get("p95_ms"),
                "encaps_mean_ms":  r.get("encaps", {}).get("mean_ms"),
                "decaps_mean_ms":  r.get("decaps", {}).get("mean_ms"),
                "total_mean_ms":   r.get("total",  {}).get("mean_ms"),
                "memory_delta_kb": r.get("memory_delta_kb"),
            }
            if "energy_uj" in r:
                row.update(r["energy_uj"])
            rows.append(row)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CSV saved → {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="SPQR-IoMT PQC Benchmark")
    parser.add_argument("--variant",    default=None, choices=list(KYBER_VARIANTS) + [None])
    parser.add_argument("--all",        action="store_true", help="Benchmark all variants")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup",     type=int, default=10)
    parser.add_argument("--energy",     action="store_true", help="Use INA219 energy meter")
    parser.add_argument("--output",     default="benchmarks/results/pqc_detailed_bench.json")
    parser.add_argument("--csv",        default="benchmarks/results/pqc_bench.csv")
    args = parser.parse_args()

    if args.all or args.variant is None:
        results = PQCBenchmark.run_all(args.iterations, args.warmup, args.energy, args.output)
    else:
        b = PQCBenchmark(args.variant, args.iterations, args.warmup, args.energy)
        results = [b.run()]

    PQCBenchmark.print_table(results)
    PQCBenchmark.save_csv(results, args.csv)


if __name__ == "__main__":
    main()
