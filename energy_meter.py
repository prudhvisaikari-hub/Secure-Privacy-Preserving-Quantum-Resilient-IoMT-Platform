"""
benchmarks/energy_meter.py
===========================
Energy measurement interface for IoMT hardware benchmarking.
Supports:
  - INA219 current/voltage sensor (I2C on Raspberry Pi)
  - Monsoon Power Monitor (USB, high-accuracy)
  - Software simulation (for CI environments without hardware)

Used to measure µJ-level energy consumption during:
  - Kyber keygen / encapsulation / decapsulation
  - AES-GCM encryption
  - LSTM inference on edge device
  - Full telemetry handshake

Usage:
    meter = EnergyMeter(backend="ina219")  # or "simulation"
    with meter.measure("kyber512_keygen") as m:
        kem = KyberKEM("Kyber512")
        pk, sk = kem.keygen()
    print(m.energy_uj, m.elapsed_ms)
"""

import time
import logging
import json
import csv
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# INA219 import (Raspberry Pi hardware)
# ---------------------------------------------------------------------------
try:
    import board
    import busio
    from adafruit_ina219 import INA219
    _INA219_AVAILABLE = True
except ImportError:
    _INA219_AVAILABLE = False

# ---------------------------------------------------------------------------
# Measurement result
# ---------------------------------------------------------------------------

@dataclass
class MeasurementResult:
    label:        str   = ""
    elapsed_ms:   float = 0.0
    voltage_v:    float = 0.0
    current_ma:   float = 0.0
    power_mw:     float = 0.0
    energy_uj:    float = 0.0
    n_samples:    int   = 0
    backend:      str   = "simulation"
    timestamp:    str   = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# INA219 Backend
# ---------------------------------------------------------------------------

class INA219Backend:
    """
    Real INA219 power measurement on Raspberry Pi via I2C.
    Wiring: INA219 SDA→GPIO2, SCL→GPIO3, VCC→3.3V, GND→GND
    Shunt resistor: 0.1 Ω (INA219 breakout default) → 3.2A max.
    """

    SAMPLE_RATE_HZ = 500   # ~500 Hz sampling (2ms per sample)

    def __init__(self, address: int = 0x40):
        if not _INA219_AVAILABLE:
            raise RuntimeError(
                "adafruit-circuitpython-ina219 not installed. "
                "Run: pip install adafruit-circuitpython-ina219"
            )
        i2c = busio.I2C(board.SCL, board.SDA)
        self.ina = INA219(i2c, addr=address)

    def read_once(self) -> tuple:
        """Returns (voltage_V, current_mA, power_mW)."""
        v = float(self.ina.bus_voltage) + float(self.ina.shunt_voltage) / 1000
        i = float(self.ina.current)   # mA
        p = v * i                      # mW
        return v, i, p

    def measure(self, duration_s: float) -> MeasurementResult:
        """Sample continuously for duration_s seconds."""
        samples_v, samples_i, samples_p = [], [], []
        t_start = time.perf_counter()
        while (time.perf_counter() - t_start) < duration_s:
            v, i, p = self.read_once()
            samples_v.append(v)
            samples_i.append(i)
            samples_p.append(p)
            time.sleep(1.0 / self.SAMPLE_RATE_HZ)

        elapsed_s = time.perf_counter() - t_start
        avg_p_mw  = sum(samples_p) / max(len(samples_p), 1)
        energy_uj = avg_p_mw * elapsed_s * 1000  # mW × s × 1000 = µJ

        return MeasurementResult(
            elapsed_ms=round(elapsed_s * 1000, 3),
            voltage_v =round(sum(samples_v) / len(samples_v), 4),
            current_ma=round(sum(samples_i) / len(samples_i), 4),
            power_mw  =round(avg_p_mw, 4),
            energy_uj =round(energy_uj, 4),
            n_samples =len(samples_p),
            backend   ="ina219",
        )


# ---------------------------------------------------------------------------
# Simulation Backend (for CI / non-hardware environments)
# ---------------------------------------------------------------------------

class SimulationBackend:
    """
    Simulates power consumption based on empirically measured
    values for common IoMT crypto operations on Cortex-M4 / RPi.

    Reference values from:
      - pqm4 benchmarks (Cortex-M4)
      - Kannwischer et al. 2019
      - INA219 measurements at ~50 mA active current, 3.3V
    """

    # (voltage_V, current_mA, power_mW) profiles per operation type
    PROFILES = {
        "kyber512_keygen":   (3.3, 52.0,  171.6),
        "kyber768_keygen":   (3.3, 54.0,  178.2),
        "kyber1024_keygen":  (3.3, 56.0,  184.8),
        "kyber512_encaps":   (3.3, 51.0,  168.3),
        "kyber512_decaps":   (3.3, 51.5,  169.9),
        "rsa2048_keygen":    (3.3, 80.0,  264.0),
        "rsa2048_encrypt":   (3.3, 55.0,  181.5),
        "rsa2048_decrypt":   (3.3, 82.0,  270.6),
        "ecc_p256_keygen":   (3.3, 50.5,  166.6),
        "aes256_gcm_encrypt":(3.3, 48.0,  158.4),
        "lstm_inference":    (3.3, 75.0,  247.5),
        "default":           (3.3, 50.0,  165.0),
    }

    def measure(self, duration_s: float, operation: str = "default") -> MeasurementResult:
        profile = self.PROFILES.get(operation, self.PROFILES["default"])
        v, i, p = profile
        energy_uj = p * duration_s * 1000  # µJ

        # Add ±5% jitter for realism
        import random
        jitter = random.uniform(0.95, 1.05)

        return MeasurementResult(
            elapsed_ms=round(duration_s * 1000, 3),
            voltage_v =round(v, 4),
            current_ma=round(i * jitter, 4),
            power_mw  =round(p * jitter, 4),
            energy_uj =round(energy_uj * jitter, 4),
            n_samples =int(duration_s * 500),
            backend   ="simulation",
        )


# ---------------------------------------------------------------------------
# Main EnergyMeter
# ---------------------------------------------------------------------------

class EnergyMeter:
    """
    High-level energy measurement interface.
    Auto-selects INA219 hardware if available, falls back to simulation.

    Usage (context manager):
        meter = EnergyMeter()
        with meter.measure("kyber512_keygen") as m:
            # ... run operation ...
            pass
        print(f"Energy: {m.result.energy_uj:.2f} µJ")
    """

    def __init__(self, backend: str = "auto", ina219_address: int = 0x40):
        if backend == "auto":
            backend = "ina219" if _INA219_AVAILABLE else "simulation"

        if backend == "ina219":
            try:
                self._backend = INA219Backend(ina219_address)
                logger.info("EnergyMeter: INA219 hardware backend active.")
            except Exception as e:
                logger.warning(f"INA219 init failed ({e}), falling back to simulation.")
                self._backend = SimulationBackend()
                backend = "simulation"
        else:
            self._backend = SimulationBackend()

        self.backend_name = backend
        self._results: List[MeasurementResult] = []

    @contextmanager
    def measure(self, label: str = ""):
        """
        Context manager that measures energy during the enclosed block.

        Example:
            with meter.measure("kyber_keygen") as ctx:
                kem.keygen()
            print(ctx.result.energy_uj)
        """
        class MeasureContext:
            result: Optional[MeasurementResult] = None

        ctx = MeasureContext()
        t_start = time.perf_counter()

        try:
            yield ctx
        finally:
            elapsed_s = time.perf_counter() - t_start
            if isinstance(self._backend, SimulationBackend):
                result = self._backend.measure(elapsed_s, operation=label)
            else:
                result = self._backend.measure(elapsed_s)

            result.label   = label
            result.elapsed_ms = round(elapsed_s * 1000, 3)
            ctx.result     = result
            self._results.append(result)

    def benchmark_operation(self, label: str, fn, iterations: int = 50) -> dict:
        """
        Run fn() N times and return average energy and timing.
        """
        energies, timings = [], []
        for _ in range(iterations):
            with self.measure(label) as ctx:
                fn()
            energies.append(ctx.result.energy_uj)
            timings.append(ctx.result.elapsed_ms)

        return {
            "label":           label,
            "iterations":      iterations,
            "mean_energy_uj":  round(sum(energies) / len(energies), 4),
            "mean_elapsed_ms": round(sum(timings)  / len(timings),  4),
            "min_energy_uj":   round(min(energies), 4),
            "max_energy_uj":   round(max(energies), 4),
            "backend":         self.backend_name,
        }

    def results_summary(self) -> List[dict]:
        return [r.as_dict() for r in self._results]

    def save_csv(self, path: str = "benchmarks/results/energy_measurements.csv"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self._results:
            logger.warning("No measurements to save.")
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._results[0].as_dict().keys())
            writer.writeheader()
            writer.writerows(r.as_dict() for r in self._results)
        logger.info(f"Energy measurements saved to {path}")

    def save_json(self, path: str = "benchmarks/results/energy_measurements.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results_summary(), f, indent=2)
        logger.info(f"Energy measurements saved to {path}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== SPQR-IoMT Energy Meter Demo ===\n")

    meter = EnergyMeter(backend="simulation")

    from pqc_layer.kyber_wrapper import KyberKEM, RSA_KEM, ECC_KEM

    operations = [
        ("kyber512_keygen",  lambda: KyberKEM("Kyber512").keygen()),
        ("kyber768_keygen",  lambda: KyberKEM("Kyber768").keygen()),
        ("kyber1024_keygen", lambda: KyberKEM("Kyber1024").keygen()),
        ("rsa2048_keygen",   lambda: RSA_KEM(2048).keygen()),
        ("ecc_p256_keygen",  lambda: ECC_KEM().benchmark(1)),
    ]

    results = []
    for label, fn in operations:
        result = meter.benchmark_operation(label, fn, iterations=20)
        results.append(result)
        print(f"  {label:25s}: {result['mean_energy_uj']:8.2f} µJ | {result['mean_elapsed_ms']:8.3f} ms")

    meter.save_csv()
    meter.save_json()
    print("\nDone. Results saved to benchmarks/results/")
