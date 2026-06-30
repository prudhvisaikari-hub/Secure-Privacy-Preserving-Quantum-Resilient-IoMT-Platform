"""
bb84_sim.py
===========
Simulation of the BB84 Quantum Key Distribution protocol for hospital fiber links.
Used to compare QKD vs PQC (Kyber) feasibility for IoMT deployments.

Simulates:
  - Alice prepares qubits in random bases (rectilinear/diagonal)
  - Quantum channel with configurable noise (QBER)
  - Bob measures in random bases
  - Sifting, error estimation, and key reconciliation
  - Privacy amplification (hash compression)
  - Final secret key rate calculation

Also provides cost_analysis.py-style comparison of QKD vs PQC deployment.

Usage:
    sim = BB84Simulator(n_qubits=10000, qber=0.02, distance_km=10)
    result = sim.run()
    print(result.summary())
"""

import json
import math
import time
import logging
import secrets
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BB84 Protocol Simulator
# ---------------------------------------------------------------------------

@dataclass
class BB84Result:
    """Result of a BB84 simulation run."""
    n_qubits_sent: int
    n_sifted: int
    n_errors_detected: int
    qber_measured: float
    n_final_key_bits: int
    key_generation_rate_bps: float
    alice_key: bytes
    bob_key: bytes
    keys_match: bool
    secure: bool          # QBER below threshold
    simulation_ms: float

    def summary(self) -> dict:
        return {
            "qubits_sent": self.n_qubits_sent,
            "sifted_bits": self.n_sifted,
            "qber_measured": round(self.qber_measured, 4),
            "final_key_bits": self.n_final_key_bits,
            "key_rate_bps": round(self.key_generation_rate_bps, 2),
            "keys_match": self.keys_match,
            "secure": self.secure,
            "simulation_ms": round(self.simulation_ms, 2),
        }


class QuantumChannel:
    """
    Simulates optical fiber quantum channel.
    Models:
      - Photon loss (attenuation)
      - Depolarizing noise (from fiber birefringence, misalignment)
      - Eve's intercept-and-resend attack (optional)
    """
    FIBER_ATTENUATION_DB_PER_KM = 0.2   # Standard SMF-28 fiber

    def __init__(self, distance_km: float = 10.0,
                 intrinsic_qber: float = 0.01,
                 eve_present: bool = False):
        self.distance_km = distance_km
        self.intrinsic_qber = intrinsic_qber
        self.eve_present = eve_present

        # Transmission probability (photon survival)
        loss_db = self.FIBER_ATTENUATION_DB_PER_KM * distance_km
        self.transmittance = 10 ** (-loss_db / 10)

        # Eve adds ~25% QBER if she intercepts all qubits
        self.total_qber = intrinsic_qber + (0.25 if eve_present else 0.0)

    def transmit(self, bits: np.ndarray, bases: np.ndarray,
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate qubit transmission through channel.
        Returns (received_bits, received_mask) where mask=1 means photon arrived.
        """
        n = len(bits)
        # Photon loss
        arrived = rng.random(n) < self.transmittance
        # Channel errors (QBER)
        flip = rng.random(n) < self.total_qber
        received_bits = (bits ^ flip.astype(int)) * arrived
        return received_bits, arrived

    @property
    def info(self) -> dict:
        return {
            "distance_km": self.distance_km,
            "transmittance": round(self.transmittance, 4),
            "intrinsic_qber": self.intrinsic_qber,
            "eve_present": self.eve_present,
            "effective_qber": round(self.total_qber, 4),
        }


class BB84Simulator:
    """
    Full BB84 QKD protocol simulation.

    Steps:
      1. Alice generates random bits + random bases
      2. Alice encodes qubits (rectilinear: 0→|0⟩, 1→|1⟩; diagonal: 0→|+⟩, 1→|−⟩)
      3. Quantum channel transmission (loss + noise + optional Eve)
      4. Bob measures in random bases
      5. Classical sifting (keep only matching-basis bits)
      6. Error estimation on sample
      7. Information reconciliation (cascade protocol — simplified here)
      8. Privacy amplification (SHA3 hash compression)
      9. Final secret key
    """

    QBER_SECURITY_THRESHOLD = 0.11  # ~11% QBER → insecure (above Shannon limit)

    def __init__(self, n_qubits: int = 10000,
                 distance_km: float = 10.0,
                 intrinsic_qber: float = 0.02,
                 eve_present: bool = False,
                 seed: Optional[int] = None):
        self.n_qubits = n_qubits
        self.channel = QuantumChannel(distance_km, intrinsic_qber, eve_present)
        self.rng = np.random.default_rng(seed or 42)

    def run(self) -> BB84Result:
        t0 = time.perf_counter_ns()

        # --- Step 1: Alice prepares ---
        alice_bits  = self.rng.integers(0, 2, self.n_qubits)  # random bits
        alice_bases = self.rng.integers(0, 2, self.n_qubits)  # 0=rectilinear, 1=diagonal

        # --- Step 2: Channel transmission ---
        received_bits, arrived_mask = self.channel.transmit(alice_bits, alice_bases, self.rng)

        # --- Step 3: Bob measures ---
        bob_bases = self.rng.integers(0, 2, self.n_qubits)
        # Bob gets correct bit only if bases match AND photon arrived
        bob_bits = np.where(
            arrived_mask & (bob_bases == alice_bases),
            received_bits,
            self.rng.integers(0, 2, self.n_qubits)  # random bit if base mismatch
        )

        # --- Step 4: Sifting (keep matching bases & arrived photons) ---
        sift_mask = arrived_mask & (alice_bases == bob_bases)
        alice_sifted = alice_bits[sift_mask]
        bob_sifted   = bob_bits[sift_mask]
        n_sifted = len(alice_sifted)

        # --- Step 5: Error estimation (sacrifice 10% of sifted bits) ---
        n_sample = max(1, n_sifted // 10)
        sample_idx = self.rng.choice(n_sifted, n_sample, replace=False)
        errors = (alice_sifted[sample_idx] != bob_sifted[sample_idx]).sum()
        qber = float(errors) / n_sample

        # Remove sample bits from key material
        keep_mask = np.ones(n_sifted, dtype=bool)
        keep_mask[sample_idx] = False
        alice_raw = alice_sifted[keep_mask]
        bob_raw   = bob_sifted[keep_mask]
        n_raw = len(alice_raw)

        # --- Step 6: Information reconciliation (simplified: discard error bits) ---
        # Real cascade/LDPC reconciliation would fix errors using public communication.
        # Here we use oracle reconciliation for simulation purity.
        reconciled = alice_raw == bob_raw  # oracle: know which bits are errors
        alice_reconciled = alice_raw[reconciled]
        bob_reconciled   = bob_raw[reconciled]
        n_reconciled = len(alice_reconciled)

        # --- Step 7: Privacy amplification ---
        # Secret key length after PA = n_reconciled * (1 - h(qber) - leakage)
        def binary_entropy(p: float) -> float:
            if p <= 0 or p >= 1:
                return 0.0
            return -p * math.log2(p) - (1-p) * math.log2(1-p)

        leakage_fraction = 0.1  # from reconciliation
        secret_fraction  = max(0.0, 1.0 - binary_entropy(qber) - leakage_fraction)
        n_secret = max(0, int(n_reconciled * secret_fraction))

        # Hash compression (privacy amplification)
        raw_bytes = bytes(alice_reconciled[:n_reconciled].tolist())
        pa_hash   = hashlib.shake_256(raw_bytes).digest(max(1, n_secret // 8))
        alice_key = pa_hash[:max(1, n_secret // 8)]

        raw_bob   = bytes(bob_reconciled[:n_reconciled].tolist())
        bob_key   = hashlib.shake_256(raw_bob).digest(max(1, n_secret // 8))

        # Key rate (bits/second) assuming 1 GHz photon source
        sim_time_s = (time.perf_counter_ns() - t0) / 1e9
        photon_rate = 1e9  # 1 GHz photon source
        wall_time_s = self.n_qubits / photon_rate
        key_rate_bps = n_secret / (wall_time_s + 1e-10)

        secure = qber < self.QBER_SECURITY_THRESHOLD
        if not secure:
            logger.warning(
                f"QBER={qber:.4f} exceeds security threshold={self.QBER_SECURITY_THRESHOLD}. "
                f"{'Eve detected!' if self.channel.eve_present else 'Channel noise too high!'}"
            )

        return BB84Result(
            n_qubits_sent=self.n_qubits,
            n_sifted=n_sifted,
            n_errors_detected=int(errors),
            qber_measured=qber,
            n_final_key_bits=n_secret,
            key_generation_rate_bps=key_rate_bps,
            alice_key=alice_key,
            bob_key=bob_key,
            keys_match=(alice_key == bob_key),
            secure=secure,
            simulation_ms=sim_time_s * 1000,
        )

    def sweep_distance(self, distances: List[float]) -> List[dict]:
        """Run BB84 at multiple distances and return results."""
        results = []
        for d in distances:
            sim = BB84Simulator(
                n_qubits=self.n_qubits,
                distance_km=d,
                intrinsic_qber=self.channel.intrinsic_qber,
            )
            r = sim.run()
            s = r.summary()
            s["distance_km"] = d
            results.append(s)
        return results


# ---------------------------------------------------------------------------
# QKD vs PQC Cost Comparison
# ---------------------------------------------------------------------------

class QKDvsPQCAnalysis:
    """
    Comparative feasibility analysis: QKD vs PQC (Kyber) for hospital networks.

    Metrics:
      - Infrastructure cost (CAPEX, OPEX)
      - Key generation latency
      - Scalability (number of nodes)
      - Post-quantum security level
      - Operational complexity
    """

    QKD_CAPEX_PER_NODE_USD = 100_000   # QKD transmitter/receiver hardware
    QKD_FIBER_COST_PER_KM  = 5_000     # dedicated dark fiber
    QKD_OPEX_PER_YEAR      = 20_000    # maintenance + calibration
    QKD_MAX_DISTANCE_KM    = 80        # without quantum repeaters

    PQC_CAPEX_PER_NODE_USD = 50        # HSM or software module (Kyber)
    PQC_OPEX_PER_YEAR      = 200       # software maintenance
    PQC_LATENCY_MS         = 0.5       # Kyber512 keygen latency (RPi-class)
    PQC_KEY_SIZE_BYTES      = 800      # Kyber512 public key

    def __init__(self, n_hospitals: int = 10, avg_distance_km: float = 15):
        self.n = n_hospitals
        self.dist = avg_distance_km

    def qkd_cost(self, years: int = 5) -> dict:
        n_links = self.n * (self.n - 1) // 2  # full mesh
        capex = (self.n * self.QKD_CAPEX_PER_NODE_USD +
                 n_links * self.dist * self.QKD_FIBER_COST_PER_KM)
        opex  = self.n * self.QKD_OPEX_PER_YEAR * years
        total = capex + opex

        sim = BB84Simulator(n_qubits=100_000, distance_km=self.dist)
        result = sim.run()

        return {
            "technology": "QKD (BB84)",
            "n_nodes": self.n,
            "capex_usd": capex,
            "opex_usd_5yr": opex,
            "total_cost_5yr_usd": total,
            "max_distance_km": self.QKD_MAX_DISTANCE_KM,
            "key_rate_bps": round(result.key_generation_rate_bps, 2),
            "final_key_bits_per_session": result.n_final_key_bits,
            "requires_dedicated_fiber": True,
            "requires_quantum_repeaters_beyond_80km": True,
            "pq_security": "Information-theoretic (unconditional)",
            "scalability": "Poor (O(n²) links for full mesh)",
            "operational_complexity": "Very High",
        }

    def pqc_cost(self, years: int = 5) -> dict:
        capex = self.n * self.PQC_CAPEX_PER_NODE_USD
        opex  = self.n * self.PQC_OPEX_PER_YEAR * years
        total = capex + opex

        return {
            "technology": "PQC (CRYSTALS-Kyber512)",
            "n_nodes": self.n,
            "capex_usd": capex,
            "opex_usd_5yr": opex,
            "total_cost_5yr_usd": total,
            "max_distance_km": "Unlimited (IP network)",
            "key_gen_latency_ms": self.PQC_LATENCY_MS,
            "public_key_bytes": self.PQC_KEY_SIZE_BYTES,
            "requires_dedicated_fiber": False,
            "requires_quantum_repeaters": False,
            "pq_security": "Computational (MLWE hardness, NIST Level 1)",
            "scalability": "Excellent (software, any network)",
            "operational_complexity": "Low",
        }

    def comparison_table(self) -> dict:
        qkd = self.qkd_cost()
        pqc = self.pqc_cost()
        return {
            "qkd": qkd,
            "pqc": pqc,
            "cost_ratio_qkd_vs_pqc": round(qkd["total_cost_5yr_usd"] / max(pqc["total_cost_5yr_usd"], 1), 1),
            "recommendation": (
                "PQC is recommended for most hospital IoMT deployments due to dramatically "
                "lower cost, unlimited range, and sufficient quantum resistance for a "
                "≥15-year horizon. QKD may be justified for ultra-high-security links "
                "(e.g., central hospital ↔ government health data center) where "
                "unconditional security is mandated and budget is not a constraint."
            )
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== SPQR-IoMT: BB84 QKD Simulation ===\n")

    sim = BB84Simulator(n_qubits=50_000, distance_km=10, intrinsic_qber=0.02)
    result = sim.run()
    print("BB84 Result (10 km fiber, no Eve):")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    print("\n--- Eve attack scenario ---")
    sim_eve = BB84Simulator(n_qubits=50_000, distance_km=10, intrinsic_qber=0.02, eve_present=True)
    r_eve = sim_eve.run()
    for k, v in r_eve.summary().items():
        print(f"  {k}: {v}")

    print("\n=== QKD vs PQC Cost Analysis (10 hospitals, 15 km avg) ===\n")
    analysis = QKDvsPQCAnalysis(n_hospitals=10, avg_distance_km=15)
    comp = analysis.comparison_table()
    print(json.dumps(comp, indent=2))
