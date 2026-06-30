"""
qkd_comparison/channel_noise.py
================================
Quantum channel noise models for BB84 QKD simulation.
Models optical fiber transmission including:
  - Photon loss (Beer-Lambert attenuation)
  - Depolarising noise (birefringence, thermal fluctuations)
  - Dark counts (detector background noise)
  - Multi-photon pulse vulnerabilities (PNS attack)
  - Eve's intercept-and-resend attack
  - Eve's beam-splitting attack (for weak coherent pulses)

Usage:
    from qkd_comparison.channel_noise import FiberChannel, EveModel
    channel = FiberChannel(distance_km=20, detector_efficiency=0.1)
    print(channel.expected_qber())
    print(channel.info())
"""

import math
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants / fiber parameters
# ---------------------------------------------------------------------------

FIBER_ATTENUATION_DB_PER_KM = 0.2    # Standard SMF-28 telecom fiber
DETECTOR_DARK_COUNT_RATE_HZ  = 1e-5  # Fraction of clock rate (InGaAs APD)
MISALIGNMENT_ERROR           = 0.005  # 0.5% due to polarisation drift
OPTICAL_COMPONENT_LOSS_DB    = 3.0   # Beam-splitters, couplers, etc.


# ---------------------------------------------------------------------------
# Fiber Channel Model
# ---------------------------------------------------------------------------

class FiberChannel:
    """
    Models a single-mode optical fiber quantum channel.

    Parameters mirror real-world deployed QKD systems (e.g.,
    Toshiba QKD, ID Quantique Clavis) at telecom wavelengths (1310/1550 nm).
    """

    def __init__(self,
                 distance_km:          float = 10.0,
                 attenuation_db_km:    float = FIBER_ATTENUATION_DB_PER_KM,
                 detector_efficiency:  float = 0.10,
                 dark_count_rate:      float = DETECTOR_DARK_COUNT_RATE_HZ,
                 misalignment_error:   float = MISALIGNMENT_ERROR,
                 component_loss_db:    float = OPTICAL_COMPONENT_LOSS_DB):
        self.distance_km         = distance_km
        self.attenuation_db_km   = attenuation_db_km
        self.detector_efficiency = detector_efficiency
        self.dark_count_rate     = dark_count_rate
        self.misalignment_error  = misalignment_error
        self.component_loss_db   = component_loss_db

    @property
    def total_loss_db(self) -> float:
        return self.attenuation_db_km * self.distance_km + self.component_loss_db

    @property
    def transmittance(self) -> float:
        """Probability a photon survives channel transit."""
        return 10 ** (-self.total_loss_db / 10)

    @property
    def detection_probability(self) -> float:
        """Probability a transmitted photon triggers detector click."""
        return self.transmittance * self.detector_efficiency

    def raw_key_rate(self, source_rate_hz: float = 1e9, mean_photon: float = 0.1) -> float:
        """
        Raw sifted key bit rate (bits/second) for weak coherent pulse source.
        mu = mean photon number per pulse (default 0.1 for WCP).
        """
        # Probability of at least one photon arriving and detected
        p_detect = 1 - math.exp(-mean_photon * self.detection_probability)
        p_sift   = 0.5     # ~50% basis match
        return source_rate_hz * p_detect * p_sift

    def expected_qber(self) -> float:
        """
        Expected QBER from optical noise only (no Eve).
        QBER = e_opt + e_dark
        e_opt  = misalignment error
        e_dark = dark count contribution
        """
        p_detect = self.detection_probability
        # Dark count QBER contribution
        e_dark = self.dark_count_rate / (2 * max(p_detect, self.dark_count_rate))
        return self.misalignment_error + e_dark

    def secure_key_rate(self, source_rate_hz: float = 1e9, mean_photon: float = 0.1) -> float:
        """
        Secret key rate after error correction and privacy amplification.
        Uses the BB84 secret key fraction: r = 1 - H(e) - H(e)
        where H(e) = binary entropy of QBER.
        """
        qber = self.expected_qber()
        if qber >= 0.11:  # Security threshold
            return 0.0
        def h(p):
            if p <= 0 or p >= 1:
                return 0.0
            return -p * math.log2(p) - (1-p) * math.log2(1-p)
        secret_fraction = max(0.0, 1 - 2 * h(qber))
        return self.raw_key_rate(source_rate_hz, mean_photon) * secret_fraction

    def info(self) -> dict:
        return {
            "distance_km":         self.distance_km,
            "total_loss_db":       round(self.total_loss_db, 2),
            "transmittance":       round(self.transmittance, 6),
            "detection_prob":      round(self.detection_probability, 6),
            "expected_qber":       round(self.expected_qber(), 4),
            "raw_key_rate_bps":    round(self.raw_key_rate(), 1),
            "secret_key_rate_bps": round(self.secure_key_rate(), 1),
            "secure":              self.expected_qber() < 0.11,
        }


# ---------------------------------------------------------------------------
# Eve Attack Models
# ---------------------------------------------------------------------------

class EveModel:
    """
    Models different eavesdropping strategies and their detectability.
    Each attack produces an QBER increase that Alice and Bob can detect.
    """

    def __init__(self, channel: FiberChannel):
        self.channel = channel

    def intercept_resend_qber(self, intercept_fraction: float = 1.0) -> float:
        """
        Intercept-and-Resend attack.
        Eve measures each photon and re-sends. She guesses basis 50% correctly.
        When she guesses wrong, she introduces 25% error (random state resent).
        ΔQBER = 0.25 × intercept_fraction
        """
        delta_qber = 0.25 * intercept_fraction
        return self.channel.expected_qber() + delta_qber

    def beam_splitting_qber(self, eve_bs_ratio: float = 0.1) -> float:
        """
        Beam-Splitting (BS) attack on weak coherent pulse (WCP) sources.
        Eve taps eve_bs_ratio fraction of pulses.
        Detectable only if multi-photon pulses are present.
        Eve gains information from multi-photon pulses without causing errors.
        Returns QBER (unchanged — BS attack is passive on WCP).
        """
        # BS attack doesn't increase QBER on single-photon pulses
        # It exploits multi-photon pulses (PNS vulnerability)
        # Detected by monitoring photon number statistics
        return self.channel.expected_qber()

    def trojan_horse_qber(self) -> float:
        """
        Trojan Horse Attack: Eve injects bright light to probe Alice's optics.
        Does not change QBER but requires optical isolation on Alice's side.
        """
        return self.channel.expected_qber()  # Passive — no QBER change

    def detection_probability(self, attack: str = "intercept_resend",
                               intercept_fraction: float = 1.0) -> dict:
        """
        Probability of detecting Eve's attack given N sifted bits.
        Eve is detected when measured QBER significantly exceeds expected QBER.
        """
        if attack == "intercept_resend":
            delta = 0.25 * intercept_fraction
        elif attack == "beam_splitting":
            delta = 0.0  # Undetectable via QBER
        else:
            delta = 0.0

        measured_qber  = self.channel.expected_qber() + delta
        detection_thr  = 0.11  # Standard threshold

        # Prob of NOT detecting after n bits
        # P(undetected | n bits) ≈ (1 - delta)^n for large n
        def p_undetected(n: int) -> float:
            if delta == 0:
                return 1.0
            return (1 - delta) ** n

        return {
            "attack":              attack,
            "delta_qber":          round(delta, 4),
            "measured_qber":       round(measured_qber, 4),
            "detectable_by_qber":  measured_qber > detection_thr,
            "p_undetected_100":    round(p_undetected(100), 6),
            "p_undetected_1000":   round(p_undetected(1000), 8),
            "n_bits_99pct_detect": math.ceil(math.log(0.01) / math.log(1 - delta + 1e-12)) if delta > 0 else -1,
        }

    def full_analysis(self) -> dict:
        return {
            "channel":   self.channel.info(),
            "attacks": {
                "intercept_resend":  self.detection_probability("intercept_resend", 1.0),
                "partial_ir_50pct":  self.detection_probability("intercept_resend", 0.5),
                "beam_splitting":    self.detection_probability("beam_splitting"),
            },
            "countermeasures": [
                "Use single-photon sources (not WCP) to eliminate PNS attacks",
                "Implement decoy-state protocol to detect beam-splitting",
                "Monitor optical insertion loss for Trojan Horse detection",
                "Apply privacy amplification proportional to estimated Eve info",
            ],
        }


# ---------------------------------------------------------------------------
# Channel comparison across distances
# ---------------------------------------------------------------------------

def distance_performance_table(distances_km: Optional[list] = None) -> list:
    """Performance metrics for standard SMF-28 fiber at various distances."""
    if distances_km is None:
        distances_km = [1, 5, 10, 20, 40, 60, 80, 100]
    rows = []
    for d in distances_km:
        ch = FiberChannel(d)
        rows.append({
            "distance_km":    d,
            "loss_db":        round(ch.total_loss_db, 1),
            "qber":           round(ch.expected_qber(), 4),
            "raw_rate_bps":   round(ch.raw_key_rate(), 1),
            "secret_rate_bps": round(ch.secure_key_rate(), 1),
            "feasible":       ch.secure_key_rate() > 0,
        })
    return rows


if __name__ == "__main__":
    print("=== Fiber Channel Performance ===\n")
    table = distance_performance_table()
    print(f"  {'Dist':>6} | {'Loss(dB)':>8} | {'QBER':>7} | {'SecretRate':>12} | Feasible")
    print(f"  {'-'*60}")
    for r in table:
        print(f"  {r['distance_km']:>6} | {r['loss_db']:>8} | {r['qber']:>7.4f} | {r['secret_rate_bps']:>12.1f} | {'✓' if r['feasible'] else '✗'}")

    print("\n=== Eve Attack Analysis (10 km link) ===\n")
    ch  = FiberChannel(10)
    eve = EveModel(ch)
    analysis = eve.full_analysis()
    import json
    print(json.dumps(analysis, indent=2))
