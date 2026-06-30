"""
hardware_sim/oscilloscope_sim.py
==================================
Software simulation of Hantek 6022BE USB oscilloscope
capturing power traces during crypto operations on STM32F446.

Generates 512-sample power traces per operation that:
  - Kyber (constant-time): flat profile, no data-dependent variation
  - RSA (variable-time): visible data-dependent power spikes
  - Fault injection attacks: anomalous glitches, zeros, phase shifts

Used for:
  - TVLA (Test Vector Leakage Assessment) analysis
  - Side-channel classifier training data
  - Paper Figure 5 (power trace comparison)

Usage:
    scope = OscilloscopeSimulator(sample_rate_mhz=1, trace_len=512)
    traces_kyber = scope.capture_traces('Kyber512', 'keygen', n=1000)
    traces_rsa   = scope.capture_traces('RSA-2048', 'keygen', n=1000)
    scope.save_traces('hardware_sim/results/')
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TraceSet:
    scheme:     str
    operation:  str
    n_traces:   int
    trace_len:  int
    sample_rate_mhz: float
    traces:     np.ndarray   # shape (n_traces, trace_len)
    labels:     np.ndarray   # shape (n_traces,)  0=normal 1=attack
    metadata:   dict


class OscilloscopeSimulator:
    """
    Simulates Hantek 6022BE USB oscilloscope capturing power traces
    from STM32F446 during cryptographic operations.

    Physics model:
      - Supply voltage: 3.3V
      - Shunt resistor: 0.1Ω
      - Signal = V_shunt = I × R_shunt
      - Noise: thermal (Johnson) + quantisation (8-bit ADC)
      - Clock frequency harmonics from CPU activity
    """

    def __init__(self, sample_rate_mhz: float = 1.0,
                 trace_len: int = 512, seed: int = 42):
        self.sample_rate  = sample_rate_mhz * 1e6
        self.trace_len    = trace_len
        self.rng          = np.random.default_rng(seed)
        self.dt           = 1.0 / self.sample_rate          # seconds per sample
        self.t            = np.arange(trace_len) * self.dt  # time axis
        self._stored: Dict[str, TraceSet] = {}

    # ── Physical noise model ──────────────────────────────────────────────

    def _thermal_noise(self, size: int, sigma: float = 0.003) -> np.ndarray:
        """Johnson-Nyquist thermal noise at room temperature."""
        return self.rng.normal(0, sigma, size)

    def _quantisation_noise(self, size: int) -> np.ndarray:
        """8-bit ADC quantisation: ±0.5 LSB = ±0.004V on 2V range."""
        return self.rng.uniform(-0.004, 0.004, size)

    def _clock_harmonics(self, cpu_freq_mhz: float = 180) -> np.ndarray:
        """CPU clock harmonics visible on power supply."""
        f1 = cpu_freq_mhz * 1e6
        harmonics = np.zeros(self.trace_len)
        for k, amp in [(1, 0.008), (2, 0.004), (3, 0.002)]:
            harmonics += amp * np.sin(2 * np.pi * k * f1 * self.t)
        return harmonics

    def _base_noise(self) -> np.ndarray:
        return (self._thermal_noise(self.trace_len) +
                self._quantisation_noise(self.trace_len) +
                self._clock_harmonics())

    # ── Operation-specific power profiles ────────────────────────────────

    def _kyber_trace(self, variant: str = 'Kyber512') -> np.ndarray:
        """
        Kyber power trace: constant-time NTT butterfly operations.
        Power is FLAT — no data-dependent variation (side-channel resistant).
        Distinctive feature: regular high-frequency ripple from NTT stages.
        """
        n_phases = {'Kyber512': 7, 'Kyber768': 10, 'Kyber1024': 13}.get(variant, 7)
        trace = np.zeros(self.trace_len)

        # Base active current: ~42 mA × 0.1Ω = 4.2 mV on shunt
        base_power = 0.0042 + self._base_noise()

        # NTT phases — regular periodic structure
        phase_len = self.trace_len // n_phases
        for i in range(n_phases):
            start = i * phase_len
            end   = min(start + phase_len, self.trace_len)
            length = end - start
            # Each NTT phase has characteristic butterfly pattern
            t_local = np.linspace(0, 2 * np.pi, length)
            ntt_ripple = 0.0008 * np.sin(t_local * 8) + 0.0004 * np.sin(t_local * 16)
            # CONSTANT amplitude — no data dependence
            trace[start:end] = 0.0042 + ntt_ripple + self._base_noise()[:length]

        return trace

    def _rsa_trace(self, key_size: int = 2048) -> np.ndarray:
        """
        RSA power trace: square-and-multiply algorithm.
        DATA-DEPENDENT: power spikes when processing '1' bits vs '0' bits.
        Visibly non-constant — vulnerable to Simple Power Analysis (SPA).
        """
        trace = np.zeros(self.trace_len)
        n_bits = key_size

        # Generate a random private exponent bit pattern
        bits = self.rng.integers(0, 2, n_bits)
        samples_per_bit = max(1, self.trace_len // n_bits)

        for i, bit in enumerate(bits):
            idx = i * samples_per_bit
            if idx >= self.trace_len:
                break
            end = min(idx + samples_per_bit, self.trace_len)

            # Square operation: always performed
            square_power = 0.0058 + self._thermal_noise(end - idx, 0.0004)

            # Multiply: only for '1' bits → visible difference
            if bit == 1:
                # 1-bit: square + multiply → higher power
                mul_spike = 0.0024 * (1 + 0.15 * self.rng.normal())
                trace[idx:end] = square_power + mul_spike
            else:
                # 0-bit: square only → lower power
                trace[idx:end] = square_power + 0.0002 * self.rng.normal(size=end - idx)

        return trace + self._clock_harmonics() * 0.3

    def _ecc_trace(self) -> np.ndarray:
        """
        ECC scalar multiplication: double-and-add.
        Partially data-dependent (vulnerable to some attacks).
        Less visible than RSA due to uniform field operations.
        """
        trace = np.zeros(self.trace_len)
        segments = 32  # 256-bit scalar, ~8 samples per bit
        seg_len  = self.trace_len // segments
        scalar_bits = self.rng.integers(0, 2, segments)

        for i, bit in enumerate(scalar_bits):
            start = i * seg_len
            end   = min(start + seg_len, self.trace_len)
            base  = 0.0048 + self._thermal_noise(end - start, 0.0003)
            if bit:
                trace[start:end] = base + 0.0008 * (1 + 0.05 * self.rng.normal())
            else:
                trace[start:end] = base
        return trace

    def _aes_trace(self) -> np.ndarray:
        """
        AES-256-GCM trace: 14 rounds, each with SubBytes, ShiftRows, MixColumns.
        Modern CPUs use AES-NI → nearly constant-time but slight variation in SubBytes.
        """
        trace = np.zeros(self.trace_len)
        n_rounds = 14
        round_len = self.trace_len // n_rounds
        for r in range(n_rounds):
            start = r * round_len
            end   = min(start + round_len, self.trace_len)
            # SubBytes (SBox lookup) — slight key-dependent variation
            sbox_var = 0.0003 * self.rng.normal(size=end-start)
            trace[start:end] = 0.0035 + sbox_var + self._thermal_noise(end-start, 0.0002)
        return trace

    # ── Attack trace generators ───────────────────────────────────────────

    def _power_glitch_trace(self, base_fn) -> np.ndarray:
        """Simulate voltage glitch attack on normal trace."""
        trace = base_fn()
        pos   = self.rng.integers(20, self.trace_len - 20)
        width = self.rng.integers(2, 8)
        trace[pos:pos+width] += self.rng.uniform(0.02, 0.08)
        return trace

    def _timing_attack_trace(self, base_fn) -> np.ndarray:
        """Simulate timing side-channel: phase-shifted trace."""
        trace = base_fn()
        shift = int(self.rng.integers(5, 25))
        return np.roll(trace, shift)

    def _fault_injection_trace(self, base_fn) -> np.ndarray:
        """Simulate fault injection: zero-out segment."""
        trace = base_fn()
        start = self.rng.integers(40, self.trace_len - 40)
        trace[start:start+20] = self.rng.uniform(-0.001, 0.001, 20)
        trace[start+20:start+25] += 0.025  # recovery spike
        return trace

    # ── Public API ────────────────────────────────────────────────────────

    def capture_traces(self, scheme: str, operation: str = 'keygen',
                       n: int = 1000, include_attacks: bool = True) -> TraceSet:
        """
        Capture N power traces for a scheme/operation combination.
        If include_attacks=True, adds 30% attack traces (labelled 1).
        """
        op_map = {
            'Kyber512':  self._kyber_trace,
            'Kyber768':  lambda: self._kyber_trace('Kyber768'),
            'Kyber1024': lambda: self._kyber_trace('Kyber1024'),
            'RSA-2048':  self._rsa_trace,
            'RSA-4096':  lambda: self._rsa_trace(4096),
            'ECC-P256':  self._ecc_trace,
            'AES-256':   self._aes_trace,
        }

        if scheme not in op_map:
            raise ValueError(f"Unknown scheme '{scheme}'")

        base_fn = op_map[scheme]
        n_normal = int(n * 0.70) if include_attacks else n
        n_attack = n - n_normal

        traces = np.zeros((n, self.trace_len), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)

        # Normal traces
        for i in range(n_normal):
            traces[i] = base_fn()
            labels[i] = 0

        # Attack traces (3 types equally split)
        if include_attacks:
            for i in range(n_attack):
                idx      = n_normal + i
                atk_type = i % 3
                if atk_type == 0:
                    traces[idx] = self._power_glitch_trace(base_fn)
                elif atk_type == 1:
                    traces[idx] = self._timing_attack_trace(base_fn)
                else:
                    traces[idx] = self._fault_injection_trace(base_fn)
                labels[idx] = 1

        # Shuffle
        perm = self.rng.permutation(n)
        traces, labels = traces[perm], labels[perm]

        ts = TraceSet(
            scheme=scheme, operation=operation,
            n_traces=n, trace_len=self.trace_len,
            sample_rate_mhz=self.sample_rate / 1e6,
            traces=traces, labels=labels,
            metadata={
                'n_normal': n_normal, 'n_attack': n_attack,
                'attack_rate': n_attack / n,
                'voltage_range': '2V', 'coupling': 'AC',
                'shunt_ohm': 0.1, 'platform': 'STM32F446RE',
            }
        )
        self._stored[f'{scheme}_{operation}'] = ts
        return ts

    def capture_all_schemes(self, n: int = 1000) -> Dict[str, TraceSet]:
        """Capture traces for all schemes — used for paper Figure 5."""
        results = {}
        for scheme in ['Kyber512', 'RSA-2048', 'ECC-P256', 'AES-256']:
            print(f"  Capturing {n} traces for {scheme}...")
            results[scheme] = self.capture_traces(scheme, 'keygen', n)
        return results

    def save_traces(self, output_dir: str = 'hardware_sim/results/'):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for key, ts in self._stored.items():
            np.save(f'{output_dir}/traces_{key}.npy',  ts.traces)
            np.save(f'{output_dir}/labels_{key}.npy',  ts.labels)
            with open(f'{output_dir}/meta_{key}.json', 'w') as f:
                json.dump(ts.metadata, f, indent=2)
        print(f"Saved {len(self._stored)} trace sets to {output_dir}")

    def tvla_analysis(self, ts: TraceSet) -> dict:
        """
        Test Vector Leakage Assessment via Welch's t-test.
        |t| > 4.5 indicates statistically significant leakage.
        """
        normal = ts.traces[ts.labels == 0]
        attack = ts.traces[ts.labels == 1]
        if len(normal) < 10 or len(attack) < 10:
            return {'error': 'Not enough samples'}

        m1, m2 = normal.mean(0), attack.mean(0)
        v1 = normal.var(0, ddof=1) / len(normal)
        v2 = attack.var(0, ddof=1) / len(attack)
        t  = (m1 - m2) / np.sqrt(v1 + v2 + 1e-12)

        leaking = np.where(np.abs(t) > 4.5)[0]
        return {
            'scheme':           ts.scheme,
            'max_t':            round(float(np.abs(t).max()), 3),
            'n_leaking_points': int(len(leaking)),
            'leakage_detected': len(leaking) > 0,
            'threshold':        4.5,
            'verdict':          'LEAKING' if len(leaking) > 0 else 'CONSTANT-TIME OK',
        }


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print('=== Oscilloscope Simulator ===\n')
    scope = OscilloscopeSimulator(sample_rate_mhz=1, trace_len=512)
    all_ts = scope.capture_all_schemes(n=500)

    # TVLA results
    print('\nTVLA Analysis:')
    for name, ts in all_ts.items():
        r = scope.tvla_analysis(ts)
        print(f"  {name:12s}: max|t|={r['max_t']:.2f}  "
              f"leaking_pts={r['n_leaking_points']}  {r['verdict']}")

    scope.save_traces('hardware_sim/results/')
    print('\nDone.')
