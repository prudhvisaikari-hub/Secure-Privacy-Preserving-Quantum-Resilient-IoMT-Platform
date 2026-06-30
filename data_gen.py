"""
intrusion_detection/data_gen.py
================================
Synthetic dataset generator for IDS training and evaluation.
Produces:
  1. Network flow sequences  (mimics UNSW-NB15 / IoT-23 format)
  2. Power trace sequences   (mimics STM32 oscilloscope captures)
  3. Telemetry anomaly sequences (abnormal vitals patterns)
  4. Labelled PCAP-style feature records (for offline evaluation)

All generators are seeded for reproducibility and export to
NumPy arrays or CSV files.

Usage:
    from intrusion_detection.data_gen import DatasetFactory
    X_net, y_net = DatasetFactory.network_flows(n_samples=5000)
    X_pwr, y_pwr = DatasetFactory.power_traces(n_samples=2000)
    DatasetFactory.save_csv(X_net, y_net, "data/network_flows.csv")
"""

import csv
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attack class definitions
# ---------------------------------------------------------------------------

NETWORK_CLASSES: Dict[int, str] = {
    0: "Normal",
    1: "DoS_UDP_Flood",
    2: "DoS_TCP_SYN",
    3: "Reconnaissance_PortScan",
    4: "MITM_ARPSpoof",
    5: "Backdoor_C2",
    6: "Fuzzing",
    7: "Injection_Replay",
}

POWER_CLASSES: Dict[int, str] = {
    0: "Normal_Crypto",
    1: "Power_Glitch_Attack",
    2: "Timing_Side_Channel",
    3: "Fault_Injection_DFA",
}

VITALS_CLASSES: Dict[int, str] = {
    0: "Normal",
    1: "Sensor_Spoof",
    2: "Replay_Old_Data",
    3: "Sudden_Disconnect",
}

# ---------------------------------------------------------------------------
# Network flow generator
# ---------------------------------------------------------------------------

class NetworkFlowDataGen:
    """
    Generates synthetic network flow feature sequences.

    Feature vector (20-dim per timestep):
      0  dur          connection duration (s)
      1  proto_tcp    (bool)
      2  proto_udp    (bool)
      3  proto_icmp   (bool)
      4  sbytes       source bytes
      5  dbytes       destination bytes
      6  rate         packets/sec
      7  sload        source bits/sec
      8  dload        destination bits/sec
      9  spkts        source packets
      10 dpkts        destination packets
      11 sttl         source TTL
      12 dttl         destination TTL
      13 sinpkt       inter-packet time (src)
      14 dinpkt       inter-packet time (dst)
      15 tcprtt       TCP round-trip time
      16 synack       SYN-ACK time
      17 ackdat       ACK-data time
      18 smean        mean src packet size
      19 dmean        mean dst packet size
    """
    N_FEATURES = 20
    N_CLASSES  = len(NETWORK_CLASSES)

    def __init__(self, seq_len: int = 20, seed: int = 42):
        self.seq_len = seq_len
        self.rng     = np.random.default_rng(seed)

    # ---- per-class generators ----

    def _normal(self) -> np.ndarray:
        f = np.zeros((self.seq_len, self.N_FEATURES), dtype=np.float32)
        f[:, 0]  = self.rng.exponential(0.5,  self.seq_len)   # dur
        f[:, 1]  = 1.0                                          # TCP
        f[:, 4]  = self.rng.exponential(500,  self.seq_len)   # sbytes
        f[:, 5]  = self.rng.exponential(300,  self.seq_len)   # dbytes
        f[:, 6]  = self.rng.normal(10, 2,     self.seq_len)   # rate
        f[:, 9]  = self.rng.integers(5, 20,   self.seq_len)   # spkts
        f[:, 10] = self.rng.integers(3, 15,   self.seq_len)   # dpkts
        f[:, 11] = self.rng.normal(64, 2,     self.seq_len)   # sttl
        f[:, 12] = self.rng.normal(64, 2,     self.seq_len)   # dttl
        f[:, 18] = self.rng.normal(500, 50,   self.seq_len)   # smean
        f[:, 19] = self.rng.normal(300, 40,   self.seq_len)   # dmean
        return np.clip(f, 0, None)

    def _dos_udp(self) -> np.ndarray:
        f = self._normal()
        f[:, 1]  = 0.0;  f[:, 2] = 1.0            # UDP
        f[:, 6]  = self.rng.normal(2000, 200, self.seq_len)  # very high rate
        f[:, 9]  = self.rng.integers(500, 2000, self.seq_len) # flood packets
        f[:, 5]  = self.rng.exponential(20, self.seq_len)    # tiny replies
        f[:, 4]  = self.rng.normal(64, 5, self.seq_len)      # small fixed payload
        return np.clip(f, 0, None)

    def _dos_syn(self) -> np.ndarray:
        f = self._normal()
        f[:, 6]  = self.rng.normal(5000, 500, self.seq_len)
        f[:, 9]  = self.rng.integers(1000, 5000, self.seq_len)
        f[:, 10] = np.zeros(self.seq_len)           # no replies (half-open)
        f[:, 16] = self.rng.exponential(10, self.seq_len)  # long synack
        return np.clip(f, 0, None)

    def _recon_portscan(self) -> np.ndarray:
        f = self._normal()
        f[:, 0]  = self.rng.exponential(0.005, self.seq_len)  # very short
        f[:, 4]  = self.rng.normal(40, 5, self.seq_len)
        f[:, 5]  = self.rng.normal(20, 3, self.seq_len)
        f[:, 9]  = np.ones(self.seq_len) * 1
        f[:, 10] = np.ones(self.seq_len) * 1
        return np.clip(f, 0, None)

    def _mitm_arp(self) -> np.ndarray:
        f = self._normal()
        f[:, 11] = self.rng.normal(200, 5, self.seq_len)   # anomalous TTL
        f[:, 12] = self.rng.normal(200, 5, self.seq_len)
        f[:, 4]  *= 2.8                                     # extra bytes (interception)
        f[:, 5]  *= 2.8
        return np.clip(f, 0, None)

    def _backdoor(self) -> np.ndarray:
        f = self._normal()
        f[:, 6]  = self.rng.normal(0.5, 0.2, self.seq_len)  # low rate, beacon-like
        f[:, 0]  = self.rng.normal(30, 5, self.seq_len)     # long connections
        f[:, 4]  = self.rng.normal(200, 20, self.seq_len)   # small but regular
        return np.clip(f, 0, None)

    def _fuzzing(self) -> np.ndarray:
        f = self.rng.uniform(0, 3000, (self.seq_len, self.N_FEATURES)).astype(np.float32)
        f[:, 11] = self.rng.uniform(0, 255, self.seq_len)   # random TTL
        return f

    def _injection_replay(self) -> np.ndarray:
        f = self._normal()
        # Replayed packets have identical sizes (anomalously regular)
        common_size = float(self.rng.integers(100, 600))
        f[:, 18] = np.full(self.seq_len, common_size)
        f[:, 19] = np.full(self.seq_len, common_size * 0.6)
        f[:, 13] = np.zeros(self.seq_len)  # perfectly timed inter-packet
        return f

    def _gen_map(self) -> Dict:
        return {
            0: self._normal,
            1: self._dos_udp,
            2: self._dos_syn,
            3: self._recon_portscan,
            4: self._mitm_arp,
            5: self._backdoor,
            6: self._fuzzing,
            7: self._injection_replay,
        }

    def generate(self, n_samples: int = 3000,
                 class_weights: Optional[Dict[int, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate balanced dataset with configurable class proportions.

        Args:
            n_samples:     Total number of flow sequences.
            class_weights: Dict mapping class_id → fraction (must sum to 1).
                           Default: 60% normal, rest split equally.
        Returns:
            (X, y): X shape (n_samples, seq_len, N_FEATURES), y shape (n_samples,)
        """
        if class_weights is None:
            n_atk_classes = self.N_CLASSES - 1
            atk_frac = 0.4 / n_atk_classes
            class_weights = {0: 0.60, **{i: atk_frac for i in range(1, self.N_CLASSES)}}

        gen_map = self._gen_map()
        X, y = [], []
        for cls_id, frac in class_weights.items():
            n = max(1, int(n_samples * frac))
            for _ in range(n):
                X.append(gen_map[cls_id]())
                y.append(cls_id)

        X = np.array(X[:n_samples], dtype=np.float32)
        y = np.array(y[:n_samples], dtype=np.int64)

        # Normalise features
        flat = X.reshape(-1, self.N_FEATURES)
        mu   = flat.mean(0)
        sig  = flat.std(0) + 1e-8
        X    = (X - mu) / sig

        # Shuffle
        idx = self.rng.permutation(len(X))
        return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Power trace generator
# ---------------------------------------------------------------------------

class PowerTraceDataGen:
    """
    Generates synthetic power traces for side-channel analysis.
    Simulates an ARM Cortex-M4 running Kyber512 keygen.

    Trace = 512 power samples at 1 MHz sampling rate (≈ 0.5 ms window).
    """
    N_FEATURES = 1   # single channel (raw power)
    N_CLASSES  = len(POWER_CLASSES)

    def __init__(self, trace_len: int = 512, seed: int = 42):
        self.trace_len = trace_len
        self.rng = np.random.default_rng(seed)

    def _base_trace(self) -> np.ndarray:
        """Normal Kyber keygen power profile."""
        t = np.linspace(0, 6 * np.pi, self.trace_len)
        return (
            0.8 * np.sin(t) + 0.35 * np.sin(3 * t) + 0.15 * np.sin(7 * t)
            + self.rng.normal(0, 0.04, self.trace_len)
        ).astype(np.float32)

    def _power_glitch(self) -> np.ndarray:
        t = self._base_trace()
        pos = self.rng.integers(20, self.trace_len - 20)
        width = self.rng.integers(2, 6)
        t[pos:pos+width] += self.rng.uniform(3.0, 8.0)
        return t

    def _timing_attack(self) -> np.ndarray:
        t = self._base_trace()
        shift = int(self.rng.integers(10, 40))
        return np.roll(t, shift)

    def _fault_injection_dfa(self) -> np.ndarray:
        t = self._base_trace()
        start = self.rng.integers(80, self.trace_len - 40)
        t[start:start+20] = self.rng.uniform(-0.1, 0.1, 20)  # zero-out segment
        t[start+20:start+25] += 2.0                            # recovery spike
        return t

    def generate(self, n_samples: int = 2000,
                 class_weights: Optional[Dict[int, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        if class_weights is None:
            class_weights = {0: 0.70, 1: 0.10, 2: 0.10, 3: 0.10}

        gen_map = {
            0: self._base_trace,
            1: self._power_glitch,
            2: self._timing_attack,
            3: self._fault_injection_dfa,
        }
        X, y = [], []
        for cls_id, frac in class_weights.items():
            n = max(1, int(n_samples * frac))
            for _ in range(n):
                X.append(gen_map[cls_id]().reshape(-1, 1))
                y.append(cls_id)

        X = np.array(X[:n_samples], dtype=np.float32)
        y = np.array(y[:n_samples], dtype=np.int64)
        idx = self.rng.permutation(len(X))
        return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Vitals anomaly generator
# ---------------------------------------------------------------------------

class VitalsAnomalyDataGen:
    """
    Generates labelled vitals telemetry for anomaly detection.
    Detects sensor spoofing, replay attacks, and disconnection patterns.
    """
    N_FEATURES = 5   # HR, SpO2, RR, SBP, Temp
    N_CLASSES  = len(VITALS_CLASSES)

    def __init__(self, seq_len: int = 24, seed: int = 42):
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def _normal(self) -> np.ndarray:
        means = np.array([75, 97, 16, 120, 36.8])
        stds  = np.array([8,   2,  3,  12,   0.3])
        return (self.rng.normal(means, stds, (self.seq_len, self.N_FEATURES))).astype(np.float32)

    def _sensor_spoof(self) -> np.ndarray:
        f = self._normal()
        # Perfect constant values — impossible in real physiology
        f[12:, :] = np.array([72.0, 98.0, 16.0, 118.0, 36.8])
        return f

    def _replay(self) -> np.ndarray:
        f = self._normal()
        # Exact repetition of first 8 timesteps
        f[8:16, :] = f[:8, :]
        return f

    def _disconnect(self) -> np.ndarray:
        f = self._normal()
        # Zeros after disconnect point
        cut = self.rng.integers(8, 16)
        f[cut:, :] = 0.0
        return f

    def generate(self, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        gen_map = {0: self._normal, 1: self._sensor_spoof, 2: self._replay, 3: self._disconnect}
        X, y = [], []
        weights = {0: 0.55, 1: 0.15, 2: 0.15, 3: 0.15}
        for cls_id, frac in weights.items():
            n = max(1, int(n_samples * frac))
            for _ in range(n):
                X.append(gen_map[cls_id]())
                y.append(cls_id)
        X = np.array(X[:n_samples], dtype=np.float32)
        y = np.array(y[:n_samples], dtype=np.int64)
        idx = self.rng.permutation(len(X))
        return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

class DatasetFactory:
    """One-stop factory for all IDS datasets."""

    @staticmethod
    def network_flows(n_samples: int = 3000, seq_len: int = 20,
                      seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        return NetworkFlowDataGen(seq_len, seed).generate(n_samples)

    @staticmethod
    def power_traces(n_samples: int = 2000, trace_len: int = 512,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        return PowerTraceDataGen(trace_len, seed).generate(n_samples)

    @staticmethod
    def vitals_anomalies(n_samples: int = 2000, seq_len: int = 24,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        return VitalsAnomalyDataGen(seq_len, seed).generate(n_samples)

    @staticmethod
    def save_csv(X: np.ndarray, y: np.ndarray, path: str):
        """Save flat (n_samples, seq_len × n_features + 1) CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        flat = X.reshape(len(X), -1)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"f{i}" for i in range(flat.shape[1])] + ["label"])
            for row, label in zip(flat, y):
                w.writerow(list(row) + [int(label)])
        logger.info(f"Dataset saved to {path} ({len(X)} samples)")

    @staticmethod
    def save_json_meta(info: dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== DatasetFactory Demo ===")
    X_net, y_net = DatasetFactory.network_flows(500)
    print(f"Network flows: X={X_net.shape}, y={y_net.shape}, classes={np.unique(y_net)}")
    X_pwr, y_pwr = DatasetFactory.power_traces(400)
    print(f"Power traces:  X={X_pwr.shape}, y={y_pwr.shape}, classes={np.unique(y_pwr)}")
    X_vit, y_vit = DatasetFactory.vitals_anomalies(400)
    print(f"Vitals anomaly: X={X_vit.shape}, y={y_vit.shape}, classes={np.unique(y_vit)}")
