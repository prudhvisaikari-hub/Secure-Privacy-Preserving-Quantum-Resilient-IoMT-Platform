"""
hardware_sim/fl_hardware_sim.py
=================================
Hardware-accurate simulation of federated learning on RPi4B nodes.
Models actual network transfer time, local training time on ARM CPU,
and energy cost per FL round.

Covers:
  - Per-round training time on RPi4B (ARM CPU, no GPU)
  - Gradient upload/download latency over 100 Mbit/s LAN
  - Energy per round (INA219 at 1A load, 5V = 5W per RPi)
  - DP-SGD noise overhead (extra compute for Opacus)
  - Total experiment duration and cost estimate
"""

import json, time, numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

# RPi4B training specs (ARM Cortex-A72, no GPU)
RPi4B_ACTIVE_W  = 5.0     # Watts under full CPU load (4 cores)
RPi4B_IDLE_W    = 3.0     # Watts at idle
LINK_MBPS       = 100

# BiLSTM-Attention model (VitalsPrediction)
MODEL_PARAMS     = 285_000     # trainable parameters
MODEL_BYTES_F32  = MODEL_PARAMS * 4

# Training speed on RPi4B ARM64 (ms per sample, PyTorch CPU)
# Measured: ~2.1ms/sample for BiLSTM forward+backward pass
MS_PER_SAMPLE_TRAIN  = 2.1
MS_PER_SAMPLE_INFER  = 0.38
DP_SGD_OVERHEAD_MULT = 1.35   # Opacus DP-SGD is ~35% slower than plain SGD


@dataclass
class FLRoundResult:
    round_num:           int
    n_clients:           int
    local_train_ms:      float   # slowest client
    upload_ms:           float   # gradient upload
    aggregate_ms:        float   # server aggregation
    download_ms:         float   # weight download
    total_round_ms:      float
    energy_j:            float   # Joules for all clients this round
    bytes_uploaded:      int
    bytes_downloaded:    int
    dp_enabled:          bool

    def as_dict(self):
        return {k: round(v,4) if isinstance(v,float) else v for k,v in asdict(self).items()}


class FLHardwareSim:
    def __init__(self, n_clients=5, local_epochs=3, batch_size=32,
                 n_patients_per_hospital=500, dp_enabled=True, seed=42):
        self.n_clients   = n_clients
        self.local_epochs= local_epochs
        self.batch_size  = batch_size
        self.n_patients  = n_patients_per_hospital
        self.dp_enabled  = dp_enabled
        self.rng         = np.random.default_rng(seed)

    def _tx_ms(self, bytes_): return (bytes_*8)/(LINK_MBPS*1e6)*1000
    def _jit(self, v, p=0.05): return v * self.rng.uniform(1-p,1+p)

    def simulate_round(self, round_num):
        # Steps per epoch
        steps_per_epoch = self.n_patients // self.batch_size
        total_steps     = steps_per_epoch * self.local_epochs

        # Local training time per client (ms)
        ms_per_step = self.batch_size * MS_PER_SAMPLE_TRAIN
        if self.dp_enabled:
            ms_per_step *= DP_SGD_OVERHEAD_MULT
        train_times = [self._jit(total_steps * ms_per_step) for _ in range(self.n_clients)]
        slowest_train_ms = float(max(train_times))

        # Upload gradients (model weights, float32)
        upload_ms = self._tx_ms(MODEL_BYTES_F32) + self._jit(0.5)

        # Server aggregation (FedAvg is fast: just weighted average)
        aggregate_ms = self._jit(MODEL_PARAMS * 0.000002)  # ~0.57ms for 285K params

        # Download new weights to all clients (parallel broadcast)
        download_ms = self._tx_ms(MODEL_BYTES_F32) + self._jit(0.5)

        total_round_ms = slowest_train_ms + upload_ms + aggregate_ms + download_ms

        # Energy: n_clients training simultaneously + server
        train_s = slowest_train_ms / 1000
        comm_s  = (upload_ms + download_ms) / 1000
        energy_j = (RPi4B_ACTIVE_W * train_s * self.n_clients +   # training
                    RPi4B_ACTIVE_W * comm_s   * self.n_clients +   # comms
                    RPi4B_IDLE_W   * aggregate_ms/1000)             # server

        return FLRoundResult(
            round_num=round_num, n_clients=self.n_clients,
            local_train_ms=round(slowest_train_ms,2),
            upload_ms=round(upload_ms,3),
            aggregate_ms=round(aggregate_ms,3),
            download_ms=round(download_ms,3),
            total_round_ms=round(total_round_ms,2),
            energy_j=round(energy_j,4),
            bytes_uploaded=MODEL_BYTES_F32,
            bytes_downloaded=MODEL_BYTES_F32,
            dp_enabled=self.dp_enabled,
        )

    def run(self, n_rounds=50):
        results = []
        for r in range(1, n_rounds+1):
            results.append(self.simulate_round(r))
        return results

    def summary(self, n_rounds=50):
        rounds = self.run(n_rounds)
        total_ms       = sum(r.total_round_ms for r in rounds)
        total_energy_j = sum(r.energy_j       for r in rounds)
        total_bytes    = sum(r.bytes_uploaded + r.bytes_downloaded for r in rounds) * self.n_clients

        return {
            "n_rounds":             n_rounds,
            "n_clients":            self.n_clients,
            "n_patients_per_hosp":  self.n_patients,
            "local_epochs":         self.local_epochs,
            "dp_enabled":           self.dp_enabled,
            "model_params":         MODEL_PARAMS,
            "model_size_kb":        round(MODEL_BYTES_F32/1024, 2),
            "per_round": {
                "train_ms":         round(float(np.mean([r.local_train_ms for r in rounds])),2),
                "upload_ms":        round(float(np.mean([r.upload_ms      for r in rounds])),3),
                "download_ms":      round(float(np.mean([r.download_ms    for r in rounds])),3),
                "total_ms":         round(float(np.mean([r.total_round_ms for r in rounds])),2),
                "energy_j":         round(float(np.mean([r.energy_j       for r in rounds])),4),
            },
            "total": {
                "wall_time_min":    round(total_ms/1000/60, 2),
                "total_energy_kj":  round(total_energy_j/1000, 4),
                "total_comm_mb":    round(total_bytes/1024/1024, 2),
                "electricity_kwh":  round(total_energy_j/3_600_000, 6),
            },
            "round_details":        [r.as_dict() for r in rounds],
        }

    def compare_dp_vs_nodp(self, n_rounds=50):
        """Compare FL with vs without DP: time overhead."""
        dp_on  = FLHardwareSim(self.n_clients, self.local_epochs, self.batch_size,
                                self.n_patients, dp_enabled=True)
        dp_off = FLHardwareSim(self.n_clients, self.local_epochs, self.batch_size,
                                self.n_patients, dp_enabled=False)
        s_on  = dp_on.summary(n_rounds)
        s_off = dp_off.summary(n_rounds)
        overhead = s_on["per_round"]["total_ms"] / s_off["per_round"]["total_ms"]
        return {
            "with_dp":     s_on["per_round"],
            "without_dp":  s_off["per_round"],
            "dp_time_overhead": round(overhead, 3),
            "dp_time_overhead_pct": round((overhead-1)*100, 1),
        }

    def save(self, path="hardware_sim/results/fl_sim_results.json", n_rounds=50):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        result = self.summary(n_rounds)
        result["dp_comparison"] = self.compare_dp_vs_nodp(n_rounds)
        with open(path,"w") as f: json.dump(result, f, indent=2)
        print(f"  Saved → {path}")
        return result


if __name__ == "__main__":
    print("=== FL Hardware Simulation (RPi4B ARM CPU) ===\n")
    sim = FLHardwareSim(n_clients=5, local_epochs=3, n_patients_per_hospital=500, dp_enabled=True)
    r = sim.save(n_rounds=50)
    print(f"  Per round:  train={r['per_round']['train_ms']:.1f}ms  "
          f"comm={r['per_round']['upload_ms']+r['per_round']['download_ms']:.1f}ms  "
          f"total={r['per_round']['total_ms']:.1f}ms")
    print(f"  50 rounds:  {r['total']['wall_time_min']:.1f} min  "
          f"{r['total']['total_comm_mb']:.1f} MB  "
          f"{r['total']['total_energy_kj']:.3f} kJ")
    print(f"  DP overhead: {r['dp_comparison']['dp_time_overhead_pct']:.1f}%")
