"""
hardware_sim/network_testbed.py
=================================
Software simulation of the RPi-to-RPi network testbed.
Simulates: Ethernet latency, packet loss, bandwidth limits,
and end-to-end secure channel performance.

Models a realistic hospital LAN:
  - Gigabit Ethernet between nodes (0.1ms base RTT)
  - 100 Mbit/s effective throughput
  - Occasional packet loss (0.01%)
  - Secure channel handshake over simulated network

Also simulates FL communication overhead:
  - Per-round gradient upload/download
  - Multi-hospital FL with realistic network delays
"""

import time
import json
import socket
import struct
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


# ── Network parameters ───────────────────────────────────────────────────────
LINK_SPEED_MBPS     = 100       # 100 Mbps Ethernet
BASE_RTT_MS         = 0.12      # ~0.12ms on local switch
PACKET_LOSS_RATE    = 0.0001    # 0.01%
SWITCH_LATENCY_MS   = 0.08      # managed switch forwarding delay
MAX_PACKET_BYTES    = 1500      # MTU


@dataclass
class NetworkStats:
    src:           str
    dst:           str
    payload_bytes: int
    tx_time_ms:    float
    rtt_ms:        float
    throughput_mbps: float
    packet_loss:   bool = False


@dataclass
class HandshakeStats:
    variant:           str
    hello_ms:          float
    server_hello_ms:   float
    client_key_ms:     float
    total_handshake_ms: float
    data_tx_ms:        float
    total_rtt_ms:      float
    wire_bytes:        int
    encrypted_payload_bytes: int


class NetworkSimulator:
    """
    Simulates a hospital Gigabit Ethernet LAN with realistic latency/loss.
    """

    def __init__(self, link_speed_mbps: float = LINK_SPEED_MBPS,
                 base_rtt_ms: float = BASE_RTT_MS, seed: int = 42):
        self.link_speed = link_speed_mbps
        self.base_rtt   = base_rtt_ms
        self.rng        = np.random.default_rng(seed)

    def transmission_time_ms(self, payload_bytes: int) -> float:
        """Time to transmit payload_bytes over link_speed_mbps Ethernet."""
        return (payload_bytes * 8) / (self.link_speed * 1e6) * 1000

    def rtt_ms(self, payload_bytes: int) -> float:
        """Full round-trip time including propagation and transmission."""
        tx    = self.transmission_time_ms(payload_bytes)
        prop  = self.base_rtt + SWITCH_LATENCY_MS
        jitter = self.rng.normal(0, 0.01)
        return max(0.05, tx + prop + abs(jitter))

    def dropped(self) -> bool:
        return self.rng.random() < PACKET_LOSS_RATE

    def send(self, src: str, dst: str, payload_bytes: int) -> NetworkStats:
        rtt  = self.rtt_ms(payload_bytes)
        time.sleep(rtt / 1000 * 0.001)  # tiny real sleep for simulation realism
        return NetworkStats(
            src=src, dst=dst, payload_bytes=payload_bytes,
            tx_time_ms=self.transmission_time_ms(payload_bytes),
            rtt_ms=rtt,
            throughput_mbps=round((payload_bytes*8)/(rtt/1000)/1e6, 2),
            packet_loss=self.dropped()
        )


class SecureChannelNetworkTest:
    """
    Simulates the SPQR-IoMT secure channel over the network testbed.
    Measures handshake latency breakdown per Kyber variant.
    """

    # Wire sizes per variant (bytes): hello + server_hello + client_key + data
    WIRE_SIZES = {
        'Kyber512':  {'hello': 64, 'server_hello': 864,  'client_key': 832,  'data_overhead': 96},
        'Kyber768':  {'hello': 64, 'server_hello': 1248, 'client_key': 1152, 'data_overhead': 96},
        'Kyber1024': {'hello': 64, 'server_hello': 1632, 'client_key': 1632, 'data_overhead': 96},
        'RSA-2048':  {'hello': 64, 'server_hello': 358,  'client_key': 288,  'data_overhead': 64},
    }

    # Crypto computation time per phase (ms, from RPi emulator)
    CRYPTO_TIMES = {
        'Kyber512':  {'server_keygen': 0.311, 'client_encaps': 0.378, 'server_decaps': 0.401,
                      'aes_encrypt': 0.042, 'aes_decrypt': 0.038},
        'Kyber768':  {'server_keygen': 0.512, 'client_encaps': 0.621, 'server_decaps': 0.655,
                      'aes_encrypt': 0.051, 'aes_decrypt': 0.045},
        'Kyber1024': {'server_keygen': 0.731, 'client_encaps': 0.889, 'server_decaps': 0.932,
                      'aes_encrypt': 0.063, 'aes_decrypt': 0.057},
        'RSA-2048':  {'server_keygen': 48.72, 'client_encaps': 0.951, 'server_decaps': 18.24,
                      'aes_encrypt': 0.891, 'aes_decrypt': 0.038},
    }

    def __init__(self, net: NetworkSimulator):
        self.net = net
        self.rng = np.random.default_rng(42)

    def _jitter(self, val: float, pct: float = 0.03) -> float:
        return val * self.rng.uniform(1 - pct, 1 + pct)

    def run_handshake(self, variant: str,
                      payload_bytes: int = 64) -> HandshakeStats:
        """Simulate full secure channel handshake + one data message."""
        ws = self.WIRE_SIZES.get(variant, self.WIRE_SIZES['Kyber512'])
        ct = self.CRYPTO_TIMES.get(variant, self.CRYPTO_TIMES['Kyber512'])

        # Phase 1: ClientHello (sensor → gateway)
        net1 = self.net.send('sensor', 'gateway', ws['hello'])
        hello_ms = net1.rtt_ms

        # Phase 2: ServerHello (gateway → sensor)
        # Gateway must first generate key pair
        crypto_kg = self._jitter(ct['server_keygen'])
        net2 = self.net.send('gateway', 'sensor', ws['server_hello'])
        server_hello_ms = crypto_kg + net2.rtt_ms

        # Phase 3: ClientKey (sensor → gateway)
        # Sensor must encapsulate
        crypto_enc = self._jitter(ct['client_encaps'])
        net3 = self.net.send('sensor', 'gateway', ws['client_key'])
        # Gateway must decapsulate
        crypto_dec = self._jitter(ct['server_decaps'])
        client_key_ms = crypto_enc + net3.rtt_ms + crypto_dec

        total_hs = hello_ms + server_hello_ms + client_key_ms

        # Data message
        data_wire = payload_bytes + ws['data_overhead']
        net4 = self.net.send('sensor', 'gateway', data_wire)
        data_ms = self._jitter(ct['aes_encrypt']) + net4.rtt_ms + self._jitter(ct['aes_decrypt'])

        return HandshakeStats(
            variant=variant,
            hello_ms=round(hello_ms, 4),
            server_hello_ms=round(server_hello_ms, 4),
            client_key_ms=round(client_key_ms, 4),
            total_handshake_ms=round(total_hs, 4),
            data_tx_ms=round(data_ms, 4),
            total_rtt_ms=round(total_hs + data_ms, 4),
            wire_bytes=ws['hello'] + ws['server_hello'] + ws['client_key'] + data_wire,
            encrypted_payload_bytes=payload_bytes,
        )

    def benchmark(self, variants: List[str] = None,
                  n_iterations: int = 100,
                  payload_bytes: int = 64) -> List[dict]:
        if variants is None:
            variants = ['Kyber512', 'Kyber768', 'Kyber1024', 'RSA-2048']
        results = []
        for variant in variants:
            hs_times, data_times = [], []
            for _ in range(n_iterations):
                r = self.run_handshake(variant, payload_bytes)
                hs_times.append(r.total_handshake_ms)
                data_times.append(r.data_tx_ms)

            results.append({
                'variant':                variant,
                'handshake_mean_ms':      round(float(np.mean(hs_times)), 4),
                'handshake_std_ms':       round(float(np.std(hs_times)),  4),
                'handshake_p95_ms':       round(float(np.percentile(hs_times, 95)), 4),
                'data_tx_mean_ms':        round(float(np.mean(data_times)), 4),
                'total_mean_ms':          round(float(np.mean(hs_times) + np.mean(data_times)), 4),
                'wire_bytes':             self.run_handshake(variant, payload_bytes).wire_bytes,
                'realtime_ok':            np.mean(hs_times) < 50.0,
                'iterations':             n_iterations,
            })
            print(f"  {variant:12s}: hs={results[-1]['handshake_mean_ms']:.2f}ms  "
                  f"data={results[-1]['data_tx_mean_ms']:.3f}ms  "
                  f"{'✓' if results[-1]['realtime_ok'] else '✗'}")
        return results


class FLNetworkSimulator:
    """
    Simulates federated learning communication overhead.
    Models gradient upload/download per round across N hospitals.
    """

    def __init__(self, net: NetworkSimulator,
                 model_params: int = 285_000,    # BiLSTM-Attention params
                 n_hospitals: int = 5):
        self.net         = net
        self.model_bytes = model_params * 4       # float32
        self.n_hospitals = n_hospitals

    def simulate_round(self, round_num: int,
                       compression_ratio: float = 1.0) -> dict:
        """Simulate one FL round: local train + upload + aggregate + download."""
        bytes_per_client = int(self.model_bytes * compression_ratio)
        upload_times, download_times = [], []

        for i in range(self.n_hospitals):
            # Upload: client → server
            up = self.net.send(f'hospital_{i}', 'fl_server', bytes_per_client)
            upload_times.append(up.rtt_ms)
            # Download: server → client
            down = self.net.send('fl_server', f'hospital_{i}', bytes_per_client)
            download_times.append(down.rtt_ms)

        return {
            'round':              round_num,
            'n_clients':          self.n_hospitals,
            'bytes_per_client':   bytes_per_client,
            'total_upload_kb':    round(bytes_per_client * self.n_hospitals / 1024, 2),
            'total_download_kb':  round(bytes_per_client * self.n_hospitals / 1024, 2),
            'total_comm_kb':      round(bytes_per_client * self.n_hospitals * 2 / 1024, 2),
            'max_upload_ms':      round(float(max(upload_times)), 3),
            'max_download_ms':    round(float(max(download_times)), 3),
            'total_round_comm_ms': round(float(max(upload_times) + max(download_times)), 3),
            'compression_ratio':  compression_ratio,
        }

    def simulate_training(self, n_rounds: int = 50,
                          compression_ratio: float = 1.0) -> List[dict]:
        results = []
        for r in range(1, n_rounds + 1):
            results.append(self.simulate_round(r, compression_ratio))
        return results

    def overhead_summary(self, n_rounds: int = 50) -> dict:
        rounds = self.simulate_training(n_rounds)
        total_kb = sum(r['total_comm_kb'] for r in rounds)
        return {
            'n_rounds':           n_rounds,
            'n_hospitals':        self.n_hospitals,
            'model_size_kb':      round(self.model_bytes / 1024, 2),
            'total_comm_mb':      round(total_kb / 1024, 3),
            'per_round_kb':       round(total_kb / n_rounds, 2),
            'max_round_delay_ms': round(max(r['total_round_comm_ms'] for r in rounds), 3),
        }


def run_full_network_benchmark(output_dir: str = 'hardware_sim/results/') -> dict:
    """Run complete network testbed simulation."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    net = NetworkSimulator()

    print('\n=== Secure Channel Network Benchmark ===')
    sc_test = SecureChannelNetworkTest(net)
    sc_results = sc_test.benchmark(n_iterations=100)

    print('\n=== FL Communication Overhead ===')
    fl_sim = FLNetworkSimulator(net, n_hospitals=5)
    fl_summary = fl_sim.overhead_summary(n_rounds=50)
    print(f"  Model size:      {fl_summary['model_size_kb']:.1f} KB")
    print(f"  Total comm (50 rounds, 5 hospitals): {fl_summary['total_comm_mb']:.2f} MB")
    print(f"  Per round:       {fl_summary['per_round_kb']:.1f} KB")
    print(f"  Max round delay: {fl_summary['max_round_delay_ms']:.1f} ms")

    output = {'secure_channel': sc_results, 'fl_overhead': fl_summary}
    path = f'{output_dir}/network_benchmark.json'
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\nResults saved to {path}')
    return output


if __name__ == '__main__':
    run_full_network_benchmark()
