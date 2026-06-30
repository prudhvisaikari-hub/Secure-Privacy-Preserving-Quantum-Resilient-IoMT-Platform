# SPQR-IoMT: Secure & Privacy-Preserving Quantum-Resilient IoMT Platform

> A full-stack research platform combining Post-Quantum Cryptography, Federated Learning with Differential Privacy, and AI-driven Intrusion Detection for Internet of Medical Things (IoMT) deployments.

---

## Project Overview

| Dimension | Detail |
|---|---|
| **Domain** | IoMT Security, Post-Quantum Cryptography, Privacy-Preserving ML |
| **Target** | Hospital edge nodes, medical sensors (ventilators, infusion pumps, vitals monitors) |
| **Timeline** | 9 months (see `/docs/roadmap.md`) |
| **Goal** | Papers + Portfolio + Grant/Internship evidence |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SPQR-IoMT Platform                     │
├───────────────┬──────────────────────┬──────────────────────┤
│  PQC Layer    │  Privacy-ML Layer    │  Security-AI Layer   │
│  (Kyber/NTRU) │  (FL + DP + HE)     │  (LSTM IDS + SCAD)  │
├───────────────┴──────────────────────┴──────────────────────┤
│              Hybrid Migration Gateway (RSA ↔ PQC)           │
├─────────────────────────────────────────────────────────────┤
│         IoMT Testbed  (RPi + ESP32 + Simulated Devices)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
SPQR-IoMT/
├── pqc_layer/              # Kyber512/768/1024 implementation + benchmarks
│   ├── kyber_wrapper.py    # Python bindings via liboqs
│   ├── benchmark.py        # Crypto overhead measurement harness
│   ├── comparison.py       # RSA/ECC vs Kyber comparison
│   └── secure_channel.py   # DTLS-style secure telemetry protocol
│
├── federated_learning/     # FL + DP + HE pipeline
│   ├── fl_server.py        # Federated aggregation server (Flower)
│   ├── fl_client.py        # Hospital node client
│   ├── dp_trainer.py       # DP-SGD training with Opacus
│   ├── he_inference.py     # Homomorphic encrypted inference (SEAL/TenSEAL)
│   └── privacy_audit.py    # RDP/DP accounting
│
├── intrusion_detection/    # AI-driven IDS + side-channel detection
│   ├── data_gen.py         # Synthetic attack traffic + power trace generation
│   ├── lstm_ids.py         # LSTM-based anomaly detector
│   ├── transformer_ids.py  # Lightweight transformer variant
│   ├── side_channel.py     # Power trace side-channel classifier
│   └── evaluate.py         # TPR/FPR/ROC/AUC evaluation
│
├── hybrid_migration/       # Legacy RSA → PQC migration gateway
│   ├── gateway.py          # Dual-mode negotiation proxy
│   ├── handshake.py        # Classical/PQC hybrid handshake
│   └── migration_planner.py # Phased rollout simulation
│
├── qkd_comparison/         # BB84 simulation vs PQC feasibility
│   ├── bb84_sim.py         # BB84 protocol simulator
│   ├── channel_noise.py    # Quantum channel noise model
│   └── cost_analysis.py    # QKD vs PQC cost/latency comparison
│
├── benchmarks/             # Unified measurement harness
│   ├── run_all.py          # Run all benchmarks end-to-end
│   ├── energy_meter.py     # INA219/Monsoon power measurement
│   └── results/            # CSV/JSON benchmark outputs
│
├── experiments/            # Reproducible experiment scripts
│   ├── exp1_crypto_overhead.py
│   ├── exp2_fl_dp_tradeoff.py
│   ├── exp3_he_inference.py
│   ├── exp4_ids_detection.py
│   └── exp5_quantum_attack_sim.py
│
├── utils/                  # Shared utilities
│   ├── logger.py
│   ├── metrics.py
│   └── dataset_loader.py
│
├── docs/
│   ├── roadmap.md          # 9-month Gantt roadmap
│   ├── threat_model.md     # Formal threat model
│   ├── paper_outline_1.md  # PQC overhead paper outline
│   └── paper_outline_2.md  # FL+DP+HE in IoMT paper outline
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourname/SPQR-IoMT
cd SPQR-IoMT
pip install -r requirements.txt

# 2. Install liboqs (for Kyber)
sudo apt-get install cmake ninja-build libssl-dev
git clone --depth 1 https://github.com/open-quantum-safe/liboqs
cd liboqs && mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=/usr/local ..
ninja && sudo ninja install

# 3. Run crypto benchmarks
python experiments/exp1_crypto_overhead.py

# 4. Run FL + DP simulation
python experiments/exp2_fl_dp_tradeoff.py

# 5. Run IDS training and evaluation
python experiments/exp4_ids_detection.py
```

---

## Key Papers Targeted

1. **"Lightweight Lattice-Based Cryptography for Resource-Constrained IoMT Devices: A Comparative Overhead Study"** — IEEE TIFS / IoTDI
2. **"Federated Learning with Differential Privacy and Homomorphic Encryption for Multi-Hospital Vitals Prediction"** — IEEE JBHI / NeurIPS Privacy Workshop

---

## Datasets Used

| Dataset | Use |
|---|---|
| MIMIC-III | Vitals / ICU time-series for FL training |
| PhysioNet (MIMIC-IV Waveforms) | ECG/SpO2 for encrypted inference |
| UNSW-NB15 | Network intrusion baseline |
| IoT-23 | IoT-specific attack traffic |
| Custom testbed traces | Power side-channel + telemetry |

---

## License
MIT License — open-source for research reuse.
