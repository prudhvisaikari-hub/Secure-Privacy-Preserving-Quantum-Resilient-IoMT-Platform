# SPQR-IoMT: 9-Month Research Roadmap

**Project:** Secure & Privacy-Preserving Quantum-Resilient IoMT Platform  
**Duration:** 9 months (M1–M9)  
**Goal:** 2 papers + open-source toolkit + grant/internship portfolio

---

## Phase Overview

```
PHASE 1: FOUNDATIONS (M1–M2)
PHASE 2: PQC LAYER (M3–M4)
PHASE 3: PRIVACY-ML LAYER (M5–M6)
PHASE 4: SECURITY-AI LAYER (M7)
PHASE 5: INTEGRATION & WRITING (M8–M9)
```

---

## Month-by-Month Detail

### Month 1 — Literature & Setup
**Goal:** Establish baseline, procure hardware, finalize scope.

| Week | Tasks |
|------|-------|
| W1 | Literature review: PQC (FIPS 203), FL+DP (Abadi 2016, McMahan 2017), HE (TenSEAL), IDS (UNSW-NB15) |
| W2 | Hardware procurement: RPi 4B, RPi Pico, STM32F4 dev board, INA219 power sensor |
| W3 | Set up dev environment: liboqs, TenSEAL, Flower, Opacus, PyTorch |
| W4 | Define all evaluation metrics; draft threat model (`docs/threat_model.md`) |

**Deliverables:**
- [ ] Annotated bibliography (50+ papers)
- [ ] Hardware testbed assembled and connected
- [ ] Development environment verified (`python -m pqc_layer.kyber_wrapper` runs)
- [ ] Threat model document drafted

---

### Month 2 — Classical Baseline Benchmarks
**Goal:** Establish RSA/ECC performance baselines on all hardware.

| Week | Tasks |
|------|-------|
| W1 | Implement RSA-2048/4096 and ECC-P256/P384 benchmarks (`pqc_layer/comparison.py`) |
| W2 | Run baseline benchmarks on RPi 4B: keygen / encrypt / decrypt timing + energy |
| W3 | Run baseline benchmarks on Cortex-M4 (STM32) via ARM Keil / openocd |
| W4 | Profile memory usage (RAM, ROM, stack) on embedded targets |

**Deliverables:**
- [ ] Baseline benchmark results CSV: `benchmarks/results/classical_baseline.csv`
- [ ] Energy measurements via INA219
- [ ] Initial analysis: classical overhead vs. IoMT timing requirements

---

### Month 3 — Kyber Implementation & Optimization
**Goal:** Integrate and optimize Kyber on all target hardware.

| Week | Tasks |
|------|-------|
| W1 | Integrate liboqs Kyber512/768/1024 on RPi 4B; verify correctness |
| W2 | Port pqm4 (Cortex-M4 NTT-optimized) to STM32F446; verify against reference |
| W3 | Run Kyber benchmarks on RPi 4B, RPi Pico, STM32 — all variants |
| W4 | Measure energy per operation; profile stack/ROM usage |

**Deliverables:**
- [ ] Kyber benchmark results: `benchmarks/results/kyber_benchmarks.csv`
- [ ] Side-by-side comparison table (Kyber vs RSA/ECC) — Paper 1 Table 1
- [ ] Port notes for Cortex-M4 integration

---

### Month 4 — Secure Telemetry Protocol
**Goal:** End-to-end PQC secure channel for IoMT telemetry.

| Week | Tasks |
|------|-------|
| W1 | Implement DTLS-style secure channel (`pqc_layer/secure_channel.py`) — handshake |
| W2 | Implement AES-256-GCM data encryption layer; test replay protection |
| W3 | Deploy on RPi ↔ STM32 testbed; measure end-to-end handshake latency |
| W4 | Implement and test hybrid migration gateway (`hybrid_migration/gateway.py`) |

**Deliverables:**
- [ ] Working secure telemetry demo (RPi ↔ sensor) with video recording
- [ ] End-to-end latency benchmark: `benchmarks/results/exp1_crypto_overhead.csv`
- [ ] Migration simulation results
- [ ] Paper 1 draft: sections 1–4 (intro, background, system, setup)

---

### Month 5 — Federated Learning + Differential Privacy
**Goal:** Multi-hospital FL with DP on vitals prediction task.

| Week | Tasks |
|------|-------|
| W1 | Build SyntheticVitalsDataset (MIMIC-III-compatible); verify distribution |
| W2 | Implement FL server + DPFedAvg strategy (`fl_server.py`); single-client test |
| W3 | Implement FL client + Opacus DP-SGD (`fl_client.py`); 5-client local simulation |
| W4 | Run privacy-utility tradeoff experiments: ε ∈ {0.5, 1.0, 3.0, 10.0, ∞} |

**Deliverables:**
- [ ] FL + DP simulation: `benchmarks/results/exp2_fl_dp_tradeoff.json`
- [ ] Privacy-utility curve figure (AUC vs. ε) — Paper 2 Figure 1
- [ ] RDP accountant verified against Opacus internal accounting
- [ ] Apply for MIMIC-III PhysioNet access (requires credentialing)

---

### Month 6 — Homomorphic Encryption Inference
**Goal:** CKKS HE for encrypted vitals inference at edge nodes.

| Week | Tasks |
|------|-------|
| W1 | Implement TenSEAL CKKS context and encryption (`he_inference.py`) |
| W2 | Implement encrypted dot product + polynomial sigmoid inference |
| W3 | Benchmark latency: CKKS-4096 / 8192 / 16384 across 20 patient samples |
| W4 | Run per-hospital privacy budget analysis (`privacy_audit.py`) |

**Deliverables:**
- [ ] HE inference benchmark: `benchmarks/results/exp3_he_inference.json`
- [ ] Latency vs. accuracy tradeoff table — Paper 2 Table 2
- [ ] Paper 2 draft: sections 1–5 (intro through results)
- [ ] Privacy budget analysis report for all 5 hospital types

---

### Month 7 — AI Intrusion & Side-Channel Detection
**Goal:** BiLSTM IDS trained and evaluated on network + power trace data.

| Week | Tasks |
|------|-------|
| W1 | Generate synthetic network attack traffic (`NetworkFlowGenerator`) |
| W2 | Collect real power traces from STM32 during Kyber ops (normal + injected faults) |
| W3 | Train BiLSTM-Attention IDS on network flows; evaluate TPR/FPR/AUC |
| W4 | Train side-channel IDS on power traces; test fault injection detection |

**Deliverables:**
- [ ] Trained IDS models: `intrusion_detection/models/network_ids.pt`, `sidechannel_ids.pt`
- [ ] IDS evaluation results: `benchmarks/results/exp4_ids_detection.json`
- [ ] Real-time detection demo script
- [ ] Power trace dataset from testbed (saved to `benchmarks/results/power_traces/`)

---

### Month 8 — QKD Simulation + Full Integration
**Goal:** BB84 simulation, QKD vs PQC comparison, and complete system integration test.

| Week | Tasks |
|------|-------|
| W1 | BB84 simulator: basic protocol, Eve detection, distance sweep (`bb84_sim.py`) |
| W2 | QKD vs PQC cost analysis (10 hospitals, 5/10/20 node scenarios) |
| W3 | Run all 5 experiments end-to-end: `python experiments/run_all_experiments.py` |
| W4 | Fix bugs, re-run experiments, finalize benchmark CSV/JSON output |

**Deliverables:**
- [ ] BB84 + QKD analysis results: `benchmarks/results/exp5_quantum_attack.json`
- [ ] Full experiment suite passing: `experiment_summary.json`
- [ ] Demo video: secure telemetry + FL prediction + IDS alert (3-minute demo)

---

### Month 9 — Writing, Submission & Portfolio
**Goal:** Submit papers, release open-source toolkit, build portfolio artifacts.

| Week | Tasks |
|------|-------|
| W1 | Complete Paper 1 draft (PQC overhead); submit to IEEE TIFS or IoTDI |
| W2 | Complete Paper 2 draft (FL+DP+HE); submit to IEEE JBHI or NeurIPS workshop |
| W3 | Release GitHub repository; write README, Docker setup, contribution guide |
| W4 | Build grant/internship portfolio: slides deck, 1-page project summary, demo link |

**Deliverables:**
- [ ] Paper 1 submitted to IEEE TIFS / IoTDI
- [ ] Paper 2 submitted to IEEE JBHI / NeurIPS Privacy Workshop
- [ ] GitHub repository released (MIT license)
- [ ] Portfolio: slides + 1-page executive summary + demo video

---

## Gantt Chart (ASCII)

```
Task                              M1  M2  M3  M4  M5  M6  M7  M8  M9
─────────────────────────────────────────────────────────────────────
Literature review                 ████
Hardware setup                    ████
Classical baselines                   ████
Kyber implementation                      ████
Secure telemetry protocol                     ████
Migration gateway                             ████
Federated learning + DP                           ████
Homomorphic encryption                                ████
Privacy audit                                         ████
IDS training (network)                                    ████
IDS training (side-channel)                               ████
BB84 / QKD simulation                                         ████
Full system integration                                       ████
Paper 1 writing                               ──────────────████
Paper 2 writing                                   ──────────████
GitHub release                                                ████
Portfolio / slides                                            ████
```

---

## Key Milestones

| Milestone | Month | Criteria |
|---|---|---|
| M1: Environment ready | 1 | All deps installed, kyber_wrapper.py runs |
| M2: Classical baseline complete | 2 | RSA/ECC results on RPi + STM32 |
| M3: Kyber outperforms RSA on latency | 3–4 | Benchmark shows Kyber ≤ RSA on all metrics |
| M4: Secure channel demo | 4 | RPi ↔ sensor handshake < 15 ms |
| M5: FL + DP running | 5 | 5-hospital simulation, AUC > 0.85 at ε=1.0 |
| M6: HE inference < 10 ms | 6 | CKKS-8192 inference on 5-dim vitals |
| M7: IDS AUC > 0.92 | 7 | BiLSTM evaluated on held-out test |
| M8: All experiments green | 8 | `run_all_experiments.py --quick` passes |
| M9: Papers submitted | 9 | Both papers submitted to target venues |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| liboqs not available on STM32 | Medium | High | Use emulation; pqm4 reference port |
| MIMIC-III access delayed | Medium | Medium | Use SyntheticVitalsDataset for experiments |
| TenSEAL latency too high | Low | Medium | Scope to offline analytics; partial HE |
| Opacus incompatible with model | Low | Low | Use ModuleValidator.fix() |
| Paper rejected on first submission | Medium | Low | Target 2 venues simultaneously |

---

## Budget Estimate

| Item | Cost (USD) |
|---|---|
| Raspberry Pi 4B (×2) | $140 |
| Raspberry Pi Pico (×2) | $16 |
| STM32F446RE Nucleo board | $25 |
| INA219 power sensors (×4) | $20 |
| USB oscilloscope (Hantek 6022BE) | $45 |
| BLE/ZigBee sensors | $30 |
| Misc cables, breadboards | $20 |
| **Total Hardware** | **~$296** |
| Cloud GPU (3 months, Google Colab Pro) | $60 |
| **Total Project** | **~$356** |
