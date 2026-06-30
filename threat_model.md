# SPQR-IoMT Formal Threat Model

**Document:** `docs/threat_model.md`  
**Version:** 1.0  
**Classification:** Research Reference

---

## 1. System Description

The SPQR-IoMT platform secures communication and computation across three tiers:

```
[Tier 1: Medical Sensors]  →  [Tier 2: Hospital Edge Nodes]  →  [Tier 3: Cloud / Central Server]
  Ventilators, pumps,           Raspberry Pi / ARM gateways,       MIMIC data servers,
  vitals monitors, ECG          PQC/FL aggregators, IDS             FL global model store
```

---

## 2. Assets

| Asset | Sensitivity | Location |
|---|---|---|
| Patient vitals (HR, SpO2, BP) | High — PHI | Tier 1 → 2 telemetry |
| ECG/waveform data | High — PHI | Tier 1 sensor |
| Trained FL model weights | Medium | Tier 2 ↔ 3 |
| Cryptographic private keys | Critical | Tier 2 HSM/TEE |
| Device control commands | Critical | Tier 2 → 1 |
| Aggregated population statistics | Medium | Tier 3 |

---

## 3. Adversary Model

### 3.1 Threat Actors

| Actor | Capability | Motivation |
|---|---|---|
| **Nation-state / CRQC adversary** | Cryptographically-relevant quantum computer (10–15 year horizon). Can break RSA-2048, ECDH. Cannot break Kyber or AES-256. | Mass surveillance, data harvest-now-decrypt-later |
| **Network eavesdropper (passive)** | Intercepts all ciphertext on hospital LAN/WAN. Classical compute only. | Data theft, PHI exfiltration |
| **Active MITM attacker** | Can inject, replay, and modify packets on hospital network. | Manipulate device commands, disrupt care |
| **Compromised edge node** | Has physical or software access to a single gateway. Cannot compromise HSM. | Model inversion, key extraction |
| **Malicious hospital (FL)** | Participates in FL as a client, sends poisoned gradients or performs membership inference. | Undermine global model, learn about other hospitals' data |
| **Side-channel attacker** | Physical access to sensor; can measure power/EM traces during crypto ops. | Key recovery via DPA/SPA |
| **Ransomware / insider** | Authenticated access to hospital network, can exfiltrate or encrypt data. | Financial gain, disruption |

### 3.2 Adversary Capabilities (Dolev-Yao + Extensions)

- **Classical Dolev-Yao:** Full control over network; can read, inject, delay, replay all messages.
- **Quantum extension:** Can break RSA/ECDH/ECDSA at sufficient scale. Cannot break AES-256, SHA-3, or Kyber (MLWE/MSIS hardness).
- **Side-channel:** Observes physical leakage (power, timing, EM) from microcontrollers.
- **Gradient leakage:** In FL, observes model updates to reconstruct training data (Zhu et al., 2019).

---

## 4. Attack Taxonomy

### 4.1 Cryptographic Attacks

| Attack | Target | Mitigation |
|---|---|---|
| Harvest-now-decrypt-later (HNDL) | RSA/ECC-encrypted telemetry | Replace with Kyber KEM immediately |
| Shor's algorithm (CRQC) | RSA-2048 key exchange | Kyber512+ (NIST Level 1–5) |
| Grover's algorithm | AES-128, SHA-256 | Use AES-256, SHA-3 |
| Lattice reduction (BKZ) | Kyber (MLWE) | Use Kyber768+ for long-term secrets |
| Key recovery via weak RNG | All schemes | Use OS CSPRNG (getrandom(), /dev/urandom) |
| Padding oracle (RSA-PKCS1) | Legacy RSA | OAEP padding; migrate to PQC |

### 4.2 Network / Protocol Attacks

| Attack | Mechanism | Mitigation |
|---|---|---|
| Replay attack | Resend captured telemetry packets | Sequence numbers + session nonces (SecureChannel) |
| MITM during handshake | Intercept Kyber public key | HMAC-authenticated ServerHello; bind to sensor identity |
| Downgrade attack | Force client to use RSA instead of Kyber | Migration gateway enforces minimum scheme per phase |
| DoS / packet flood | Overwhelm edge gateway CPU | Rate limiting + BiLSTM IDS detection |
| Rogue sensor injection | Connect unauthorized device | Device attestation via pre-shared identity + nonce |

### 4.3 AI / FL Attacks

| Attack | Mechanism | Mitigation |
|---|---|---|
| Model poisoning | Malicious client sends crafted updates | FedAvg averaging (dilutes outliers); Byzantine-robust aggregation |
| Backdoor / Trojan | Inject trigger pattern into global model | Differential testing; anomaly detection on update norms |
| Membership inference | Query model to detect training membership | Differential privacy (DP-SGD, ε ≤ 3) |
| Gradient inversion (DLG) | Recover training images from gradients | DP noise + gradient compression + secure aggregation |
| Data poisoning | Corrupt local training data | Robust preprocessing; outlier removal on vitals data |
| GAN-based reconstruction | Generate fake patient vitals from model | HE inference (server never sees raw features) |

### 4.4 Side-Channel Attacks

| Attack | Target | Mitigation |
|---|---|---|
| Simple Power Analysis (SPA) | Kyber key generation | Constant-time implementation (liboqs) |
| Differential Power Analysis (DPA) | AES operations | Masking; hardware AES-NI |
| Timing attack | Variable-time decapsulation | Constant-time Kyber reference implementation |
| Fault injection | Modify computation mid-operation | BiLSTM side-channel IDS; power trace monitoring |
| Cold boot / memory dump | Key material in RAM | HSM / TEE; memory encryption (TrustZone) |

### 4.5 Physical / Insider Attacks

| Attack | Mechanism | Mitigation |
|---|---|---|
| Device theft | Physical access to sensor/gateway | Full-disk encryption; remote wipe; HSM |
| Firmware tampering | Flash malicious firmware | Secure boot; signed firmware; TPM attestation |
| Insider key theft | Hospital admin copies key material | RBAC; audit logging; HSM non-exportable keys |

---

## 5. Security Properties Claimed

| Property | Mechanism | Formal Guarantee |
|---|---|---|
| **Confidentiality** | Kyber KEM + AES-256-GCM | IND-CCA2 under MLWE |
| **Integrity** | HMAC-SHA3-256 + AES-GCM tag | Existential unforgeability |
| **Authentication** | Sensor identity + session nonce binding | Binding to pre-provisioned identity |
| **Replay protection** | Monotone sequence numbers | Reject any duplicate or out-of-order seq |
| **Forward secrecy** | Ephemeral Kyber key pairs per session | Session key not derivable from long-term keys |
| **Data minimization** | HE inference (encrypted features) | Server learns zero about input features |
| **Privacy (FL)** | DP-SGD + DP-FedAvg (ε, δ) | (ε, δ)-indistinguishability of adjacent datasets |
| **Quantum resistance** | Kyber/NTRU (NIST finalists) | Security under MLWE/NTRU hardness |

---

## 6. Out-of-Scope Threats

The following are acknowledged but outside this platform's scope:

- **Physical coercion of hospital staff** — organizational security control
- **Compromise of NIST PQC standards** — assumed secure; monitor NIST PQC updates
- **Hardware supply-chain attacks** — TPM attestation partially mitigates; full mitigation requires certified hardware procurement
- **Social engineering / phishing** — user security training

---

## 7. Residual Risks

| Risk | Likelihood | Impact | Residual Mitigation |
|---|---|---|---|
| Kyber broken by novel lattice algorithm | Very Low (pre-NIST analysis complete) | Critical | Use hybrid Kyber + X25519 during transition |
| DP noise too low (ε > 3) for small hospitals | Medium | High | Adaptive noise based on dataset size |
| IDS false positive triggers on valid burst traffic | Medium | Medium | Tunable threshold; alert escalation not auto-block |
| Legacy RSA device never upgraded | High | Medium | Migration gateway logs + compliance enforcement |

---

## 8. References

1. NIST SP 800-208: Recommendation for Stateful Hash-Based Signature Schemes
2. Bos et al., "CRYSTALS-Kyber: A CCA-Secure Module-Lattice-Based KEM" (2018)
3. Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
4. Zhu et al., "Deep Leakage from Gradients" (NeurIPS 2019)
5. Kocher et al., "Differential Power Analysis" (CRYPTO 1999)
6. ETSI GS QKD 002: Quantum Key Distribution Use Cases (2010)
