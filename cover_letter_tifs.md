# Cover Letter — IEEE TIFS Submission

**To be submitted via:** https://mc.manuscriptcentral.com/tifs

---

[Date]

Editor-in-Chief  
IEEE Transactions on Information Forensics and Security

Dear Editor,

We submit for your consideration our manuscript entitled:

**"Lightweight Post-Quantum Cryptography for Resource-Constrained Internet of Medical Things Devices: A Comparative Overhead Analysis of CRYSTALS-Kyber Against RSA and ECC"**

## Why IEEE TIFS

This manuscript is ideally suited for IEEE TIFS because it addresses a critical intersection of cryptographic security and system performance that is central to the journal's scope: the practical deployment of post-quantum cryptographic standards on constrained systems, with formal security analysis and empirical evaluation.

## Significance

The "harvest-now, decrypt-later" (HNDL) threat to medical data is present and active today — adversaries are collecting encrypted patient records now for decryption once a quantum computer becomes available. NIST standardised CRYSTALS-Kyber (FIPS 203) in August 2024, but no prior work has systematically characterised its feasibility for Internet of Medical Things (IoMT) devices under realistic medical telemetry workloads.

## Key Contributions

Our paper provides:
1. The **first comprehensive Kyber vs. RSA/ECC benchmark** on ARM Cortex-M4 (STM32F446) and Raspberry Pi 4B under IoMT telemetry workloads, measuring latency, energy (via INA219 current sensing), and memory
2. A **DTLS-inspired secure telemetry protocol** achieving <9 ms end-to-end handshake latency with Kyber512 — 16.6× faster than RSA-2048
3. **Power trace side-channel analysis** demonstrating Kyber's constant-time advantage over RSA
4. A **phased hybrid migration framework** enabling practical transition for heterogeneous hospital fleets
5. An open-source toolkit (SPQR-IoMT) enabling immediate adoption

## Key Result

Kyber512 is **157× faster** than RSA-2048 at key generation on Cortex-M4, consumes **271× less energy** (0.79 µJ vs. 214 µJ), and requires **6.8× less RAM** (6.2 KB vs. 42.0 KB), conclusively demonstrating that quantum-resistant cryptography is deployable on today's medical IoT hardware.

## Originality

This manuscript has not been published, submitted, or is under review elsewhere. A two-page abstract was presented at [Workshop Name, Year] (cite if applicable), but this full paper represents substantially expanded work including the Cortex-M4 benchmark, energy characterisation, side-channel analysis, and migration framework.

## Suggested Reviewers

1. Prof. [Reviewer 1 Name], [Institution] — expertise in embedded cryptography ([email])
2. Prof. [Reviewer 2 Name], [Institution] — expertise in IoT security ([email])
3. Prof. [Reviewer 3 Name], [Institution] — expertise in post-quantum cryptography ([email])

## Excluded Reviewers

[List anyone with conflict of interest]

We look forward to your consideration.

Sincerely,

[Your Name]  
[Title, Department, University]  
[Email] | [Phone]  
[ORCID]
