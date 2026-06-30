# Data Management Plan — SPQR-IoMT

**Project:** Secure and Privacy-Preserving Federated Learning for ICU Prediction
**PI:** [Your Name]
**Version:** 1.0

---

## 1. Data Description

| Dataset | Type | Size | Sensitivity | Source |
|---|---|---|---|---|
| MIMIC-III v1.4 | De-identified clinical | ~6 GB | Low (de-identified) | PhysioNet |
| Synthetic vitals | Synthetic | ~50 MB | None | Generated in-house |
| Trained model weights | Derived | ~10 MB | None | Research output |
| Benchmark results | Aggregate statistics | ~1 MB | None | Research output |
| Power trace captures | Hardware measurements | ~100 MB | None | In-house testbed |

---

## 2. Data Collection and Storage

### 2.1 MIMIC-III Access Procedure

1. Complete CITI Program training (Data or Specimens Only Research)
2. Register at physionet.org and apply for credentialed access
3. Execute PhysioNet Data Use Agreement
4. Download only required tables: `CHARTEVENTS`, `ICUSTAYS`, `PATIENTS`
5. Store on **encrypted local drive only** (VeraCrypt AES-256, 256-bit key)

### 2.2 Storage Locations

| Data | Location | Encryption | Backup |
|---|---|---|---|
| Raw MIMIC-III | Local encrypted drive | AES-256 | Encrypted external drive (same room) |
| Processed features | Local encrypted drive | AES-256 | Same |
| Model weights | Local SSD | None (non-sensitive) | Git LFS |
| Benchmark results | Local SSD + GitHub | None | GitHub |
| Synthetic data | GitHub repo | None | GitHub |

### 2.3 Access Control

- Raw MIMIC-III: PI only (single-user encrypted volume)
- Processed features: PI + credentialed co-investigators only
- Model weights / results: All team members, GitHub

---

## 3. Data Processing Pipeline

```
MIMIC-III raw tables
        ↓
  mimic_loader.py (extract CHARTEVENTS for 5 vitals)
        ↓
  Hourly resampling → 24-hour windows
        ↓
  Normalisation (z-score per feature)
        ↓
  SyntheticVitalsDataset-compatible format
        ↓
  FL training (local only, never transmitted)
        ↓
  Aggregate metrics only → published results
```

**At no point is patient data transmitted over a network.**
All FL simulation is local (single-machine simulation of 5 hospitals).

---

## 4. Data Retention and Deletion

| Data | Retention Period | Deletion Method |
|---|---|---|
| Raw MIMIC-III | Duration of project + 1 year | Secure wipe (DoD 5220.22-M) |
| Processed features | Duration of project | Secure wipe |
| Model weights | 5 years (per journal policy) | Standard delete |
| Published results | Indefinite | N/A (non-sensitive) |
| Synthetic data | Indefinite | N/A |

---

## 5. Data Sharing

- **MIMIC-III:** Cannot be shared. Researchers obtain independent access from PhysioNet.
- **Synthetic data:** Freely available in the SPQR-IoMT GitHub repository.
- **Model weights:** Available on request (non-sensitive derived data).
- **Benchmark results:** Fully published in papers and repository.

---

## 6. Compliance

| Regulation | Compliance Measure |
|---|---|
| HIPAA | Using already de-identified data (Safe Harbor); no PHI handled |
| GDPR Art. 9 | UK/EU institutions: de-identified data not subject to Art. 9 |
| PhysioNet DUA | Signed DUA; no re-identification attempts |
| IRB Protocol | Approved protocol on file (see irb_application.md) |
| FAIR Principles | Synthetic data and code fully open; results published |

---

## 7. Security Incident Response

If a security incident occurs (e.g., accidental disclosure of MIMIC-III data):
1. Immediately isolate affected systems
2. Notify IRB within 24 hours
3. Notify PhysioNet within 48 hours
4. Document incident and remediation steps
5. Review and update security procedures

**Emergency contact:** [IRB Office phone number]
