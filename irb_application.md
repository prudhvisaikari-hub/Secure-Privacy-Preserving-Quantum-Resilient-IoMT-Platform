# IRB Application — SPQR-IoMT Research Project

**Application Type:** Expedited Review (Category 4 — Secondary Research Using Existing Data)
**Protocol Title:** Secure and Privacy-Preserving Federated Learning for ICU Patient Deterioration Prediction Using De-identified Clinical Data
**Principal Investigator:** [Your Name], [Department], [University]
**Co-Investigators:** [List co-investigators if any]
**Funding Source:** [Grant number / self-funded]
**Submission Date:** [Date]

---

## 1. Project Summary

This research develops and evaluates a privacy-preserving federated learning system (SPQR-FL) for predicting ICU patient deterioration across multiple hospital sites. We will use the MIMIC-III Clinical Database — a publicly available, de-identified critical care database — to train and evaluate machine learning models for vitals-based deterioration prediction.

**No new patient data will be collected.** All data used is existing, de-identified, and publicly accessible under a data use agreement with PhysioNet.

---

## 2. Purpose and Background

### 2.1 Scientific Background

ICU patient deterioration events (cardiac arrest, respiratory failure, septic shock) account for significant preventable mortality. Early warning systems using vital signs time-series (heart rate, SpO₂, respiratory rate, blood pressure, temperature) have shown promise but require large, diverse training datasets that no single hospital possesses.

Federated learning enables multi-hospital collaboration without sharing raw patient data. However, gradient inversion attacks can still reconstruct sensitive records from model updates. Differential privacy (DP) provides formal privacy guarantees but trades off model utility. This project quantifies that tradeoff for ICU applications.

### 2.2 Research Objectives

1. Evaluate the privacy-utility tradeoff of DP-SGD at ε ∈ {0.5, 1.0, 3.0, 10.0} for ICU deterioration prediction
2. Demonstrate CKKS homomorphic encryption for privacy-preserving inference at <10ms latency
3. Characterise per-hospital privacy budgets under heterogeneous dataset sizes
4. Release open-source tools for privacy-preserving medical AI

---

## 3. Subject Population

**This study involves NO human subjects.** We use only:

- **MIMIC-III Clinical Database** (v1.4): A de-identified database of 53,423 distinct hospital admissions to Beth Israel Deaconess Medical Center ICUs between 2001–2012. All protected health information (PHI) has been removed per HIPAA Safe Harbor method prior to public release by PhysioNet/MIT.

Data access requires:
- Completion of CITI Program Data or Specimens Only Research training
- Execution of PhysioNet Data Use Agreement
- IRB approval at home institution (this document)

**No new data collection, no patient contact, no identifiable information.**

---

## 4. Data Sources and Access

### 4.1 MIMIC-III

| Attribute | Detail |
|---|---|
| Database | MIMIC-III Clinical Database v1.4 |
| Source | PhysioNet (physionet.org/content/mimiciii/) |
| Records | 53,423 ICU admissions, 38,597 distinct patients |
| De-identification | HIPAA Safe Harbor — all PHI removed |
| Access | Credentialed (free, requires training + DUA) |
| Storage | Local encrypted drive, never transmitted |

### 4.2 Variables Used

We extract only the following non-identifying physiological measurements:
- Heart rate (bpm)
- Peripheral oxygen saturation SpO₂ (%)
- Respiratory rate (breaths/min)
- Systolic blood pressure (mmHg)
- Temperature (°C)
- Outcome label: in-hospital deterioration event within 6 hours (binary)

**No demographic data, no dates, no location, no free text is used.**

---

## 5. Risks and Benefits

### 5.1 Risks

**Risk to subjects: NONE.** There are no human subjects in this study. The data is already de-identified and publicly available.

**Risk of re-identification:** Extremely low. We use only 5 physiological measurements in aggregated form. No attempt is made to identify any individual.

### 5.2 Benefits

**Direct benefits to society:**
- Open-source privacy-preserving AI tools for hospitals
- Formal quantification of privacy-utility tradeoffs for clinical AI
- Advancement of quantum-resilient IoMT security standards

---

## 6. Privacy and Confidentiality

- MIMIC-III data stored on encrypted local drive (AES-256) accessible only to PI and co-investigators
- No patient data is uploaded to cloud services
- All intermediate model weights and gradients are deleted after experiments
- Published results contain only aggregated statistics (AUC, F1, ε values) — no individual records
- Code repository contains only synthetic data; real MIMIC-III data never committed to version control

---

## 7. Informed Consent

**Waiver of informed consent requested** under 45 CFR 46.116(f):
- Research involves no more than minimal risk
- Research could not practicably be carried out without the waiver
- Subjects' rights and welfare will not be adversely affected
- Subjects will not be asked to perform additional acts
- Data is already de-identified per HIPAA Safe Harbor

---

## 8. CITI Training Certification

The PI and all co-investigators have completed the required CITI Program modules:
- [ ] Data or Specimens Only Research (required for MIMIC-III access)
- [ ] Biomedical Research (required by home institution)
- [ ] Responsible Conduct of Research

*(Attach certificates as Appendix A)*

---

## 9. PhysioNet Data Use Agreement

The PhysioNet DUA requires:
- Not to attempt to identify any patient
- Not to share data with unauthorised individuals
- To report any accidental disclosure immediately
- To acknowledge MIMIC-III in all publications

*(Signed DUA to be attached as Appendix B)*

---

## 10. Publication and Dissemination

Results will be published in peer-reviewed venues (IEEE TIFS, IEEE JBHI) and as open-source code. Only aggregate model performance metrics will be published. No patient-level data will appear in any publication.

---

## 11. Principal Investigator Statement

I certify that the information provided in this application is accurate and complete. I will conduct this research in accordance with the approved protocol, IRB regulations, and all applicable policies.

**PI Signature:** _______________________ **Date:** _____________

**Department Chair Signature:** _______________________ **Date:** _____________

---

## Appendix A: Required CITI Training Certificates
*(Attach PDF certificates here)*

## Appendix B: PhysioNet Data Use Agreement
*(Attach signed DUA here)*

## Appendix C: Data Security Plan
*(See data_management_plan.md)*
