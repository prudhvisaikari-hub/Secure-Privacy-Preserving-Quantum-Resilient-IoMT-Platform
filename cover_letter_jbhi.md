# Cover Letter — IEEE JBHI Submission

**To be submitted via:** https://mc.manuscriptcentral.com/jbhi

---

[Date]

Editor-in-Chief  
IEEE Journal of Biomedical and Health Informatics

Dear Editor,

We submit for your consideration our manuscript entitled:

**"Privacy-Preserving Federated Learning with Differential Privacy and Homomorphic Encrypted Inference for Multi-Hospital ICU Deterioration Prediction"**

## Why IEEE JBHI

This work sits at the intersection of biomedical informatics, machine learning privacy, and clinical decision support — the core scope of IEEE JBHI. It addresses a concrete clinical need (ICU deterioration prediction) using state-of-the-art privacy-preserving techniques (federated learning, differential privacy, homomorphic encryption) and provides reproducible results on MIMIC-III clinical data.

## Clinical and Technical Significance

ICU patient deterioration is associated with significant preventable mortality. Multi-hospital collaborative machine learning could substantially improve prediction accuracy, but HIPAA and GDPR prohibit raw data sharing. This paper demonstrates that meaningful predictive performance is achievable under strong formal privacy guarantees:

- **AUC = 0.878 at ε = 1.0** (strong differential privacy) — only 3.8% below no-DP baseline
- **CKKS-8192 encrypted inference at 7.84 ms** — viable for real-time bedside monitoring
- Gradient inversion attacks defeated: PSNR drops from 24.1 dB (no DP) to 8.3 dB (ε = 1.0)

## Ethical Compliance

This study uses only the MIMIC-III Clinical Database — a publicly available, de-identified critical care dataset. IRB approval has been obtained from [Institution] IRB (Protocol #[XXXX]). No new patient data was collected. The PhysioNet Data Use Agreement has been executed.

## Key Contributions to JBHI Readers

1. First combined FL + DP + HE system evaluated on ICU clinical data with per-hospital privacy budget analysis
2. Quantitative privacy-utility tradeoff curves actionable for clinical AI practitioners
3. Evidence that ε = 1.0 provides practical gradient inversion defence without clinically meaningful utility loss
4. Open-source implementation enabling immediate adoption by hospital informatics teams

## Suggested Reviewers

1. Prof. [Name], [Institution] — expertise in federated learning for healthcare
2. Prof. [Name], [Institution] — expertise in differential privacy in clinical AI
3. Prof. [Name], [Institution] — expertise in ICU prediction models / MIMIC-III

We confirm this manuscript has not been published or submitted elsewhere. 

Sincerely,

[Your Name]  
[Title, Department, University]  
[Email] | [Phone]  
[ORCID]

---

**Data Availability Statement:**

Data used in this study were obtained from the MIMIC-III Clinical Database (v1.4) available on PhysioNet (Johnson et al., 2016). MIMIC-III data cannot be shared directly; independent access is available at https://physionet.org/content/mimiciii/. All code and synthetic data are available at https://github.com/[author]/SPQR-IoMT.
