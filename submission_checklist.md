# Paper Submission Checklist — SPQR-IoMT

## Paper 1 — IEEE TIFS (PQC Overhead)
**Target:** IEEE Transactions on Information Forensics and Security  
**Portal:** https://mc.manuscriptcentral.com/tifs  
**Format:** IEEE double-column, 12–14 pages max

### Pre-submission Checklist

#### Experiments (Must be REAL hardware results)
- [ ] Run Exp1 on RPi 4B (200 iterations each scheme) → `benchmarks/results/exp1_crypto_overhead.csv`
- [ ] Run Exp1 on STM32F446 (pqm4, 100 iterations) → `benchmarks/results/exp1_crypto_m4.csv`
- [ ] Record INA219 energy measurements for all schemes → `benchmarks/results/exp1_energy.csv`
- [ ] Collect power traces (1000 traces each: normal Kyber, normal RSA) → `real_results/power_traces/`
- [ ] Run handshake latency benchmark (100 iterations) → `benchmarks/results/bench_secure_channel.csv`
- [ ] Run migration fleet simulation → `benchmarks/results/exp5_migration_plan.json`

#### Figures (All must be generated from REAL results)
- [ ] Fig 1: Keygen latency comparison (RPi4 + M4) — log scale bar chart
- [ ] Fig 2: Key and ciphertext sizes — grouped bar chart
- [ ] Fig 3: Energy per operation — log scale bar chart
- [ ] Fig 4: End-to-end handshake latency breakdown
- [ ] Fig 5: Power trace comparison (Kyber vs RSA — from oscilloscope)
- [ ] Fig 6: RAM/ROM footprint comparison
- [ ] Fig 7: Migration fleet PQC adoption over phases

#### Paper Content
- [ ] Abstract: 250 words, includes key numbers (157×, 0.79 µJ, 8.31 ms)
- [ ] Introduction: HNDL threat clearly motivated, gap identified, 5 contributions listed
- [ ] Background: Kyber math (MLWE), IoMT constraints, related work (10+ citations)
- [ ] System Design: Protocol diagram, security proof sketch, migration phases
- [ ] Experiments: Hardware table, measurement methodology, metric definitions
- [ ] Results: All tables filled with REAL numbers, statistical confidence intervals
- [ ] Discussion: IoMT recommendations, limitations, future work
- [ ] Conclusion: Summary, impact statement
- [ ] References: 15–25 references, all IEEE formatted

#### LaTeX Formatting
- [ ] Compile with `pdflatex` — zero warnings
- [ ] All figures as PDF (vector) for print quality
- [ ] All tables use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`)
- [ ] Equations numbered where referenced
- [ ] Algorithms in `algorithm` environment
- [ ] No orphaned lines or overfull hboxes

#### IEEE Compliance
- [ ] IEEEtran class, `[10pt,journal,compsoc]`
- [ ] Author information removed for blind review
- [ ] Keywords: 4–6 IEEE keywords
- [ ] No acknowledgements in review version
- [ ] Supplementary material uploaded separately (code, data)

#### Ethics and Reproducibility
- [ ] No patient data in paper or supplementary
- [ ] Code repository link included (anonymous for review)
- [ ] All random seeds specified
- [ ] Hardware platforms fully described (clock speed, RAM, OS version)
- [ ] `reproduce_all.sh` tested end-to-end

---

## Paper 2 — IEEE JBHI (FL + DP + HE)
**Target:** IEEE Journal of Biomedical and Health Informatics  
**Portal:** https://mc.manuscriptcentral.com/jbhi  
**Format:** IEEE double-column, 10–12 pages

### Pre-submission Checklist

#### Data and Ethics
- [ ] CITI training certificate obtained and uploaded to PhysioNet
- [ ] PhysioNet DUA signed and on file
- [ ] IRB approval obtained (see `docs/irb/irb_application.md`)
- [ ] IRB protocol number included in paper

#### Experiments
- [ ] MIMIC-III downloaded and processed → `data/mimic_processed/`
- [ ] Exp2 FL+DP sweep (ε ∈ {0.5,1.0,3.0,10.0,∞}) → `benchmarks/results/exp2_fl_dp_tradeoff.json`
- [ ] Exp3 HE inference benchmark (CKKS-4096/8192/16384) → `benchmarks/results/exp3_he_inference.json`
- [ ] Per-hospital privacy budget analysis → `benchmarks/results/exp2_privacy_budget.json`
- [ ] DLG gradient inversion attack evaluation on FL updates

#### Figures
- [ ] Fig 1: Privacy-utility curve (AUC vs ε) with error bars
- [ ] Fig 2: Per-hospital ε bar chart (5 hospital types)
- [ ] Fig 3: FL convergence curves (AUC over 50 rounds, multiple ε)
- [ ] Fig 4: HE inference latency vs CKKS config (bar + error bars)
- [ ] Fig 5: CKKS approximation error (line plot over 20 samples)
- [ ] Fig 6: DLG reconstruction quality (PSNR) with/without DP

#### Paper Content
- [ ] IRB protocol number in manuscript
- [ ] MIMIC-III citation and PhysioNet acknowledgement
- [ ] Positive rate of deterioration events reported
- [ ] All ε values reported with δ = 1e-5
- [ ] Opacus version number reported
- [ ] RDP accountant parameters documented
- [ ] HE polynomial approximation error bounded

#### Ethics Statement
The following must appear verbatim in the paper:

> *"This study used data from the MIMIC-III Clinical Database (v1.4), a publicly available de-identified critical care dataset available through PhysioNet under a data use agreement. IRB approval was obtained from [Institution] IRB (Protocol #[XXXX]). No new patient data was collected."*

---

## Cover Letters

### IEEE TIFS Cover Letter Template
See `docs/cover_letter_tifs.md`

### IEEE JBHI Cover Letter Template
See `docs/cover_letter_jbhi.md`

---

## After Submission

- [ ] Note submission date and manuscript number
- [ ] Set calendar reminder for 90-day follow-up (if no decision)
- [ ] Prepare reviewer response template (`docs/response_to_reviewers_template.md`)
- [ ] Archive final submitted version in `paper1_latex/submitted/` and `paper2_latex/submitted/`
