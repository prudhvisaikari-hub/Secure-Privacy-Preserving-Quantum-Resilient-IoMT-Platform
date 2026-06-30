# Response to Reviewers — SPQR-IoMT

**Manuscript:** [Title]  
**Journal:** [IEEE TIFS / IEEE JBHI]  
**Manuscript ID:** [XXXXX]  
**Date of Revision:** [Date]

---

## General Response

We thank the Associate Editor and all reviewers for their thorough and constructive comments. We have addressed all concerns raised and believe the manuscript is substantially improved. Below we provide a point-by-point response. **Reviewer comments are in plain text; our responses are in bold.**

All changes in the revised manuscript are highlighted in blue for ease of review.

---

## Associate Editor Comments

> [AE comment here]

**Response:** [Your response here]

---

## Reviewer 1

### Major Comments

**R1.1:** [Reviewer comment]

**Response:** We thank Reviewer 1 for this observation. [Detailed response explaining what you changed and why]. This is now addressed in Section [X], lines [Y–Z]:

> "[Quoted new/revised text from paper]"

---

**R1.2:** [Reviewer comment about experimental validation]

**Response:** We agree that additional experimental evidence strengthens the claim. We have [added/repeated/extended] the experiments on [platform] with [N] iterations. The new results appear in Table [X]:

| Scheme | Old result | New result | Change |
|---|---|---|---|
| Kyber512 | 12.1 ms | 11.9 ms | −1.7% |

The conclusion remains unchanged: Kyber512 outperforms RSA-2048 by 154× on Cortex-M4.

---

**R1.3:** [Reviewer comment about related work]

**Response:** We have added [N] additional references to the Related Work section, specifically:
- [Author et al., Year]: [What it adds]
- [Author et al., Year]: [What it adds]

The revised Related Work section appears on pages [X–Y].

---

### Minor Comments

**R1.4:** [Minor comment]

**Response:** Corrected. [Brief explanation of change.]

---

## Reviewer 2

### Major Comments

**R2.1:** [Comment about privacy analysis]

**Response:** [Detailed response]

---

**R2.2:** [Comment about clinical relevance]

**Response:** We thank the reviewer for raising this important clinical context. [Response]. We have added the following to the Discussion section:

> "[New text added to paper]"

---

### Minor Comments

**R2.3:** Typo in Section 3, line 12: "encapuslation" should be "encapsulation".

**Response:** Corrected, thank you.

---

## Reviewer 3

### Major Comments

**R3.1:** [Comment about comparison with prior work]

**Response:** [Detailed comparison added to paper. Table comparing with prior work:]

| Prior Work | Platform | Kyber512 keygen | Notes |
|---|---|---|---|
| pqm4 [Kannwischer 2019] | Cortex-M4 @ 168 MHz | 13.1 ms | Different clock, no IoMT protocol |
| This work | STM32F446 @ 180 MHz | 12.1 ms | Full protocol + energy measurement |

Our results are consistent with pqm4 after normalising for clock frequency.

---

## Summary of Changes

| Section | Change | Motivation |
|---|---|---|
| Abstract | Added confidence intervals to key numbers | R1.1 |
| Sec. 2 | Added 5 references to related work | R1.3 |
| Sec. 4 | Extended benchmark to 500 iterations (Kyber), 50 (RSA) | R1.2 |
| Sec. 5 | Added Table 6: comparison with pqm4 | R3.1 |
| Sec. 6 | Expanded clinical context paragraph | R2.2 |
| App. A | Added raw benchmark data as supplementary | R2.1 |

---

## Checklist Before Resubmission

- [ ] All reviewer responses written
- [ ] All changes highlighted in blue in revised manuscript
- [ ] Supplementary material updated
- [ ] Line numbers verified for all quoted passages
- [ ] Word count within journal limits
- [ ] New figures are vector PDF
- [ ] Response document is ≤10 pages
- [ ] Cover letter for resubmission written
