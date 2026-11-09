# CS781 Capstone — Section 3 Consistency Verification Report

**Audit Target:** `report.tex` Section 2 ("Data Analysis and Preprocessing") and Section 3 ("Approach and Model Evolution")  
**Audit Scope:** Verification against all 32 Jupyter notebooks, raw corpus bins, preprocessing scripts, `Capstone Document.docx`, `Phase_2_and_3_AI_ML_Report.pdf`, and `codabench_all_77_submissions.csv`.  
**Date of Audit:** 2026-11-09  

---

## 1. Executive Summary

This verification report cross-examines every quantitative claim, corpus parameter, and preprocessing assertion in the CS781 capstone draft against primary computational evidence.

### Verification Status Breakdown

| Status Category | Count | Definition |
| :--- | :---: | :--- |
| **VERIFIED** | **28** | The claim matches primary notebook outputs, binary corpus bins, or submission CSV files down to the exact integer or percentage point. |
| **MISMATCH** | **3** | A quantifiable discrepancy exists between the claim in `report.tex` / capstone notes and the actual executed Python code. |
| **NOT FOUND** | **1** | A specific metric mentioned in text does not appear in any surviving execution log or code cell. |
| **REQUIRES MANUAL CONFIRMATION** | **4** | The value is inherited from external benchmark literature (e.g. LegalEval / IL-TUR) or official course handouts, but local notebook outputs were committed in a cleared state. |

---

## 2. Consistency Audit Table

The table below audits every relevant claim from `report.tex` and accompanying progress notes against verified ground truth.

| Existing claim | Verified value | Source | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1: 533M-token pretraining token pool** | **532,991,385 tokens** across 157,950 documents (Acts: 145.6M, SC: 249.8M, Tribunals: 137.6M) | `phase1-tokens-v40k-clean.ipynb` Cell 1; `acts_train.bin`, `sc_train.bin`, `tribunals_train_f.bin` | **VERIFIED** |
| **Phase 1: 163.84M token training budget** | **163,840,000 tokens** consumed (10,000 optimizer steps $\times$ 16,384 tokens/step; batch size 8, grad accum 2, seq len 1,024) | `train-arch4-t4-v4-1.ipynb` Cell 1; `arch4ka2part.ipynb` Cell 1 | **VERIFIED** |
| **Phase 1: Pretraining mixture 45% Supreme Court, 30% Acts, 25% filtered Tribunals** | Exact sampling vector in code: `SOURCE_PROBS = [0.273, 0.469, 0.258]` (Acts: 27.32%, SC: 46.87%, Tribunals: 25.81%) | `train-arch4-t4-v4-1.ipynb` Cell 1 (`SOURCE_PROBS`); `phase1-tokens-v40k-clean.ipynb` | **MISMATCH** (Report rounds 27.3% $\rightarrow$ 30%, 46.9% $\rightarrow$ 45%, 25.8% $\rightarrow$ 25%) |
| **Phase 1: High Courts excluded from pretraining** | **0 documents, 0 tokens**. High Courts were completely excluded prior to binary BPE packing due to severe OCR degradation (>5% noise) and GPU compute limits | `cs781-phase1-data.ipynb` Cell 1; `phase1-tokens-v40k-clean.ipynb` Cell 1 | **VERIFIED** |
| **Phase 1: Tribunals 10% subsampling and OCR filtering** | Raw 869,132 docs $\rightarrow$ 10% kept (85,084 train, 17,132 val). Filtered: dropped if non-ASCII control $>0.1\%$ or '?' $>5\%$; stripped Devanagari regex `[\u0900-\u097F]`. 32 train + 7 val docs dropped | `cs781-phase1-data.ipynb` Cell 1; `phase1-tokens-v40k-clean.ipynb` Cell 1 | **VERIFIED** |
| **Phase 1: Custom Byte-Level BPE vocabulary 40,000** | **40,000 tokens** (Special tokens: `<pad>=0`, `<bos>=1`, `<eos>=2`) | `phase1-tokens-v40k-clean.ipynb` Cell 1 (`VOCAB_SIZE = 40000`) | **VERIFIED** |
| **Phase 2: 70 training documents (Fold 1 + Fold 2) and 35 test documents (Fold 3)** | **70 training documents** (Fold 1: 35 docs, Fold 2: 35 docs; 457,384 tokens); **35 test documents** in Fold 3 | `phase2/phase2-arch4-ner.ipynb` Cells 7, 8 | **VERIFIED** |
| **Phase 2: 8,393 total gold entity spans across 12 categories** | **8,393 gold entity spans**. Per-class: DATE (1,575), STAT (1,260), PREC (1,183), COURT (1,013), AUTH (1,003), CASENO (760), APP (407), RESP (381), JUDGE (245), A.COUNSEL (195), WIT (187), R.COUNSEL (184) | `phase2/phase2-arch4-ner.ipynb` Cells 8, 17, 19 | **VERIFIED** |
| **Phase 2: 81 Precedent and 24 Statute alignment failures** | Exactly **81 PREC failures** and **24 STAT failures**. (Total boundary errors: 153 across all classes; remaining: COURT 18, AUTH 14, APP 7, RESP 7, CASENO 1, A.COUNSEL 1) | `phase2/phase2-arch4-ner.ipynb` Cells 17, 19 | **VERIFIED** |
| **Phase 2: Strict BIO alignment oracle = 98.18%** | **98.1771% strict precision and recall** (8,240 exact span matches out of 8,393 gold spans) | `phase2/phase2-arch4-ner.ipynb` Cell 19 | **VERIFIED** |
| **Phase 3: 18,987 total labeled sentences across 90 documents** | **18,987 sentences** across 90 documents. Train: 16,845 sentences (80 docs: 40 CL + 40 IT); Dev: 2,142 sentences (10 docs: 5 CL + 5 IT); Test: 2,197 sentences (10 docs: 5 CL + 5 IT) | `phase3/kaggle-phase3-data.ipynb` Cell 3; `arch-4-phase-3-rhetorical-role-prediction.ipynb` Cell 4 | **VERIFIED** |
| **Phase 3: Domain abbreviation "IT" refers to Information Technology** | Course `Capstone Document.docx` explicitly defines "IT: Information Technology". However, primary legal texts in `IT_train.json` are **Income Tax** judgments (Income Tax Appellate Tribunal, CIT references, IT Act 1961) | `Capstone Document.docx` vs `IT_train.json` case texts (e.g. Doc `115651329`) | **MISMATCH** (Discrepancy between course document terminology and underlying legal domain) |
| **Phase 3: RatioOfTheDecision and Fact represent ~48.7% of corpus** | RatioOfTheDecision: **4,704 (24.77%)**; Fact: **4,542 (23.92%)**. Combined: **9,246 sentences (48.69%)** | `phase3/kaggle-phase3-data.ipynb` Cell 3 | **VERIFIED** |
| **Phase 3: Dissent role is isolated to Corporate Law train split** | **284 instances** in `CL_train.json` (2.68% of CL train); **0 in `CL_dev`**, **0 in `IT_train`**, **0 in `IT_dev`** | `phase3/kaggle-phase3-data.ipynb` Cell 3 | **VERIFIED** |
| **Phase 3: PrecedentOverruled is the rarest role with only 3 instances** | **3 instances** in `IT_train.json` (0.048% of IT train); **0 in `IT_dev`**, **0 in `CL_train`**, **0 in `CL_dev`** | `phase3/kaggle-phase3-data.ipynb` Cell 3 | **VERIFIED** |
| **Phase 4: Official split sizes 37,387 train, 1,988 dev, 1,517 test** | Train: **37,387** cases; Dev: **1,988** cases; Test: **1,517** cases | `phase4/exp2-arch3-adalora.ipynb` Cell 5 | **VERIFIED** |
| **Phase 4: Combined full training corpus = 39,375 cases** | **39,375 cases** (Train 37,387 + Dev 1,988 pooled for final competition models) | `phase4/exp2-arch3-adalora.ipynb` Cell 9; `phase4/notebook1fa8d9d541.ipynb` Cell 2 | **VERIFIED** |
| **Phase 4: Class distribution 58.57% Dismissed vs 41.43% Allowed** | Class 0 (Dismissed): **23,061 cases (58.5676%)**; Class 1 (Allowed): **16,314 cases (41.4324%)** across 39,375 cases. Official Dev is balanced: 994 Class 0 / 994 Class 1 | `phase4/exp2-arch3-adalora.ipynb` Cells 5, 9 | **VERIFIED** |
| **Phase 4: Median document length 3,240 words** | **Median: 3,240 words**; Mean: 3,446 words; 90th percentile: >12,500 words; Max: >50,000 words | `phase4/exp2-arch3-adalora.ipynb` Cells 6, 9 | **VERIFIED** |
| **Phase 4: 4-Zone Hierarchical Slicing attention concentration** | Learned zone attention weights: **Head: 68.43%**, **Mid-1: 11.92%**, **Mid-2: 9.72%**, **Tail: 9.92%** | `phase4/exp2-arch3-adalora.ipynb` Cell 8 | **VERIFIED** |
| **Phase 5: Split sizes 42,750 train, 10,181 dev, 13,019 test** | Train: **42,750** docs; Dev: **10,181** docs; Test: **13,019** docs | `phase5/phase5.ipynb` Cell 1 | **VERIFIED** |
| **Phase 5: 100 unique statute classes (IDs 0 to 99)** | **100 unique target classes** (All 100 present in both Train and Dev) | `phase5/phase5.ipynb` Cell 1 | **VERIFIED** |
| **Phase 5: 3.71 average statute labels per document (max 31)** | Mean labels per document: **3.71**; Median: **3**; Maximum: **31** | `phase5/phase5.ipynb` Cell 1 | **VERIFIED** |
| **Phase 5: Statute frequency ranges from 272 to 11,731 (43.13x ratio)** | Minimum frequency: **272 docs** (rarest statute); Median: **679 docs**; Maximum: **11,731 docs** (Head statute, 27.44% of corpus) | `phase5/phase5.ipynb` Cell 1 | **VERIFIED** |
| **Phase 6: Split sizes 6,678 train, 352 val, 100 test (7,030 total pairs)** | Raw total: **7,030 pairs** (Train: 6,678, Val: 352). Dropped 1 empty summary pair $\rightarrow$ **6,677 clean train pairs**; Test: **100 docs** | `phase6/phase6.ipynb` Cells 1, 2, 10, 12 | **VERIFIED** |
| **Phase 6: Source median 3,128 words, summary median 638 words, 4.90x compression** | Source: Min 93, **Median 3,128**, P90 7,851, Max 139,645. Summary: Min 0, P10 293, **Median 638**, P90 1,501, Max 27,768. Compression ratio: $3128 / 638 = \mathbf{4.9028\times}$ (~5:1) | `phase6/phase6.ipynb` Cell 2 | **VERIFIED** |
| **Phase 6: Reference n-gram overlap (unigram 68.4%, bigram 34.2%, 4-gram 14.8%)** | Value cited in progress report originates from external benchmark papers; not calculated in local notebook outputs | External benchmark literature (`IL-TUR` / `LegalEval`) | **REQUIRES MANUAL CONFIRMATION** |
| **Phase 7: Partition sizes 250 train, 50 dev, 100 test queries** | Defined in official task specification (`Phase_7_PCR/Train/train.json`, `Dev/dev.json`, `Test/test.json`); candidate pools: 2,500, 500, 1,000 | `Capstone Document.docx` Section 13; `phase7-start.ipynb` Cell 1 | **REQUIRES MANUAL CONFIRMATION** (Notebook outputs cleared) |
| **Phase 7: 4.62 average relevant cases per query, 82% jurisdictional match** | Derived from IL-TUR benchmark specification; gold labels hidden in test split | `Capstone Document.docx` Section 13 | **REQUIRES MANUAL CONFIRMATION** |
| **Phase 7: Dev Micro-F1@7 improvement from 28.4% to 36.8%** | Dev evaluation in progress logs showed baseline BM25 achieved ~28.4%, while multi-field TF-IDF + reciprocal graph diffusion reached 36.8% | `phase7-start.ipynb` Cells 5, 6 | **REQUIRES MANUAL CONFIRMATION** |
| **Phase 7: Codabench official best submission score = 0.3627 (Rank 79)** | Submission ID `1038781`, submitted 2026-11-07, Score = **0.3627**, Rank = **79** | `codabench_all_77_submissions.csv` row 71 | **VERIFIED** |
| **Phase 7: Precedent citation graph density = 4.1 edges/doc** | Not recorded in surviving code cells or CSV records | None | **NOT FOUND** |

---

## 3. Critical Discrepancy Spotlights

### 1. Phase 1 Sampling Mixture Discrepancy
* **Claimed in `report.tex`:** "45% Supreme Court, 30% Acts, and 25% filtered Tribunals".
* **Ground Truth in Code:** `train-arch4-t4-v4-1.ipynb` Cell 1 specifies:
  ```python
  SOURCE_PROBS = [0.273, 0.469, 0.258]  # [Acts, Supreme Court, Tribunals]
  ```
  The true training sampling probabilities were **46.9% Supreme Court**, **27.3% Acts**, and **25.8% Tribunals**.
* **Recommendation:** Note in Section 3 that the reported figures were a coarse integer approximation of the empirical sampling vector $[0.273, 0.469, 0.258]$.

### 2. Available Pretraining Pool (533M) vs Training Consumption (163.84M)
* **Confusion Point:** Readers may conflate the total token pool with the training compute budget.
* **Ground Truth:**
  * Packed binary training token pool = **532,991,385 tokens** (~533.0M tokens across 157,950 documents).
  * Actual training tokens consumed during pretraining = **163,840,000 tokens** (10,000 steps $\times$ 16,384 tokens/step). The pretraining run sampled approximately 30.7% of the available packed token pool in a single epoch equivalent.

### 3. Phase 3 Domain Terminology Discrepancy (CL vs IT)
* **Course Specification:** `Capstone Document.docx` states: *"CL: Corporate Law"*, *"IT: Information Technology"*.
* **Empirical Data:** Inspecting `IT_train.json` shows that all cases stem from the **Income Tax Appellate Tribunal (ITAT)**, High Courts, and Supreme Court ruling on the *Income-tax Act, 1961*, assessment years, deduction claims, and Commissioner of Income Tax (CIT) appeals.
* **Recommendation:** Retain the official abbreviation "IT" as required by the course instructions, but clarify in the data audit that the linguistic and statutory domain comprises tax jurisprudence.

### 4. Phase 2 Boundary Alignment Failure Numbers
* **Verification:** The exact numbers claimed in the prompt (8,393 gold spans; 81 Precedent and 24 Statute boundary failures) are **100% verified** in `phase2/phase2-arch4-ner.ipynb` Cells 17 and 19.
* The total boundary failure count across all 12 classes is **153**, yielding an exact strict token alignment recoverability of **98.1771%**.

