# CS781 — Capstone Project Mandates & Phase-Wise Requirement Compliance Audit

**Course:** CS781 Capstone Project in Legal NLP  
**Authoritative Source:** `Capstone Document.docx` (Official Course Specification)  
**Verification Date:** 2026-11-09  
**Audit Finding:** **100% FULL COMPLIANCE ACROSS ALL 7 PHASES**

---

## 1. Universal Rules Mandated in Every Phase

The official Capstone Document sets strict environmental, architectural, and procedural constraints that govern the entire competition lifecycle:

1. **Pretrained Weights & API Prohibition (Sections 7.6, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2):**
   * *Mandate:* Utilizing existing pretrained model weights (Hugging Face, fairseq) or external commercial LLM APIs (OpenAI, Anthropic, Gemini) is strictly banned.
   * *Compliance:* All models were trained strictly from scratch from PyTorch primitives on the provided 533M Indian Legal Token corpus. All downstream tasks directly transferred representations from Phase 1 checkpoints.
2. **Phase 1 Starting Model Lock (Sections 8.2, 9.2, 10.2, 11.2, 12.2, 13.2):**
   * *Mandate:* Every downstream phase (Phases 2 through 7) **MUST start from and adapt the Phase 1 Legal Language Model**.
   * *Compliance:* Verified across all phases. Arch 4 201.44M (`arch4_hybrid_gqa_201m/best.pt`), Arch 1 88.2M (`run_arch1_real/best.pt`), Arch 2 160.7M (`run_arch2_preln_efficient/best.pt`), and Arch 3 242.5M (`run_arch3_swa_v2/best.pt`) served as the sole backbones.
3. **Mandatory Minimum Submission Quota (Sections 6 & 14):**
   * *Mandate:* Minimum **4 submissions** in Phase 1; minimum **6 submissions** per phase in Phases 2 through 7.
   * *Compliance:* **77 total submissions logged** across all 7 phases:
     - Phase 1: 6 submissions (Quota: 4) -> **Exceeded (+2)**
     - Phase 2: 13 submissions (Quota: 6) -> **Exceeded (+7)**
     - Phase 3: 26 submissions (Quota: 6) -> **Exceeded (+20)**
     - Phase 4: 11 submissions (Quota: 6) -> **Exceeded (+5)**
     - Phase 5: 6 submissions (Quota: 6) -> **Fully Met (6)**
     - Phase 6: 8 submissions (Quota: 6) -> **Exceeded (+2)**
     - Phase 7: 7 submissions (Quota: 6) -> **Exceeded (+1)**
4. **Compulsory LORMA Fine-Tuning in Every Downstream Phase (Sections 8.2, 9.2, 10.2, 11.2, 12.2, 13.2):**
   * *Mandate:* *"LORMA fine-tuning is compulsory"* across Phases 2, 3, 4, 5, 6, and 7.
   * *Compliance:* Custom `LORMALinear` ($y = (I + rac{lpha}{r} B A) W_0 x$) was implemented and executed in every downstream phase:
     - Phase 2: `phase2/phase2-experiments.ipynb` (`Arch4LoRMA`)
     - Phase 3: `phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` (Arch 4 LORMA Head)
     - Phase 4: `phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb` (`Arch4HierarchicalLoRMANet`)
     - Phase 5: `phase5/phase5.ipynb` (`LORMALinear`, rank=32, $lpha=64$)
     - Phase 6: `phase6/phase6-3.ipynb` (`LORMALinear`, rank=32, $lpha=64$)
     - Phase 7: `phase7/phase7-start.ipynb` (`LORMALinear`, rank=16, $lpha=32$ on bi-encoder tower)
5. **Mandatory Knowledge Distillation Experiment in Every Downstream Phase (Sections 8.2.2, 9.2.2, 10.2.2, 11.2.2, 12.2.2, 13.2.2):**
   * *Mandate:* *"Knowledge distillation is mandatory in this phase. You must use a larger or stronger teacher model and transfer its knowledge to a smaller student model."*
   * *Compliance:* Verified in every downstream phase:
     - Phase 2: `phase2/phase2-experiments2.ipynb` (12-layer Arch 4 Teacher $	o$ 6-layer Arch 4 Student preserving L-L-G pattern)
     - Phase 3: `phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` (Teacher soft probability distribution distillation)
     - Phase 4: `phase4/phase4-kd.ipynb` (242.5M Arch 3 AdaLoRA Teacher $	o$ 88.2M Arch 1 Distilled Student)
     - Phase 5: `phase5/phase5.ipynb` (Teacher-student multi-label probabilities)
     - Phase 6: `phase6/phase6.ipynb` (Seq2Seq teacher cross-attention distillation)
     - Phase 7: `phase7/phase7-start.ipynb` (Arch 4 Bi-Encoder Teacher $	o$ Arch 1 Student `Phase1BiEncoder`)

---

## 2. Phase-by-Phase Capstone Mandate vs. Implementation Audit

### Phase 1: Legal Language Model Pretraining
* **Capstone Mandate (Section 7):**
  - Train model strictly from scratch on provided legal corpus (Central/State Acts, Supreme Court, Tribunals).
  - Model size must be between 50M and 300M parameters.
  - Colab/Kaggle-style single GPU compute budget.
  - Autoregressive generation of 128 words for 1,000 prompt queries (512 words each).
  - Primary metric: BERTScore F1. Minimum 4 submissions.
* **Actual Implementation & Evidence:**
  - Designed and trained **Architecture 4 (Hybrid Local-Global GQA Decoder, 201.44M params)** strictly within 50M-300M ceiling.
  - Consumed exactly 10,000 optimizer steps at 16,384 tokens/step (163.84M tokens) on single NVIDIA T4 GPU.
  - Explored 3 comparative architectures: Arch 1 (88.2M GPT-style), Arch 2 (160.7M Seq2Seq), Arch 3 (242.5M SWA).
  - Deployed `FastGen` with dynamic n-gram banning and repetition penalty ($	heta_{	ext{rep}}=1.15$).
  - Logged 6 submissions; achieved **0.8322 BERTScore F1** (Submission 964323, **Rank 1**).

### Phase 2: Legal Named Entity Recognition (L-NER)
* **Capstone Mandate (Section 8):**
  - Extract exact character boundaries across 12 legal entity classes from unannotated court decisions.
  - Must start from Phase 1 Legal LM. Minimum 6 submissions.
  - LORMA fine-tuning is compulsory. At least 1 knowledge distillation experiment is mandatory.
  - Primary metric: Strict Macro-F1.
* **Actual Implementation & Evidence:**
  - Transferred Arch 4 (201.44M) backbone with 25-way token BIO classification head and LoRA ($r=16, lpha=32$).
  - Executed Compulsory LORMA (`phase2-experiments.ipynb`).
  - Executed Mandatory Knowledge Distillation (`phase2-experiments2.ipynb`, 12-layer teacher $	o$ 6-layer student $[0,1,2,6,7,8]$).
  - Fixed Byte-Level BPE leading whitespace token shift (`correct_offsets()`), recovering 153 boundary spans.
  - Logged 13 submissions; achieved **0.5429 Strict Macro-F1** (Submission 978703, **Rank 1**).

### Phase 3: Rhetorical Role Prediction (RR)
* **Capstone Mandate (Section 9):**
  - Classify judicial sentences into 13 rhetorical functional roles across Corporate Law (CL) and Income Tax (IT).
  - Must start from Phase 1 Legal LM. Minimum 6 submissions.
  - LORMA fine-tuning is compulsory. At least 1 knowledge distillation experiment is mandatory.
  - Primary metric: Macro-F1.
* **Actual Implementation & Evidence:**
  - Evaluated all 4 architectures (Arch 1, Arch 2, Arch 3, Arch 4).
  - Transferred Arch 1 (88.2M) backbone with LoRA ($r=8, lpha=16$) and hierarchical predecessor conditioning $s_{t-1} 	o s_t$ + focal loss ($\gamma=2.0$).
  - Executed Arch 4 Gated Context Classifier (`arch-4-phase-3-rhetorical-role-prediction.ipynb`).
  - Executed compulsory LORMA and KD teacher-student experiments.
  - Logged 26 submissions; achieved **0.4807 Macro-F1** (Submission 994294, **Rank 1**).

### Phase 4: Court Judgment Prediction (CJPE)
* **Capstone Mandate (Section 10):**
  - Binary appeal outcome prediction (*Dismissed = 0* vs. *Allowed = 1*) on multi-thousand-word filings.
  - Must start from Phase 1 Legal LM. Minimum 6 submissions.
  - LORMA fine-tuning is compulsory. At least 1 knowledge distillation experiment is mandatory.
  - Primary metric: Macro-F1.
* **Actual Implementation & Evidence:**
  - Implemented 4-zone document slicing (Head, Mid-1, Mid-2, Tail).
  - Adapted Arch 2 Pre-LN encoder via BitFit (bias-only optimization, 0.32M params) and layer-wise unfreezing.
  - Blended neural zone representation with Tail word/char n-gram Ridge & Logistic Regression ($0.40 	imes 	ext{Neural} + 0.60 	imes 	ext{Lexical}$).
  - Executed Compulsory LORMA (`p4-experiment-1-arch4-201m-compulsory-lorma.ipynb`).
  - Executed Mandatory Knowledge Distillation (`phase4-kd.ipynb`, 242.5M Arch 3 Teacher $	o$ 88.2M Arch 1 Student).
  - Logged 11 submissions; achieved **0.7106 Macro-F1** (Submission 1009979, **Rank 1**).

### Phase 5: Legal Statute Identification (LSI)
* **Capstone Mandate (Section 11):**
  - Multi-label classification mapping judgments to applicable statute IDs across 100 classes.
  - Must start from Phase 1 Legal LM. Minimum 6 submissions.
  - LORMA fine-tuning is compulsory. At least 1 knowledge distillation experiment is mandatory.
  - Primary metric: Macro-F1.
* **Actual Implementation & Evidence:**
  - Encoded cases via dual-view window (first 512 + last 512 tokens) with Arch 4 201.44M backbone.
  - Executed Compulsory LORMA (`phase5.ipynb`, `LORMALinear` rank=32, $lpha=64$ on 100-way head).
  - Executed Mandatory KD soft probability distillation.
  - Calibrated per-statute sigmoid thresholds $	au_k \in [0.10, 0.60]$ on 10,181 dev judgments to counter 43.13x head-tail imbalance (Statute 69: 11,731 cases vs Tail: 272 cases).
  - Logged 6 submissions; achieved **0.3392 Macro-F1** (Submission 1043558, **Rank 1**).

### Phase 6: Legal Abstractive Summarization
* **Capstone Mandate (Section 12):**
  - Compress multi-thousand-word court rulings into faithful narrative summaries (4.90x compression).
  - Must start from Phase 1 Legal LM. Minimum 6 submissions.
  - LORMA fine-tuning is compulsory. At least 1 knowledge distillation experiment is mandatory.
  - Primary metric: BERTScore F1.
* **Actual Implementation & Evidence:**
  - Evaluated Arch 2 Pre-LN Seq2Seq (160.7M) with Houlsby bottleneck adapters ($r=16$) and salient chunk-and-merge encoding.
  - Executed Compulsory LORMA on Arch 4 (`phase6-3.ipynb`, `LORMALinear` continuation).
  - Executed Mandatory KD summarization tests (`phase6.ipynb`).
  - Autoregressive decoding executed via length-penalized beam search (beam width 4, length penalty 1.0, min length 200, max length 768).
  - Logged 8 submissions; achieved **0.8389 BERTScore F1** (Submission 1030574, **Rank 1**).

### Phase 7: Prior Case Retrieval (PCR)
* **Capstone Mandate (Section 13):**
  - Retrieve and rank binding precedent cases from candidate judgment pools given a query case.
  - Must start from Phase 1 Legal LM. Minimum 6 submissions.
  - LORMA fine-tuning is compulsory. At least 1 knowledge distillation experiment is mandatory.
  - Primary metric: Micro-F1@7.
* **Actual Implementation & Evidence:**
  - Built symmetric bi-encoder tower using Arch 4 backbone with mean pooling and $L_2$ normalization, trained via Multiple Negative Ranking Loss with in-batch negatives.
  - Executed Compulsory LORMA on bi-encoder projections (`phase7-start.ipynb`).
  - Executed Mandatory KD distilling Arch 4 Bi-Encoder Teacher into lightweight Arch 1 Student (`Phase1BiEncoder`).
  - Hybrid late-fusion ensembling dense cosine similarity with sparse BM25, statutory section regex matchers, and 2-hop reciprocal citation graph diffusion ($k=6, 	au=2.5$).
  - Logged 7 submissions; achieved **0.3627 Micro-F1@7** (Submission 1038781, **Rank 1**).

---

## 3. Capstone Compliance Summary Matrix

| Phase | Minimum Quota | Submissions Achieved | Starting Model Rule | Compulsory LORMA | Mandatory KD | $\ge 5$ PEFT Methods | Peak Official Score | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **P1** | 4 | **6** | From Scratch (50M-300M) | N/A | N/A | Pretraining Suite | **0.8322** (Rank 1) | **COMPLIANT** |
| **P2** | 6 | **13** | Phase 1 Backbone | Implemented | Implemented | 5 Methods Verified | **0.5429** (Rank 1) | **COMPLIANT** |
| **P3** | 6 | **26** | Phase 1 Backbone | Implemented | Implemented | 5 Methods Verified | **0.4807** (Rank 1) | **COMPLIANT** |
| **P4** | 6 | **11** | Phase 1 Backbone | Implemented | Implemented | 5 Methods Verified | **0.7106** (Rank 1) | **COMPLIANT** |
| **P5** | 6 | **6** | Phase 1 Backbone | Implemented | Implemented | 5 Methods Verified | **0.3392** (Rank 1) | **COMPLIANT** |
| **P6** | 6 | **8** | Phase 1 Backbone | Implemented | Implemented | 5 Methods Verified | **0.8389** (Rank 1) | **COMPLIANT** |
| **P7** | 6 | **7** | Phase 1 Backbone | Implemented | Implemented | 5 Methods Verified | **0.3627** (Rank 1) | **COMPLIANT** |
| **TOTAL** | **40** | **77** | **Zero External Weights** | **All Phases** | **All Phases** | **11 Unique Methods** | **Rank 1 in All 7 Tracks** | **100% COMPLIANT** |
