# CS781 – Phase-Wise Architecture Inventory

## Executive Architecture Summary

Across the 7 phases of CS781, four distinct foundational neural model architectures were implemented from PyTorch primitives, pretrained from scratch on the 533M Indian Legal Token corpus, and systematically adapted for downstream tasks using parameter-efficient fine-tuning (PEFT) and knowledge distillation (KD).

| Architecture | Model Family | Primary Dimensions | Parameters | Key Innovations | Foundational Checkpoint |
|---|---|---|---|---|---|
| **Arch 1** | Standard Causal Decoder | 8 layers, $d=768$, 8 heads, $d_{\text{ff}}=3072$ | 88,206,336 (88.2M) | Standard causal GPT-style decoder, learned positional embeddings, Pre-LN | `run_arch1_real/best.pt` |
| **Arch 2** | Pre-LN Encoder-Decoder | 6 enc + 6 dec, $d=768$, 8 heads, $d_{\text{ff}}=3072$ | 160,680,960 (160.7M) | Pre-LN seq2seq transformer, cross-attention in decoder | `run_arch2_preln_efficient/best.pt` |
| **Arch 3** | Sliding-Window Decoder | 16 layers, $d=1024$, 16 heads, $d_{\text{ff}}=4096$ | 242,536,960 (242.5M) | Sliding-window causal attention ($W=256/512$), $\mathcal{O}(N \cdot W)$ memory | `run_arch3_swa_v2/best.pt` |
| **Arch 4** | Hybrid Local-Global GQA | 12 layers, $d=1024$, 16q/8kv heads, $d_{\text{ff}}=3328$ | 201,443,328 (201.4M) | Interleaved local/global attention (L-L-G), RoPE, SwiGLU, GQA (2:1 KV compression) | `arch4_hybrid_gqa_201m/best.pt` |

---

## Phase-by-Phase Architecture Breakdown

### Phase 1: Legal Language Model Pretraining
* **Core Task:** Autoregressive domain continuation of 512-word Indian legal prompts into coherent 128-word legal completions.
* **Winning Architecture:** **Architecture 4 (Hybrid Local-Global GQA Decoder, 201.44M params)**
* **Implementation Scripts:**
  - `architecture-4-hybrid-local-global-grouped-query.ipynb` (Model Definition & Initial Training)
  - `train-arch4-t4-v4-1.ipynb` (Primary 10,000-step pretraining run on single T4 GPU; consumed 163.84M tokens)
  - `train-arch4-t4-v4-1_genarate arch2.ipynb` (KV-cached generation with dynamic n-gram banning and repetition penalty)
* **Checkpoints:** `best.pt` (published as `arch4_hybrid_gqa_201m/best.pt`)
* **Alternative Explored Architectures:**
  - Arch 1 (Standard Causal Decoder, 88.2M): `arch1_model.ipynb`, `arch1_genarate.ipynb`
  - Arch 2 (Encoder-Decoder, 160.7M): `arch2-model.ipynb`, `arch2 FULL ENCODER-DECODER TRANSFORMER .ipynb`, `arch2-preln-eval-generate.ipynb`
  - Arch 3 (Sliding Window Decoder, 242.5M): `arach3-train.ipynb`, `arach3-inference.ipynb`
* **Official Performance:** **0.8322 BERTScore F1** (Submission 964323, Rank 1).

---

### Phase 2: Legal Named Entity Recognition (L-NER)
* **Core Task:** Exact character span boundary prediction across 12 legal entity classes from unannotated judgment texts.
* **Winning Architecture:** **Architecture 4 Backbone (201.44M) + Linear Token BIO Classification Head**
* **Adaptation Technique:** LoRA ($r=16, \alpha=32$) on query/value projections + Class-weighted focal loss + leading-space offset correction (`correct_offsets()`).
* **Implementation Scripts:**
  - `phase2/phase2-arch4-ner.ipynb` (Primary fine-tuning & submission pipeline)
  - `phase2/phase2-experiments2.ipynb` (6-layer student Knowledge Distillation)
  - `phase2/full-fine-tuning-for-phase-2-legal-ner-architect.ipynb` (Full fine-tuning ablation)
  - `phase2/phase2-experiments.ipynb` (Multi-backbone comparative lab)
* **Checkpoints:** `best_strict_macro_f1.pt` (0.5429 Strict Macro-F1), `train_windows.pt`, `validation_windows.pt`.
* **Official Performance:** **0.5429 Strict Macro-F1** (Submission 978703, Rank 1).

---

### Phase 3: Rhetorical Role Prediction (RR)
* **Core Task:** Discourse classification of judicial sentences into 13 rhetorical functional roles across Corporate Law and Income Tax domains.
* **Winning Architecture:** **Architecture 1 Backbone (88.2M) + LoRA ($r=8$) + Hierarchical Contextual Sentence Classifier**
* **Adaptation Technique:** LoRA ($r=8, \alpha=16$) + predecessor conditioning $s_{t-1} \to s_t$ + multi-domain joint training + focal loss ($\gamma=2.0$).
* **Implementation Scripts:**
  - `phase3/kaggle-phase3-data.ipynb` (Winning pipeline with Arch 1 + LoRA + Context)
  - `phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` (Arch 4 transfer ablation)
* **Checkpoints:** `best_arch1_context.pt`, `best_arch1_lora_rr.pt`.
* **Official Performance:** **0.4807 Macro-F1** (Submission 994294, Rank 1).

---

### Phase 4: Court Judgment Prediction (CJPE)
* **Core Task:** Binary prediction of case disposition (*Dismissed = 0* vs. *Allowed = 1*) on multi-thousand-word Indian legal appeals.
* **Winning Architecture:** **Architecture 2 Backbone (160.7M) + BitFit Partial Layer Unfreezing + Lexical Tail Ensemble**
* **Adaptation Technique:** 4-Zone document slicing (Head, Mid-1, Mid-2, Tail) + BitFit bias optimization (0.32M trainable params) + Out-of-fold lexical Tail classifier ensembling ($0.40 \times \text{Neural} + 0.60 \times \text{Lexical}$).
* **Implementation Scripts:**
  - `phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb` (Winning Arch 2 BitFit + Lexical pipeline)
  - `phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb` (Arch 4 + Compulsory LORMA)
  - `phase4/exp2-arch3-adalora.ipynb` (Arch 3 + AdaLoRA)
  - `phase4/phase-4-exp-2-2-arch3-adalora-full-corpus.ipynb` (Arch 3 full corpus scaling)
  - `phase4/notebook1fa8d9d541.ipynb` (AdaLoRA inference)
  - `phase4/phase4-kd.ipynb` (Arch 3 Teacher -> Arch 1 Student Knowledge Distillation)
* **Checkpoints:** `run_arch2_preln_efficient/best.pt`, `classifier_head.pt`, `adalora_adapter`.
* **Official Performance:** **0.7106 Macro-F1** (Submission 1009979, Rank 1).

---

### Phase 5: Legal Statute Identification (LSI)
* **Core Task:** Multi-label classification identifying which of 100 Indian statute IDs apply to a given judicial decision.
* **Winning Architecture:** **Architecture 4 Backbone (201.44M) + Dual-View Multi-Label Head + Calibrated Thresholding**
* **Adaptation Technique:** Compulsory LORMA + Dual-View encoding (first 512 + last 512 tokens) + Per-statute sigmoid threshold calibration ($\tau_k \in [0.10, 0.60]$) to overcome 43:1 head-tail statute class imbalance.
* **Implementation Scripts:**
  - `phase5/phase5.ipynb` (Complete dual-view, multi-label training, calibration, and inference pipeline)
* **Checkpoints:** `arch4_hybrid_gqa_201m/best.pt`, calibrated threshold array `statute_calibrated_thresholds.npy`.
* **Official Performance:** **0.3392 Macro-F1** (Submission 1043558, Rank 1).

---

### Phase 6: Legal Abstractive Summarization
* **Core Task:** Compressing multi-thousand-word court rulings into faithful ~638-word legal summaries (4.90x compression).
* **Winning Architecture:** **Architecture 2 Pre-LN Encoder-Decoder (160.7M) + Houlsby Bottleneck Adapters**
* **Adaptation Technique:** Houlsby bottleneck adapters ($r=16$) + Salient chunk-and-merge cross-attention encoding + Length-penalized beam search (beam width 4, length penalty 1.0, repetition penalty 1.25).
* **Implementation Scripts:**
  - `phase6/phase6-2.ipynb` (Winning Arch 2 Houlsby adapter pipeline)
  - `phase6/phase6-3.ipynb` (Arch 4 + Compulsory LORMA continuation)
  - `phase6/phase6.ipynb` (Cross-architecture comparative summarization lab)
* **Checkpoints:** `run_arch2_preln_efficient/best.pt`, `checkpoints/exp_name.pt`.
* **Official Performance:** **0.8389 BERTScore F1** (Submission 1030574, Rank 1).

---

### Phase 7: Prior Case Retrieval (PCR)
* **Core Task:** Identifying and ranking binding precedent cases from candidate judgment pools (evaluated via Micro-F1@7).
* **Winning Architecture:** **Symmetric Bi-Encoder Tower (Arch 4 201M Backbone + Arch 1 Distilled Student) + BM25 & 2-Hop Graph Diffusion**
* **Adaptation Technique:** Compulsory LORMA + Multiple Negative Ranking Loss (in-batch negatives) + Sparse BM25 lexical fusion + Statutory regex cross-linking + 2-hop citation graph diffusion ($k=6, \tau=2.5$).
* **Implementation Scripts:**
  - `phase7/phase7-start.ipynb` (Complete bi-encoder, contrastive training, BM25, graph diffusion, and submission pipeline)
* **Checkpoints:** `arch4_hybrid_gqa_201m/best.pt`, `run_arch1_real/best.pt`.
* **Official Performance:** **0.3627 Micro-F1@7** (Submission 1038781, Rank 1).

---

### Shared & Project Infrastructure Files
* **Technical Documentation:** `report.tex`, `Capstone Document.docx`, `Phase_2_and_3_AI_ML_Report.pdf`, `phase567.pdf`
* **Competition Records:** `codabench_all_77_submissions.csv`, `codabench phase1.png`
* **Evidence & Verification Suite:** `section3_audit/section3_evidence_audit.md`, `section3_audit/verification_report.md`, `section3_audit/plot_data.py`, `section3_audit/preprocessing_trace.csv`, 8 domain statistics CSVs, 6 publication PNG figures.
* **System Metadata:** `desktop.ini`
