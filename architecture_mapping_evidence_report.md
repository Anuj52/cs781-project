# CS781 – Architecture Mapping Evidence Report

## Executive Overview
This document details the rigorous empirical evidence behind every architecture, phase, and role assignment for all 56 files in the CS781 project. Assignments are verified against code implementations, class signatures, variable definitions, dataset file references, logged evaluation metrics, and official competition submission IDs recorded in `codabench_all_77_submissions.csv`.

---

## Summary Mapping Table

| File Path | Suggested Architecture | Phase | Role | Verification | Primary Evidence Key |
|---|---|---|---|---|---|
| `arch1_model.ipynb` | Arch1 (Standard Causal Decoder) | Phase 1 – Legal LM Pretraining | Training | Automatically Detected | `LegalDecoderLM`, `CausalSelfAttention`, `d_model=768`, `n_layers=8`, `n_heads=8`, `run_arch1_real/best.pt` |
| `arch1_genarate.ipynb` | Arch1 (Standard Causal Decoder) | Phase 1 – Legal LM Pretraining | Inference | Automatically Detected | Loads `run_arch1_real/best.pt`, evaluates test prompts, Submission 956293 |
| `arch2-model.ipynb` | Arch2 (Pre-LN Encoder-Decoder) | Phase 1 – Legal LM Pretraining | Training | Automatically Detected | `TransformerModelPreLN`, 6 enc + 6 dec layers, `d_model=768`, `run_arch2_preln_efficient/best.pt` |
| `arch2 FULL ENCODER-DECODER TRANSFORMER .ipynb` | Arch2 (Post-LN Encoder-Decoder) | Phase 1 – Legal LM Pretraining | Model Definition | Automatically Detected | `TransformerModel` (Post-LN), loss progression, gradient instability ablation |
| `arch2-preln-eval-generate.ipynb` | Arch2 (Pre-LN Encoder-Decoder) | Phase 1 – Legal LM Pretraining | Inference | Automatically Detected | Loads `run_arch2_preln_efficient/latest.pt`, prompt continuation evaluation |
| `arach3-train.ipynb` | Arch3 (Sliding Window Decoder) | Phase 1 – Legal LM Pretraining | Training | Automatically Detected | `SlidingWindowCausalSelfAttention`, `d_model=1024`, `n_layers=16`, `run_arch3_swa_v2/best.pt` |
| `arach3-inference.ipynb` | Arch3 (Sliding Window Decoder) | Phase 1 – Legal LM Pretraining | Inference | Automatically Detected | Loads `run_arch3_swa_v2/latest.pt`, val loss 2.381, prompt evaluation |
| `architecture-4-hybrid-local-global-grouped-query.ipynb` | Arch4 (Hybrid Local-Global GQA) | Phase 1 – Legal LM Pretraining | Model Definition | Automatically Detected | `Architecture4`, `GQAAttention`, `SwiGLU`, `RotaryEmbedding`, `arch4_hybrid_gqa_195m/best.pt` |
| `train-arch4-t4-v4-1.ipynb` | Arch4 (Hybrid Local-Global GQA) | Phase 1 – Legal LM Pretraining | Training | Automatically Detected | 10,000 steps, 163.84M tokens, 201.44M params, `arch4_hybrid_gqa_201m/best.pt` |
| `arch4ka2part.ipynb` | Arch4 (Hybrid Local-Global GQA) | Phase 1 – Legal LM Pretraining | Training | Automatically Detected | Continuation training run, `arch4_hybrid_gqa_195m/best.pt` |
| `arch4-1-gen.ipynb` | Arch4 (Hybrid Local-Global GQA) | Phase 1 – Legal LM Pretraining | Inference | Automatically Detected | Generates continuations with 195M resume checkpoint |
| `train-arch4-t4-v4-1_genarate arch.ipynb` | Arch4 (Hybrid Local-Global GQA) | Phase 1 – Legal LM Pretraining | Inference | Automatically Detected | Loads `arch4_hybrid_gqa_201m/best.pt`, top-p generation, Submissions 960793, 963120 |
| `train-arch4-t4-v4-1_genarate arch2.ipynb` | Arch4 (Hybrid Local-Global GQA) | Phase 1 – Legal LM Pretraining | Inference | Automatically Detected | `FastGen`, `NGramBan`, Submission 964323 (Peak BERTScore 0.8322, Rank 1) |
| `cs781-phase1-data.ipynb` | None (Data Pipeline) | Phase 1 – Legal LM Pretraining | Data Preprocessing | Automatically Detected | Ingests Acts, SC, Tribunals; splits 98% train / 2% val; filters OCR noise |
| `phase1-tokens-v40k-clean.ipynb` | None (Tokenizer Pipeline) | Phase 1 – Legal LM Pretraining | Data Preprocessing | Automatically Detected | 40,000 Byte-Level BPE tokenizer, builds `acts_train.bin`, `sc_train.bin`, `tribunals_train_f.bin` |
| `phase2/phase2-arch4-ner.ipynb` | Arch4 Backbone + BIO Head | Phase 2 – Legal NER | Fine-tuning | Automatically Detected | Arch4 201M backbone + 25-way BIO head, `correct_offsets()`, Sub 978703 (Strict F1: 0.5429, Rank 1) |
| `phase2/full-fine-tuning-for-phase-2-legal-ner-architect.ipynb` | Arch4 Backbone | Phase 2 – Legal NER | Fine-tuning | Automatically Detected | Full fine-tuning ablation (lr=2e-5) across cross-validation folds |
| `phase2/phase2-experiments.ipynb` | Comparative Backbones | Phase 2 – Legal NER | Evaluation | Automatically Detected | Comparative benchmarking of Arch 1, 2, 3, 4 with LoRA, BitFit, LORMA |
| `phase2/phase2-experiments2.ipynb` | Arch4 Teacher + 6-Layer Student | Phase 2 – Legal NER | Training | Automatically Detected | `HoulsbyAfterLinear`, `StudentLoRALinear`, KD distillation of 12-layer Arch4 to 6-layer student |
| `phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Arch4 Backbone + Context Head | Phase 3 – Rhetorical Role Prediction | Fine-tuning | Automatically Detected | `SentenceClassifier`, `ContextSentenceClassifier`, $s_{t-1} \to s_t$ conditioning |
| `phase3/kaggle-phase3-data.ipynb` | Arch1 + LoRA + Context Head | Phase 3 – Rhetorical Role Prediction | Fine-tuning | Automatically Detected | Arch1 backbone + LoRA (r=8) + joint CL/IT training, Sub 994294 (Macro-F1: 0.4807, Rank 1) |
| `phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb` | Arch4 201M + LORMA | Phase 4 – Court Judgment Prediction | Fine-tuning | Automatically Detected | `Arch4HierarchicalLoRMANet`, `LoRMALinear`, 4-zone document representation, Macro-F1: 0.6935 |
| `phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb` | Arch2 + BitFit + Lexical Blend | Phase 4 – Court Judgment Prediction | Fine-tuning | Automatically Detected | Arch2 encoder + BitFit bias tuning + Tail Ridge/LogReg blend, Sub 1009979 (Macro-F1: 0.7106, Rank 1) |
| `phase4/exp2-arch3-adalora.ipynb` | Arch3 242M + AdaLoRA | Phase 4 – Court Judgment Prediction | Fine-tuning | Automatically Detected | AdaLoRA on Arch3 backbone, zone attention weights (Head 68.4%, Tail 9.9%), Macro-F1: 0.7027 |
| `phase4/phase-4-exp-2-2-arch3-adalora-full-corpus.ipynb` | Arch3 242M + AdaLoRA | Phase 4 – Court Judgment Prediction | Fine-tuning | Automatically Detected | Full corpus training (39,375 cases) with class-balanced focal loss, Sub 1008332 (Macro-F1: 0.7053) |
| `phase4/notebook1fa8d9d541.ipynb` | Arch3 242M + AdaLoRA | Phase 4 – Court Judgment Prediction | Inference | Automatically Detected | Generates test predictions for 1,517 unannotated cases, Sub 1008459 |
| `phase4/phase4-kd.ipynb` | Arch3 Teacher -> Arch1 Student | Phase 4 – Court Judgment Prediction | Training | Automatically Detected | `Arch3AdaLoRATeacherNet`, `Arch1DistilledStudentNet`, KL divergence distillation |
| `phase5/phase5.ipynb` | Arch4 201M + Multi-Label Head | Phase 5 – Legal Statute Identification | Fine-tuning | Automatically Detected | Dual-view encoding (first 512 + last 512), 100 statute classes, per-statute threshold calibration, Sub 1043558 (Macro-F1: 0.3392, Rank 1) |
| `phase6/phase6-2.ipynb` | Arch2 + Houlsby Adapters | Phase 6 – Abstractive Summarization | Fine-tuning | Automatically Detected | Arch2 Pre-LN Seq2Seq + Houlsby adapters (r=16) + beam search, Sub 1030574 (BERTScore F1: 0.8389, Rank 1) |
| `phase6/phase6-3.ipynb` | Arch4 201M + LORMA | Phase 6 – Abstractive Summarization | Fine-tuning | Automatically Detected | Arch4 decoder + compulsory LORMA continuation, Submissions 1029803, 1029788 |
| `phase6/phase6.ipynb` | Arch2 & Arch4 Comparative Lab | Phase 6 – Abstractive Summarization | Evaluation | Automatically Detected | Comparative evaluation between Seq2Seq cross-attention and decoder continuation |
| `phase7/phase7-start.ipynb` | Arch4 Bi-Encoder + BM25/Graph | Phase 7 – Prior Case Retrieval | Fine-tuning | Automatically Detected | Symmetric bi-encoder + LORMA + Multiple Negative Ranking Loss + BM25 + 2-hop graph diffusion, Sub 1038781 (Micro-F1@7: 0.3627, Rank 1) |
| `report.tex` | Shared (Arch1, Arch2, Arch3, Arch4) | Shared / Multiple Phases | Other | Automatically Detected | Master LaTeX capstone report documenting all 7 phases, models, formulas, and results |
| `Capstone Document.docx` | Shared | Shared / Multiple Phases | Configuration | Automatically Detected | Official course syllabus defining constraints, parameter limits (50M-300M), rules |
| `Phase_2_and_3_AI_ML_Report.pdf` | Shared | Shared / Multiple Phases | Other | Automatically Detected | Formal milestone report for Phase 2 (Legal NER) and Phase 3 (Rhetorical Role Prediction) |
| `phase567.pdf` | Shared | Shared / Multiple Phases | Other | Automatically Detected | Formal milestone report for Phase 5 (Statute), Phase 6 (Summary), Phase 7 (PCR) |
| `codabench_all_77_submissions.csv` | Shared | Shared / Multiple Phases | Evaluation | Automatically Detected | Complete dump of all 77 official Codabench competition submissions across Phases 1--7 |
| `codabench phase1.png` | Arch4 | Phase 1 – Legal LM Pretraining | Evaluation | Automatically Detected | Visual verification of Phase 1 leaderboard rank 1 (BERTScore 0.8322) |
| `desktop.ini` | None | Unassigned | Other | Automatically Detected | Windows desktop folder configuration file; no code or model contents |
| `section3_audit/plot_data.py` | None | Shared / Multiple Phases | Other | Automatically Detected | Standalone Python script that generates publication Figures 1--6 |
| `section3_audit/section3_evidence_audit.md` | Shared | Shared / Multiple Phases | Other | Automatically Detected | Formal verification report linking every metric to specific notebook cells |
| `section3_audit/verification_report.md` | Shared | Shared / Multiple Phases | Other | Automatically Detected | Consistency audit checking text vs code values |
| `section3_audit/preprocessing_trace.csv` | Shared | Shared / Multiple Phases | Data Preprocessing | Automatically Detected | Audit traceability log linking each finding to specific files and notebook cells |
| `section3_audit/phase1_corpus_statistics.csv` | None | Phase 1 – Legal LM Pretraining | Data Preprocessing | Automatically Detected | Exact token and document statistics for Acts, SC, Tribunals |
| `section3_audit/phase2_ner_statistics.csv` | None | Phase 2 – Legal NER | Data Preprocessing | Automatically Detected | Gold entity span counts, alignments, recoverability for all 12 NER classes |
| `section3_audit/phase3_role_statistics.csv` | None | Phase 3 – Rhetorical Role Prediction | Data Preprocessing | Automatically Detected | Sentence counts and class balance across CL and IT for 13 rhetorical roles |
| `section3_audit/phase4_judgment_statistics.csv` | None | Phase 4 – Court Judgment Prediction | Data Preprocessing | Automatically Detected | Class distributions, word length distributions, and zone weights |
| `section3_audit/phase5_statute_statistics.csv` | None | Phase 5 – Legal Statute Identification | Data Preprocessing | Automatically Detected | Document counts, multi-label statistics, and 43:1 head-to-tail imbalance |
| `section3_audit/phase6_summary_statistics.csv` | None | Phase 6 – Abstractive Summarization | Data Preprocessing | Automatically Detected | Document and summary word counts, 4.90x compression ratio |
| `section3_audit/phase7_retrieval_statistics.csv` | None | Phase 7 – Prior Case Retrieval | Data Preprocessing | Automatically Detected | Query cases, candidate pools, and top-7 precision/recall metrics |
| `section3_audit/fig1_phase1_mixture.png` to `fig6_phase6_compression.png` | None | Corresponding Phase | Other | Automatically Detected | Publication figures verifying empirical distributions |

---

## Detailed Evidence Lineage

### Architecture 4: The 201.44M Foundation Model
* **Analytic Formulation in `report.tex` (Line 319-325):**
  $$N_{\text{emb}} = 40,000 \times 1024 = 40,960,000$$
  $$N_{\text{attn}} = 12 \times (1024 \times 2048 + 1024^2) = 37,748,736$$
  $$N_{\text{ffn}} = 12 \times (3 \times 1024 \times 3328) = 122,683,392$$
  $$N_{\text{norm}} = 12 \times (2 \times 1024) + 1024 = 25,600$$
  $$\text{Total Parameters} = 201,443,328\ (\mathbf{201.44M})$$
* **Empirical Code Corroboration:**
  In `train-arch4-t4-v4-1.ipynb` (Cell 2):
  ```python
  total_params = sum(p.numel() for p in model.parameters())
  # Output: 201,443,328 (201.44M params)
  ```
* **Training Lineage:**
  `train-arch4-t4-v4-1.ipynb` executed for exactly 10,000 optimizer steps at a sequence length of 1,024, batch size 8, and gradient accumulation 2 ($10,000 \times 16,384 = 163,840,000$ tokens consumed).
* **Generation & Leaderboard Validation:**
  In `train-arch4-t4-v4-1_genarate arch2.ipynb` (Cell 3), `FastGen` with dynamic n-gram banning generated the test continuations that scored **0.8322 BERTScore F1** (Submission 964323, Rank 1).

### Phase 2: Legal Named Entity Recognition
* **Code Corroboration in `phase2/phase2-arch4-ner.ipynb`:**
  - Loads Phase 1 foundation checkpoint from `/kaggle/input/.../arch4_hybrid_gqa_201m/best.pt`.
  - Appends `nn.Linear(1024, 25)` for 12 BIO entity classes + `O`.
  - Uses `correct_offsets()` to resolve BPE whitespace token boundary shifts.
  - Achieved peak validation Strict Macro-F1 of **0.5429** at Epoch 3 (matching Submission 978703).

### Phase 3: Rhetorical Role Prediction
* **Code Corroboration in `phase3/kaggle-phase3-data.ipynb`:**
  - Loads Arch 1 backbone (`run_arch1_real/best.pt`).
  - Injects LoRA with rank $r=8$ and $\alpha=16$ into attention projections.
  - Implements contextual sentence classification conditioning $s_t$ on $s_{t-1}$.
  - Produced peak competition Macro-F1 of **0.4807** (Submission 994294).

### Phase 4: Court Judgment Prediction
* **Code Corroboration in `phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb`:**
  - Employs 4-zone document slicing (Head, Mid-1, Mid-2, Tail).
  - Adapts Arch 2 Pre-LN encoder via BitFit (only bias terms trained, 0.32M parameters).
  - Blends neural probability with Tail word/char n-gram Ridge and Logistic Regression models ($0.40 \times \text{Neural} + 0.60 \times \text{Lexical}$).
  - Achieved peak Macro-F1 of **0.7106** (Submission 1009979).

### Phase 5: Legal Statute Identification
* **Code Corroboration in `phase5/phase5.ipynb`:**
  - Encodes dual-view (first 512 + last 512 tokens) with Arch 4 backbone.
  - Applies compulsory LORMA adaptation.
  - Calibrates per-statute decision thresholds $\tau_k \in [0.10, 0.60]$ across 100 classes on 10,181 dev judgments.
  - Improves development Macro-F1 from 29.4% to 35.59%, yielding competition score **0.3392** (Submission 1043558).

### Phase 6: Legal Abstractive Summarization
* **Code Corroboration in `phase6/phase6-2.ipynb`:**
  - Loads Arch 2 Pre-LN Seq2Seq backbone (`run_arch2_preln_efficient/best.pt`).
  - Injects Houlsby bottleneck adapters ($r=16$) into encoder and decoder layers.
  - Generates summaries via length-penalized beam search (beam width 4, length penalty 1.0, min length 200).
  - Achieved peak competition score of **0.8389 BERTScore F1** (Submission 1030574).

### Phase 7: Prior Case Retrieval
* **Code Corroboration in `phase7/phase7-start.ipynb`:**
  - Encodes case text into 768-dimensional normalized dense vectors using Arch 4 backbone + Arch 1 distilled student.
  - Trained using Multiple Negative Ranking Loss with in-batch negatives.
  - Fuses dense cosine similarity with BM25, statutory section regex matchers, and 2-hop reciprocal citation graph diffusion ($k=6, \tau=2.5$).
  - Achieved official score of **0.3627 Micro-F1@7** (Submission 1038781).
