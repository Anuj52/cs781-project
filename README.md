# In-House Legal Intelligence System (CS781 Capstone Project)

End-to-end Indian legal NLP codebase trained strictly from scratch within the 50M–300M parameter budget on single NVIDIA T4 GPUs across seven competition phases.

- **Author**: Anuj Chaudhary (Roll No: `210167`, Codabench: `a_210167`)
- **Institution**: Indian Institute of Technology Kanpur (IIT Kanpur)

---

## Directory Layout

```text
├── code/
│   ├── phase1/   # Phase 1: Foundation Model Pretraining (Arch 1–4, 40k Byte-Level BPE, Hindi/OCR filter)
│   ├── phase2/   # Phase 2: Legal Named Entity Recognition (LoRMA, LoRA, BitFit, KD, Offset Alignment)
│   ├── phase3/   # Phase 3: Rhetorical Role Prediction (LoRMA, Predecessor Context, KD)
│   ├── phase4/   # Phase 4: Court Judgment Prediction (BitFit + Tail TF-IDF Blend, LoRMA, AdaLoRA, KD)
│   ├── phase5/   # Phase 5: Legal Statute Identification (KD, LoRMA, Dual-View, Per-Statute Thresholds)
│   ├── phase6/   # Phase 6: Legal Abstractive Summarization (LoRA, Houlsby Adapters, LoRMA, KD, Beam 4)
│   └── phase7/   # Phase 7: Prior Case Retrieval (KD Bi-Encoder, LoRMA, Hybrid BM25/TF-IDF, 2-Hop Graph)
└── report/       # Local report directory (excluded from Git via .gitignore)
    └── CS781-CP-Anuj-Chaudhary-210167-report.pdf
```

---

## Phase-Wise Best-Performing Methods & Official Scores

| Phase | Task | Best-Performing Method | Backbone / Setup | Primary Metric | Official Score |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **P1** | Legal Language Model Pretraining | From-Scratch Pretraining | Arch 4 Hybrid GQA (201.44M) | BERTScore-F1 | **0.8322** |
| **P2** | Legal Named Entity Recognition (L-NER) | **LoRMA** ($r=16, \alpha=32$) | Arch 4 (201.44M) + BIO Head | Strict Macro-F1 | **0.5429** |
| **P3** | Rhetorical Role Prediction | **LoRMA** ($r=8, \alpha=16$) | Arch 1/4 Contextual Classifier | Macro-F1 | **0.4807** |
| **P4** | Court Judgment Prediction (CJPE) | **BitFit** + Tail TF-IDF Blend | Arch 2 Encoder (160.68M) | Macro-F1 | **0.7106** |
| **P5** | Legal Statute Identification (LSI) | **Knowledge Distillation (KD)** | Arch 4 Teacher $\to$ Arch 1 Student | Macro-F1 | **0.3392** |
| **P6** | Legal Abstractive Summarization | **LoRA** ($r=16, \alpha=32$) + Beam Search | Arch 2 Seq2Seq (160.68M) | BERTScore-F1 | **0.8389** |
| **P7** | Prior Case Retrieval (PCR) | **Knowledge Distillation (KD)** + 2-Hop Graph | Arch 4 Teacher $\to$ Arch 1 Bi-Encoder | Micro-F1@7 | **0.3627** |

---

## Code Index by Phase

### `code/phase1/` — Foundation Model Pretraining & Data Pipeline
* **Data Filtering & Byte-Level BPE Tokenization (40k vocab, OCR & $\sim 1\%$ Hindi/Devanagari filter):**
  * `code/phase1/cs781-phase1-data.ipynb` — Streams Acts, Supreme Court judgments, and 10% sampled Tribunal orders into deterministic 98/2 splits.
  * `code/phase1/phase1-tokens-v40k-clean.ipynb` — Strips OCR control characters and $\sim 1\%$ Hindi (Devanagari) script, trains the 40,000-token Byte-Level BPE tokenizer, and packs `uint16` binary shards.
* **Architecture 4 (201.44M Hybrid Local-Global GQA Decoder — Primary Backbone):**
  * `code/phase1/architecture-4-hybrid-local-global-grouped-query.ipynb` — Arch 4 definition (`L-L-G` schedule, RoPE, SwiGLU, GQA).
  * `code/phase1/train-arch4-t4-v4-1.ipynb` — 10,000-step T4 pretraining loop (163.84M tokens).
  * `code/phase1/train-arch4-t4-v4-1_genarate arch2.ipynb`, `code/phase1/train-arch4-t4-v4-1_genarate arch.ipynb`, `code/phase1/arch4-1-gen.ipynb`, `code/phase1/arch4ka2part.ipynb` — KV-cached autoregressive continuation with 4-gram blocking and repetition penalty.
* **Comparative Phase-1 Backbones (Arch 1, Arch 2, Arch 3):**
  * `code/phase1/arch1_model.ipynb`, `code/phase1/arch1_genarate.ipynb` — Architecture 1 (88.21M Standard Causal GPT Decoder).
  * `code/phase1/arch2-model.ipynb`, `code/phase1/arch2 FULL ENCODER-DECODER TRANSFORMER .ipynb`, `code/phase1/arch2-preln-eval-generate.ipynb` — Architecture 2 (160.68M Pre-LN Encoder-Decoder).
  * `code/phase1/arach3-train.ipynb`, `code/phase1/arach3-inference.ipynb` — Architecture 3 (242.54M Pure Sliding-Window Decoder).

### `code/phase2/` — Legal Named Entity Recognition
* `code/phase2/phase2-experiments.ipynb` — **LoRMA** ($r=16, \alpha=32$) multiplicative token classification, LoRA, BitFit, and leading-space offset alignment (`correct_offsets()`).
* `code/phase2/phase2-arch4-ner.ipynb` — Arch 4 25-tag BIO sequence labeling and sliding-window inference.
* `code/phase2/phase2-experiments2.ipynb` — Knowledge Distillation (12-layer Arch 4 teacher $\to$ 6-layer student preserving `L-L-G` pattern).
* `code/phase2/full-fine-tuning-for-phase-2-legal-ner-architect.ipynb` — Full fine-tuning baseline ablation.

### `code/phase3/` — Rhetorical Role Prediction
* `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` — **LoRMA** adaptation, gated contextual sentence classification ($s_{t-1} \to s_t$), and Knowledge Distillation across Corporate Law (CL) and Income Tax (IT).
* `code/phase3/kaggle-phase3-data.ipynb` — Predecessor-conditioned training with class-weighted focal loss ($\gamma=2.0$).

### `code/phase4/` — Court Judgment Prediction
* `code/phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb` — **BitFit** (bias-only tuning) on 4-zone sliced inputs (Head, Mid-1, Mid-2, Tail) blended with out-of-fold Tail word/character TF-IDF Ridge & Logistic classifiers.
* `code/phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb` — Compulsory LoRMA hierarchical 4-zone classifier on Arch 4.
* `code/phase4/exp2-arch3-adalora.ipynb`, `code/phase4/phase-4-exp-2-2-arch3-adalora-full-corpus.ipynb`, `code/phase4/notebook1fa8d9d541.ipynb` — Arch 3 AdaLoRA adaptive rank allocation experiments.
* `code/phase4/phase4-kd.ipynb` — Knowledge Distillation (242.5M Arch 3 teacher $\to$ 88.2M Arch 1 student).

### `code/phase5/` — Legal Statute Identification
* `code/phase5/phase5.ipynb` — **Knowledge Distillation (KD)** multi-label student training, compulsory LoRMA ($r=32, \alpha=64$), Dual-View input encoding (first 512 + last 512 tokens), and per-statute threshold tuning ($\tau_k \in [0.10, 0.60]$).

### `code/phase6/` — Legal Abstractive Summarization
* `code/phase6/phase6.ipynb` — **LoRA** ($r=16, \alpha=32$) Seq2Seq cross-attention adaptation, Knowledge Distillation, and length-penalized beam search ($B=4$).
* `code/phase6/phase6-2.ipynb` — Arch 2 Pre-LN Seq2Seq with Houlsby bottleneck adapters ($r=16$) and chunk-and-merge encoding.
* `code/phase6/phase6-3.ipynb` — Compulsory LoRMA ($r=32, \alpha=64$) causal prompt-continuation summarization on Arch 4.

### `code/phase7/` — Prior Case Retrieval
* `code/phase7/phase7-start.ipynb` — **Knowledge Distillation (KD)** from Arch 4 Bi-Encoder teacher into Arch 1 student (`Phase1BiEncoder`), compulsory LoRMA ($r=16, \alpha=32$), Multiple Negatives Ranking Loss, hybrid TF-IDF/BM25/citation scoring, and 2-hop citation graph diffusion.
