# CS781 – Duplicate and Conflicting Architecture Report

## Purpose of this Report
In an extensive research project developed across remote GPU sessions (Kaggle/Colab), model architectures frequently appear across multiple files. This report documents, explains, and resolves:
1. Verbatim code duplications across notebooks.
2. Architecture evolution and parameter discrepancies (e.g. Arch 4 195M vs. 201M).
3. Structural variant conflicts (e.g. Arch 2 Pre-LN vs. Post-LN).
4. Multi-architecture notebooks (ensembles, distillation, comparative labs).
5. File naming anomalies (e.g. `arach3` vs `arch3`).

---

## 1. Verbatim Architecture Code Duplication Across Notebooks
### The Kaggle Standalone Execution Pattern
Unlike a modular local Python package (`import my_model`), Kaggle notebooks execute in isolated container environments. Importing code between notebooks requires publishing intermediate Kaggle Datasets or Utility Scripts. To ensure zero-dependency, one-click reproducibility, the team embedded model classes verbatim into downstream notebooks.

* **Architecture 4 (GQA + SwiGLU + RoPE) Duplication:**
  The classes `Architecture4`, `Arch4Block`, `GQAAttention`, `SwiGLU`, and `RotaryEmbedding` appear identically across:
  - `architecture-4-hybrid-local-global-grouped-query.ipynb` (Root)
  - `train-arch4-t4-v4-1.ipynb` (Root)
  - `arch4ka2part.ipynb` (Root)
  - `arch4-1-gen.ipynb` (Root)
  - `train-arch4-t4-v4-1_genarate arch.ipynb` (Root)
  - `train-arch4-t4-v4-1_genarate arch2.ipynb` (Root)
  - `phase2/phase2-arch4-ner.ipynb` (Phase 2)
  - `phase2/phase2-experiments2.ipynb` (Phase 2)
  - `phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` (Phase 3)
  - `phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb` (Phase 4)
  - `phase5/phase5.ipynb` (Phase 5)
  - `phase6/phase6-3.ipynb` (Phase 6)
  - `phase7/phase7-start.ipynb` (Phase 7)
  *Resolution:* These are not competing or conflicting models. They represent the exact same foundation architecture (`Arch 4`), replicated to permit standalone execution in each task container.

* **Architecture 1 (Causal Decoder) Duplication:**
  Appears in `arch1_model.ipynb`, `arch1_genarate.ipynb`, `phase2/phase2-experiments.ipynb`, `phase3/kaggle-phase3-data.ipynb`, `phase4/phase4-kd.ipynb`, `phase6/phase6.ipynb`, `phase7/phase7-start.ipynb`.

* **Architecture 2 (Encoder-Decoder) Duplication:**
  Appears in `arch2-model.ipynb`, `arch2-preln-eval-generate.ipynb`, `phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb`, `phase6/phase6-2.ipynb`, `phase6/phase6.ipynb`.

* **Architecture 3 (Sliding Window Decoder) Duplication:**
  Appears in `arach3-train.ipynb`, `arach3-inference.ipynb`, `phase4/exp2-arch3-adalora.ipynb`, `phase4/phase-4-exp-2-2-arch3-adalora-full-corpus.ipynb`, `phase4/phase4-kd.ipynb`.

---

## 2. Architecture Parameter Variations: Arch 4 195M vs 201.44M
* **Discrepancy:**
  In early Phase 1 notebooks (`architecture-4-hybrid-local-global-grouped-query.ipynb`, `arch4ka2part.ipynb`, `arch4-1-gen.ipynb`), parameter counts of **195M** (`arch4_hybrid_gqa_195m/best.pt`) are reported. In later notebooks (`train-arch4-t4-v4-1.ipynb`, `train-arch4-t4-v4-1_genarate arch2.ipynb`, and downstream phases), the parameter count is finalized at **201,443,328 (201.44M)** (`arch4_hybrid_gqa_201m/best.pt`).
* **Root Cause Analysis:**
  The 195M prototype used an intermediate feed-forward dimension $d_{\text{ff}} = 3072$. During final pretraining on the T4 GPU (`train-arch4-t4-v4-1.ipynb`), the feed-forward dimension was expanded to $d_{\text{ff}} = 3328$ ($= \frac{13}{4} \times 1024$) to maximize capacity within the 200M compute sweet spot.
* **Resolution:**
  195M is the preliminary development checkpoint; **201.44M is the official, final Phase-1 foundation checkpoint** utilized across all competition submissions.

---

## 3. Structural Variants: Arch 2 Pre-LN vs. Post-LN
* **Discrepancy:**
  Two distinct implementations of Arch 2 exist in the root folder:
  1. `arch2 FULL ENCODER-DECODER TRANSFORMER .ipynb`: Implements standard **Post-LN** (LayerNorm applied after residual connection additions).
  2. `arch2-model.ipynb` and `arch2-preln-eval-generate.ipynb`: Implements **Pre-LN** (`TransformerModelPreLN`, with LayerNorm before multi-head attention and feed-forward sublayers).
* **Root Cause & Experimental Outcome:**
  Post-LN exhibited severe gradient explosion and vanishing during initial training steps, requiring small learning rates and failing to converge below a loss of 10.77. Pre-LN stabilized training completely, allowing higher learning rates ($3 \times 10^{-4}$) and steady convergence.
* **Resolution:**
  Post-LN was an early ablation that was discarded. **Pre-LN Arch 2** (`run_arch2_preln_efficient/best.pt`) is the verified production architecture deployed downstream in Phase 4 (BitFit) and Phase 6 (Houlsby summarization).

---

## 4. Multi-Architecture and Ensembling Notebooks
Several notebooks contain code for multiple architectures simultaneously:
* `phase4/phase4-kd.ipynb`: Implements a **Teacher-Student Pair** where 242.5M Arch 3 (Teacher with AdaLoRA) distills knowledge into 88.2M Arch 1 (Student).
* `phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb`: Ensembles neural representations from **Arch 2 Pre-LN** with **Lexical Classifiers** (Ridge & Logistic Regression trained on Tail word/char n-grams).
* `phase5/phase5.ipynb`, `phase6/phase6.ipynb`, `phase7/phase7-start.ipynb`: Contain comparative harness blocks loading checkpoints from Arch 1, Arch 2, Arch 3, and Arch 4 to evaluate transfer performance and perform late fusion.
* **Resolution:** In the mapping dashboard, these files are assigned to their primary driving architecture while their comparative roles and secondary dependencies are explicitly linked in metadata.

---

## 5. File Naming Typographical Discrepancies
* **Discrepancy:**
  - `arach3-train.ipynb` and `arach3-inference.ipynb` contain the typo `arach3` instead of `arch3`.
  - `arch4ka2part.ipynb` contains informal shorthand (`ka2part` = "part 2").
  - `train-arch4-t4-v4-1_genarate arch.ipynb` contains the typo `genarate`.
  - `phase4/notebook1fa8d9d541.ipynb` retains an auto-generated Kaggle notebook hash name.
* **Resolution:**
  Per rule 7 ("Do not modify, rename, move, or delete original files"), all original filenames and paths are strictly preserved. Their canonical architectural identities (`Arch 3`, `Arch 4`, `Phase 4 Exp 2.2 Inference`) are surfaced cleanly in the discovery dashboard.
