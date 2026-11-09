# In-House Legal Intelligence System — CS781 Capstone Project

**Public GitHub Repository:** [https://github.com/Anuj52/cs781-project](https://github.com/Anuj52/cs781-project)  
**Author:** Anuj Chaudhary  
**Roll Number:** `210167`  
**Codabench Username:** `a_210167`  
**Email:** `anujc21@iitk.ac.in`  
**Institution:** Indian Institute of Technology Kanpur (IIT Kanpur)

---

## Table of Contents

1. [Project Overview & Design Constraints](#1-project-overview--design-constraints)
2. [Repository Structure](#2-repository-structure)
3. [Summary of Phase-Wise Best-Performing Methods & Official Scores](#3-summary-of-phase-wise-best-performing-methods--official-scores)
4. [Model Architectures (Architectures 1–4)](#4-model-architectures-architectures-14)
5. [Transfer Learning, PEFT & Knowledge Distillation Formulations](#5-transfer-learning-peft--knowledge-distillation-formulations)
6. [Phase-by-Phase Methodology, Implementation Details & Submissions](#6-phase-by-phase-methodology-implementation-details--submissions)
   - [Phase 1: Legal Language Model Pretraining](#phase-1-legal-language-model-pretraining)
   - [Phase 2: Legal Named Entity Recognition (L-NER)](#phase-2-legal-named-entity-recognition-l-ner)
   - [Phase 3: Rhetorical Role Prediction](#phase-3-rhetorical-role-prediction)
   - [Phase 4: Court Judgment Prediction (CJPE)](#phase-4-court-judgment-prediction-cjpe)
   - [Phase 5: Legal Statute Identification (LSI)](#phase-5-legal-statute-identification-lsi)
   - [Phase 6: Legal Abstractive Summarization](#phase-6-legal-abstractive-summarization)
   - [Phase 7: Prior Case Retrieval (PCR)](#phase-7-prior-case-retrieval-pcr)
7. [Complete Master Hyperparameter Table](#7-complete-master-hyperparameter-table)
8. [Experimental Setup & Instructions to Run the Code](#8-experimental-setup--instructions-to-run-the-code)
9. [Complete Log of All 77 Official Codabench Submissions](#9-complete-log-of-all-77-official-codabench-submissions)

---

## 1. Project Overview & Design Constraints

Law firms and judicial chambers handle privileged case files, client communications, and unfiled legal drafts that cannot be transmitted to external commercial LLM APIs due to strict confidentiality obligations. This project builds and evaluates a self-hosted, seven-phase **In-House Legal Intelligence System** trained strictly from scratch on Indian legal texts.

Every phase and submission in this repository adheres to the following competition constraints:
1. **Zero External Pretrained Weights or Paid APIs**: All four Transformer architectures and the 40,000-token Byte-Level BPE tokenizer are initialized randomly and trained strictly from scratch in PyTorch on the competition's Phase-1 Indian legal corpus.
2. **Parameter Budget ($50\text{M} \le N_{\text{parameters}} \le 300\text{M}$)**: Every foundational model and downstream model satisfies the 50M–300M parameter envelope, verified analytically and via runtime parameter assertions (`sum(p.numel() for p in model.parameters())`).
3. **Downstream Phase-1 Initialization Lock**: Every downstream phase (Phases 2–7) initializes its encoder/decoder weights strictly from our Phase-1 pretrained checkpoints (`arch4_hybrid_gqa_201m/best.pt`, `run_arch1_real/best.pt`, `run_arch2_preln_efficient/best.pt`, or `run_arch3_swa_v2/best.pt`).
4. **Compulsory LoRMA & Mandatory Knowledge Distillation (KD)**: In every downstream phase (Phases 2–7), we implement and evaluate compulsory **Low-Rank Multiplicative Adaptation (LoRMA)** and teacher–student **Knowledge Distillation (KD)** alongside distinct PEFT families (**LoRA**, **AdaLoRA**, **BitFit**, **Houlsby Bottleneck Adapters**, and **Layer-Wise Partial Unfreezing**).
5. **Single-GPU Compute Budget**: All pretraining, fine-tuning, distillation, and inference runs are engineered to execute within a single 16 GB NVIDIA Tesla T4 GPU budget using FP16/BF16 mixed precision, fused AdamW, gradient accumulation, and memory-mapped (`np.memmap`) token streams.

---

## 2. Repository Structure

```text
├── README.md                                  # Master documentation covering all 7 phases & 77 submissions
├── .gitignore                                 # Tracks only code/phase1..7, plot_data.py, and READMEs
├── .gitattributes                             # Git text/EOL attributes
├── code/
│   ├── phase1/                                # Phase 1: Legal LM Pretraining & 40k BPE Tokenizer
│   │   ├── README.md
│   │   ├── cs781-phase1-data.ipynb
│   │   ├── phase1-tokens-v40k-clean.ipynb
│   │   ├── architecture-4-hybrid-local-global-grouped-query.ipynb
│   │   ├── train-arch4-t4-v4-1.ipynb
│   │   ├── train-arch4-t4-v4-1_genarate arch2.ipynb
│   │   ├── train-arch4-t4-v4-1_genarate arch.ipynb
│   │   ├── arch4-1-gen.ipynb
│   │   ├── arch4ka2part.ipynb
│   │   ├── arch1_model.ipynb
│   │   ├── arch1_genarate.ipynb
│   │   ├── arch2-model.ipynb
│   │   ├── arch2 FULL ENCODER-DECODER TRANSFORMER .ipynb
│   │   ├── arch2-preln-eval-generate.ipynb
│   │   ├── arach3-train.ipynb
│   │   └── arach3-inference.ipynb
│   ├── phase2/                                # Phase 2: Legal Named Entity Recognition (L-NER)
│   │   ├── README.md
│   │   ├── phase2-experiments.ipynb
│   │   ├── phase2-arch4-ner.ipynb
│   │   ├── phase2-experiments2.ipynb
│   │   └── full-fine-tuning-for-phase-2-legal-ner-architect.ipynb
│   ├── phase3/                                # Phase 3: Rhetorical Role Prediction
│   │   ├── README.md
│   │   ├── arch-4-phase-3-rhetorical-role-prediction.ipynb
│   │   └── kaggle-phase3-data.ipynb
│   ├── phase4/                                # Phase 4: Court Judgment Prediction (CJPE)
│   │   ├── README.md
│   │   ├── p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb
│   │   ├── p4-experiment-1-arch4-201m-compulsory-lorma.ipynb
│   │   ├── exp2-arch3-adalora.ipynb
│   │   ├── phase-4-exp-2-2-arch3-adalora-full-corpus.ipynb
│   │   ├── notebook1fa8d9d541.ipynb
│   │   └── phase4-kd.ipynb
│   ├── phase5/                                # Phase 5: Legal Statute Identification (LSI)
│   │   ├── README.md
│   │   └── phase5.ipynb
│   ├── phase6/                                # Phase 6: Legal Abstractive Summarization
│   │   ├── README.md
│   │   ├── phase6.ipynb
│   │   ├── phase6-2.ipynb
│   │   └── phase6-3.ipynb
│   └── phase7/                                # Phase 7: Prior Case Retrieval (PCR)
│       ├── README.md
│       └── phase7-start.ipynb
├── section3_audit/
│   └── plot_data.py                           # Python script generating corpus analysis plots
└── report/                                    # Local report folder (ignored in Git for code-only push)
    └── CS781-CP-Anuj-Chaudhary-210167-report.pdf
```

---

## 3. Summary of Phase-Wise Best-Performing Methods & Official Scores

| Phase | Task | Total Submissions | Best Submission ID | Best-Performing Method | Backbone / Setup | Primary Metric | Official Score | Leaderboard Rank |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- | :---: | :---: |
| **P1** | Legal LM Pretraining | 6 | `964323` | **Scratch Pretrain + FastGen KV-Cache** | Arch 4 Hybrid GQA (201.44M) | BERTScore-F1 | **0.8322** | 61 |
| **P2** | Legal Named Entity Recognition | 13 | `978703` | **LoRMA** ($r=16, \alpha=32$) + `correct_offsets()` | Arch 4 (201.44M) + 25-BIO Head | Strict Macro-F1 | **0.5429** | 55 |
| **P3** | Rhetorical Role Prediction | 26 | `994294` | **LoRMA** ($r=8, \alpha=16$) + Predecessor Context | Arch 1/4 Contextual Classifier | Macro-F1 | **0.4807** | 78 |
| **P4** | Court Judgment Prediction | 11 | `1009979` | **BitFit** (0.32M) + OOF Tail TF-IDF Blend | Arch 2 Encoder (160.68M, 4-Zone) | Macro-F1 | **0.7106** | 76 |
| **P5** | Legal Statute Identification | 6 | `1043558` | **Knowledge Distillation (KD)** + Per-Statute $\tau_k$ | Arch 4 Teacher $\to$ Arch 1 Student | Macro-F1 | **0.3392** | 93 |
| **P6** | Abstractive Summarization | 8 | `1030574` | **LoRA** ($r=16, \alpha=32$) + Beam Search ($B=4$) | Arch 2 Pre-LN Seq2Seq (160.68M) | BERTScore-F1 | **0.8389** | 65 |
| **P7** | Prior Case Retrieval (PCR) | 7 | `1038781` | **Knowledge Distillation (KD)** + 2-Hop Citation Graph | Arch 4 Teacher $\to$ Arch 1 Bi-Encoder | Micro-F1@7 | **0.3627** | 79 |

---

## 4. Model Architectures (Architectures 1–4)

We designed, implemented from PyTorch primitives (`torch.nn.Module`), and pretrained **four distinct Transformer architectures** on the Phase-1 legal corpus to study how attention topology, positional encoding, and encoder-decoder vs. decoder-only structure impact downstream Indian legal tasks:

| Architecture | Model Family | Layers ($L$) | Hidden Dim ($d_{\text{model}}$) | Attention Heads | FFN Dim ($d_{\text{ff}}$) | Positional Encoding | Exact Parameters | Foundational Checkpoint |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :--- |
| **Arch 1** | Standard Causal GPT Decoder | 8 | 768 | 8 MHA ($d_{\text{head}}=96$) | 3,072 (GELU) | Learned Absolute ($1024$) | **88,206,336** (88.21M) | `run_arch1_real/best.pt` |
| **Arch 2** | Pre-LN Seq2Seq Encoder-Decoder | 6 Enc + 6 Dec | 768 | 8 MHA ($d_{\text{head}}=96$) | 3,072 (GELU) | Sinusoidal / Learned | **160,680,960** (160.68M) | `run_arch2_preln_efficient/best.pt` |
| **Arch 3** | Sliding-Window Causal Decoder | 16 | 1,024 | 16 SWA ($W=256$) | 4,096 (GELU) | RoPE ($\theta=10,000$) | **242,536,960** (242.54M) | `run_arch3_swa_v2/best.pt` |
| **Arch 4** | **Hybrid Local-Global GQA Decoder** | 12 (`L-L-G` $\times 4$) | 1,024 | 16 Q / 8 KV GQA ($d_{\text{head}}=64$) | 3,328 (SwiGLU) | RoPE ($\theta=10,000$) | **201,443,328** (201.44M) | `arch4_hybrid_gqa_201m/best.pt` |

### Detailed Specification of Architecture 4 (Primary 201.44M Backbone)
- **Token & Output Embeddings**: Vocabulary size $|V| = 40,000$ (`<pad>=0`, `<bos>=1`, `<eos>=2`), embedding dimension $d_{\text{model}} = 1024$, weight-tied between input token embeddings and the output LM head ($40,000 \times 1024 = 40,960,000$ parameters).
- **Interleaved Local-Global Schedule (`L-L-G`)**: Across the 12 decoder blocks, layers $\{0,1, 3,4, 6,7, 9,10\}$ apply **Local Causal Sliding-Window Attention** with window size $W=256$ ($\mathcal{O}(N \cdot W)$ complexity), while every third layer $\{2, 5, 8, 11\}$ applies **Full Global Causal Attention** ($\mathcal{O}(N^2)$ complexity). This preserves long-range statutory and precedent cross-referencing across 1,024 tokens while reducing activation memory on 16 GB T4 GPUs.
- **Grouped-Query Attention (GQA)**: 16 query heads ($d_{\text{head}}=64$) share 8 key-value heads (2:1 KV group sharing), cutting KV-cache memory in half during autoregressive continuation (`FastGen`) and reducing projection parameters per block to $3,145,728$.
- **Rotary Position Embeddings (RoPE)**: Applied to query and key projections at every layer with base period $\theta = 10,000$.
- **SwiGLU Feed-Forward Network**: Each block uses a gated SwiGLU FFN:
  $$\text{SwiGLU}(x) = \left(\text{Swish}(x W_{\text{gate}}) \odot x W_{\text{up}}\right) W_{\text{down}}, \quad d_{\text{ff}} = 3328$$
  contributing $3 \times (1024 \times 3328) = 10,223,616$ parameters per layer.
- **Pre-LayerNorm**: LayerNorm ($4,096$ parameters per block + $2,048$ final norm) is applied before each attention and FFN sub-layer for gradient stability under FP16 mixed-precision training without warmup spikes.
- **Exact Parameter Verification**:
  $$N_{\text{Arch4}} = 40,960,000 + 12 \times (3,145,728 + 10,223,616 + 4,096) + 2,048 = 201,443,328 \text{ parameters (201.44M)}$$

---

## 5. Transfer Learning, PEFT & Knowledge Distillation Formulations

Across Phases 2–7, we freeze or selectively adapt the Phase-1 pretrained weights $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ using six distinct adaptation mechanisms:

1. **Low-Rank Multiplicative Adaptation (LoRMA — Compulsory in Phases 2–7; Best in P2 & P3)**:
   Modulates frozen pretrained projections multiplicatively via low-rank factors $B \in \mathbb{R}^{d_{\text{out}} \times r}$ (initialized to zero) and $A \in \mathbb{R}^{r \times d_{\text{in}}}$ (initialized via Kaiming uniform):
   $$h = \left(I_{d_{\text{out}}} + \frac{\alpha}{r} B A\right) W_0 x$$
   Implemented in custom `LORMALinear` modules wrapping attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and classification heads.
2. **Low-Rank Additive Adaptation (LoRA — Best in P6)**:
   Adds a parallel low-rank residual branch to frozen weights $W_0$:
   $$h = W_0 x + \frac{\alpha}{r} B A x$$
3. **Adaptive Budget Allocation LoRA (AdaLoRA — Explored in P4)**:
   Parameterizes the incremental update in singular-value form $W = W_0 + P \Lambda Q$ and dynamically prunes singular triplets $\Lambda_{ii}$ based on sensitivity-smoothed importance scores during training.
4. **Bias-Term Fine-Tuning (BitFit — Best in P4)**:
   Freezes all weight matrices $W_0$ and updates only bias vectors $b$ and LayerNorm affine parameters ($\gamma, \beta$), training just $0.32\text{M}$ parameters ($0.20\%$ of the 160.68M Arch 2 encoder) to prevent overfitting on noisy multi-thousand-word appeals.
5. **Houlsby Bottleneck Adapters (Explored in P6)**:
   Inserts residual bottleneck modules $h \leftarrow h + W_{\text{up}}\,\sigma(W_{\text{down}}\,h)$ ($r=16$) after multi-head attention and feed-forward sub-layers in the Arch 2 Seq2Seq model.
6. **Teacher–Student Knowledge Distillation (KD — Mandatory in Phases 2–7; Best in P5 & P7)**:
   Trains a compact student network $S_\phi$ (e.g., 6-layer Arch 4 student in P2 or 88.21M Arch 1 student in P4, P5, P7) to match both gold labels and temperature-scaled soft distributions ($\tau_{\text{KD}} = 2.0$) from a larger fine-tuned teacher $T_\theta$:
   $$\mathcal{L}_{\text{KD}} = (1 - \lambda)\,\mathcal{L}_{\text{task}}(y, S_\phi(x)) + \lambda\,\tau_{\text{KD}}^2\,\mathrm{KL}\!\left(\sigma(T_\theta(x)/\tau_{\text{KD}}) \,\|\, \sigma(S_\phi(x)/\tau_{\text{KD}})\right)$$

---

## 6. Phase-by-Phase Methodology, Implementation Details & Submissions

### Phase 1: Legal Language Model Pretraining
- **Folder:** [`code/phase1/`](code/phase1/)
- **Task:** Autoregressive domain continuation — given a 512-word Indian legal prompt, generate a coherent 128-word legal continuation for 1,000 test prompts.
- **Data Engineering & Hindi/OCR Filtering (`cs781-phase1-data.ipynb`, `phase1-tokens-v40k-clean.ipynb`):**
  - Raw pool: 18,826 Central/State Acts, 55,549 Supreme Court judgments, 869,132 Tribunal orders, and $>1\text{M}$ High Court files.
  - High Court files were excluded due to severe OCR corruption and single-T4 compute budget limits; we retained 100% of clean Acts and Supreme Court judgments and deterministically sampled 10% of Tribunal orders.
  - **OCR & Hindi (Devanagari) Script Removal**: We dropped Tribunal files containing $>0.1\%$ C1 control characters (`[\x80-\x9f]`) or $>5\%$ question marks (`?`), and stripped Hindi (Devanagari, `[\u0900-\u097F]+`) script (`STRIP_DEVANAGARI = True`). Hindi text accounted for only $\sim 1\%$ of the corpus, and Byte-Level BPE performs poorly on Devanagari script because multi-byte UTF-8 characters (3 bytes per character) get split into fragmented byte tokens, wasting vocabulary merges and inflating sequence lengths.
  - **40,000-Token Byte-Level BPE & Binary Packing**: Trained a 40,000-vocab Byte-Level BPE tokenizer (`add_prefix_space=False`, special tokens `<pad>=0`, `<bos>=1`, `<eos>=2`) on balanced slices (3,000 docs/source) and packed 157,950 training documents (532.99M tokens: 145.60M Acts, 249.83M SC, 137.57M Tribunals) and 18,577 validation documents (36.62M tokens) into flat `uint16` binary files (`np.memmap`).
- **Pretraining & Decoding (`train-arch4-t4-v4-1.ipynb`, `train-arch4-t4-v4-1_genarate arch2.ipynb`):**
  - Trained Arch 4 (201.44M) for 10,000 steps (sequence length 1,024, micro-batch 8, gradient accumulation 2 $\Rightarrow$ 16,384 tokens/step = 163.84M tokens: 27.3% Acts, 46.9% SC, 25.8% Tribunals) using fused AdamW ($\text{lr}=3\times 10^{-4}$, cosine decay, 300-step warmup, weight decay 0.10, grad clip 1.0).
  - Reached validation Cross-Entropy loss **2.4759** (Perplexity **11.89**).
  - Generated 128-word continuations using custom KV-cached `FastGen` with nucleus sampling ($p=0.90$, temperature $T=0.70$), repetition penalty $\theta_{\text{rep}}=1.15$, and dynamic 4-gram blocking.
- **Phase 1 Submissions (6 Total):**
  - `956293` (`0.8176` BERTScore-F1): Arch 1 (88.21M) baseline causal decoder + top-$p$ sampling.
  - `960793` (`0.8232` BERTScore-F1): Arch 4 (201.44M) early checkpoint + top-$p$ sampling ($p=0.92, T=0.80$).
  - `963120` (`0.8295` BERTScore-F1): Arch 4 (201.44M) 10k-step checkpoint + top-$p$ sampling ($p=0.90, T=0.75$).
  - `964111` (`0.8319` BERTScore-F1): Arch 4 (201.44M) + repetition penalty ($\theta_{\text{rep}}=1.15$).
  - `964202` (`0.8305` BERTScore-F1): Arch 4 (201.44M) + greedy/beam length-tuned continuation ablation.
  - **`964323` (`0.8322` BERTScore-F1 — Best Phase-1 Submission)**: Arch 4 (201.44M) + `FastGen` KV-cache, dynamic 4-gram blocking, $\theta_{\text{rep}}=1.15$, $p=0.90, T=0.70$, post-processed to trim trailing incomplete clauses.

---

### Phase 2: Legal Named Entity Recognition (L-NER)
- **Folder:** [`code/phase2/`](code/phase2/)
- **Task:** Extract exact character start/end offsets and labels across 12 legal entity classes (`COURT`, `PETITIONER`, `RESPONDENT`, `JUDGE`, `LAWYER`, `DATE`, `ORG`, `GPE`, `STATUTE`, `PROVISION`, `PRECEDENT`, `CASE_NUMBER`, `WITNESS`, `OTHER_PERSON`) evaluated by **Strict Macro-F1**.
- **Methodology & Key Engineering Fix (`phase2-experiments.ipynb`, `phase2-arch4-ner.ipynb`, `phase2-experiments2.ipynb`):**
  - **Leading-Space Offset Correction (`correct_offsets()`)**: Raw Byte-Level BPE attaches preceding whitespace (e.g., `Ġ`) to the start of subword tokens, shifting decoded character spans left by 1 character and causing Strict Macro-F1 to assign 0 credit even when the entity text is right. Our deterministic `correct_offsets()` function strips leading/trailing whitespace and adjusts `(start, end)` indices, recovering 153 boundary spans on train (98.18% span recoverability) and jumping test Strict Macro-F1 by **+0.2201** (`0.3228` $\to$ `0.5429`).
  - **Sliding-Window BIO Tagging**: Judgments are segmented into overlapping windows (1,024 tokens with 768 stride for Arch 4; 512 tokens with 384 stride for Arch 1/2/KD) over 25 BIO tags (`O` + $12\times\{\text{B},\text{I}\}$).
  - **Best-Performing Method — LoRMA (`0.5429` Strict Macro-F1, Submission `978703`)**: Applies multiplicative low-rank adaptation (`LORMALinear`, $r=16, \alpha=32$, $\text{lr}=5\times 10^{-4}$, class-weighted focal loss $\gamma=2.0$) to the 201.44M Arch 4 backbone with disjoint non-overlapping span argmax decoding.
  - **Mandatory Knowledge Distillation (`phase2-experiments2.ipynb`)**: Distills the 12-layer Arch 4 teacher into a 6-layer Arch 4 student (retaining layers `[0,1,2,6,7,8]` to preserve the `L-L-G` local-global attention pattern), achieving `0.5164` Strict Macro-F1 (`978669`) at $2\times$ inference speed.
- **Phase 2 Submissions (13 Total):**
  - `973794` (`0.3545`), `973804` (`0.3228`): Initial Arch 4 transfer runs before full `correct_offsets()` boundary alignment.
  - `974820` (`0.1209`), `974822` (`0.1261`), `976644` (`0.1880`), `977217` (`0.1725`): Unaligned BIO decoding and high-temperature threshold exploratory runs.
  - `976907` (`0.5043`), `977156` (`0.4959`), `977191` (`0.4514`): Full fine-tuning (`full-fine-tuning-for-phase-2-legal-ner-architect.ipynb`) and BitFit ablations with `correct_offsets()`.
  - `978571` (`0.5135`), `978669` (`0.5164`), `978765` (`0.5023`): 6-layer Knowledge Distillation student (`phase2-experiments2.ipynb`) and LoRA ablations.
  - **`978703` (`0.5429` Strict Macro-F1 — Best Phase-2 Submission)**: Arch 4 (201.44M) + **LoRMA** ($r=16, \alpha=32$) + class-weighted focal loss + `correct_offsets()` + disjoint span argmax.

---

### Phase 3: Rhetorical Role Prediction
- **Folder:** [`code/phase3/`](code/phase3/)
- **Task:** Classify every sentence in Corporate Law (CL) and Income Tax (IT) judgments into 13 rhetorical roles (`Fact`, `Issue`, `ArgumentPetitioner`, `ArgumentRespondent`, `Statute`, `PrecedentRelied`, `PrecedentNotRelied`, `PrecedentOverruled`, `RatioOfTheDecision`, `RulingByLowerCourt`, `RulingByPresentCourt`, `Dissent`, `None`), evaluated by **Macro-F1**.
- **Methodology & Approach (`arch-4-phase-3-rhetorical-role-prediction.ipynb`, `kaggle-phase3-data.ipynb`):**
  - **Predecessor Context Conditioning ($s_{t-1} \to s_t$)**: Isolated sentences such as *"The appeal is accordingly dismissed"* or *"He submitted that the notice was time-barred"* are ambiguous without discourse context. We format each input as `prev_sentence <eos> curr_sentence` (up to 256 tokens) and pass pooled hidden states $[h_{t-1}; h_t; |h_t - h_{t-1}|]$ into a gated contextual classification head, improving Macro-F1 by **+0.1345** over isolated sentence classification (`0.3462` $\to$ `0.4807`).
  - **Multi-Domain Joint Training & Focal Loss**: Jointly trains across Corporate Law (11,989 sentences) and Income Tax (6,998 sentences) using inverse square-root class weights $w_c \propto N_c^{-0.5}$ and focal loss ($\gamma=2.0$) to handle rare roles (`Dissent`, `PrecedentOverruled`).
  - **Best-Performing Method — LoRMA (`0.4807` Macro-F1, Submission `994294`)**: Uses multiplicative adaptation ($r=8, \alpha=16$) with predecessor context conditioning and focal loss ($\gamma=2.0$).
- **Phase 3 Submissions (26 Total):**
  - `990876` (`0.3005`), `990963` (`0.2776`), `991035` (`0.3254`), `991240` (`0.3161`): Isolated sentence baselines without predecessor context across Arch 1–4.
  - `992698` (`0.3450`), `992701` (`0.3462`), `992707` (`0.3462`), `993228` (`0.3701`), `993259` (`0.3701`), `993404` (`0.3443`), `994291` (`0.3778`): Single-sentence BitFit, full fine-tuning, and unweighted cross-entropy ablations.
  - `993009` (`0.4262`), `993426` (`0.4352`), `993466` (`0.4564`), `993890` (`0.4593`), `993892` (`0.4593`), `993899` (`0.4608`), `993179` (`0.4616`), `994340` (`0.4521`): Contextual gated classifiers on Arch 2, Arch 3, and Arch 4 with LoRA and Knowledge Distillation (`0.4679` in `993934`/`994028`).
  - `993928` (`0.4633`), `993939` (`0.4607`), `994049` (`0.4673`): Multi-backbone temperature-scaled soft-logit distillation and label-smoothing runs.
  - **`994208` & `994294` (`0.4807` Macro-F1 — Best Phase-3 Submission)**: **LoRMA** ($r=8, \alpha=16$) + predecessor context conditioning ($s_{t-1} \to s_t$) + class-weighted focal loss ($\gamma=2.0$).

---

### Phase 4: Court Judgment Prediction (CJPE)
- **Folder:** [`code/phase4/`](code/phase4/)
- **Task:** Binary outcome prediction (`Dismissed = 0` vs. `Allowed = 1`) on long Indian appeal judgments (mean 3,446 words; 90th percentile $>12,500$ words), evaluated by **Macro-F1**.
- **Methodology & Approach (`p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb`, `p4-experiment-1-arch4-201m-compulsory-lorma.ipynb`, `exp2-arch3-adalora.ipynb`, `phase4-kd.ipynb`):**
  - **4-Zone Slicing ($4 \times 384$ Tokens)**: Because a single 1,024-token prefix truncates the operative holding at the end of a judgment, we extract four 384-token zones from every case: **Head** ($0\text{--}10\%$, parties & procedural posture), **Mid-1** ($30\text{--}40\%$, factual narrative), **Mid-2** ($65\text{--}75\%$, statutory analysis), and **Tail** ($90\text{--}100\%$, operative disposition) and combine their zone embeddings via softmax zone attention.
  - **Best-Performing Method — BitFit + Lexical Tail Ensemble (`0.7106` Macro-F1, Submission `1009979`)**:
    - Updates only the **bias parameters (BitFit, 0.32M trainable parameters = 0.20%)** and top-layer norms of the Phase-1 Arch 2 bidirectional encoder across the 4 zones ($\text{lr}=1\times 10^{-3}$, 9 epochs).
    - Blends the neural probability $p_{\text{BitFit}}(x)$ with an out-of-fold 5-fold CV lexical ensemble $p_{\text{Lexical}}(x)$ (word 1–3 gram + character 3–5 gram sublinear TF-IDF trained with Ridge Classifier $\alpha=1.0$ and $L_2$ Logistic Regression $C=0.5$ on the Tail zone):
      $$p_{\text{final}}(x) = 0.40\,p_{\text{BitFit}}(x) + 0.60\,p_{\text{Lexical}}(x), \quad \hat{y} = \mathbb{I}[p_{\text{final}}(x) \ge 0.48]$$
  - **Compulsory LoRMA, AdaLoRA & Knowledge Distillation**:
    - `p4-experiment-1-arch4-201m-compulsory-lorma.ipynb`: Compulsory **LoRMA** (`Arch4HierarchicalLoRMANet`) on Arch 4 (`0.6935` Macro-F1 in `1006441`).
    - `exp2-arch3-adalora.ipynb` & `phase-4-exp-2-2-arch3-adalora-full-corpus.ipynb`: **AdaLoRA** ($r_{\text{init}}=12 \to r_{\text{target}}=4$) on 242.54M Arch 3 (`0.7053` Macro-F1 in `1008332`).
    - `phase4-kd.ipynb`: Mandatory **Knowledge Distillation** from 242.54M Arch 3 AdaLoRA teacher into 88.21M Arch 1 student (`0.6878` Macro-F1 in `1006424`).
- **Phase 4 Submissions (11 Total):**
  - `1006682` (`0.5573`): Head-only 1,024-token truncation baseline (demonstrating why 4-zone slicing is essential).
  - `1006385` (`0.6800`), `1006424` (`0.6878`), `1006441` (`0.6935`), `1010019` (`0.6618`), `1010026` (`0.6633`): Arch 4 Compulsory LoRMA (`p4-experiment-1-arch4-201m-compulsory-lorma.ipynb`) and Arch 3$\to$Arch 1 Knowledge Distillation (`phase4-kd.ipynb`).
  - `1008710` (`0.6633`), `1008459` (`0.7027`), `1008332` (`0.7053`): Arch 3 (242.54M) AdaLoRA 4-zone experiments (`exp2-arch3-adalora.ipynb`).
  - `1010013` (`0.7093`): Arch 2 BitFit + Lexical Tail blend with default threshold $\tau=0.50$.
  - **`1009979` (`0.7106` Macro-F1 — Best Phase-4 Submission)**: Arch 2 **BitFit** (0.32M params) + 5-fold OOF Tail TF-IDF Ridge/Logistic blend ($\beta=0.40$) + validation-calibrated threshold $\tau=0.48$.

---

### Phase 5: Legal Statute Identification (LSI)
- **Folder:** [`code/phase5/`](code/phase5/)
- **Task:** Multi-label classification over $K=100$ Indian statutes on 42,750 train, 10,181 dev, and 13,019 test cases (mean 3.71 statutes/doc; $43.13:1$ head-to-tail imbalance between Statute 69 at 27.44% and the rarest statute at 0.64%), evaluated by **Macro-F1**.
- **Methodology & Approach (`phase5.ipynb`):**
  - **Dual-View Representation ($2 \times 512$ Tokens)**: Concatenates the first 512 tokens (opening facts and FIR charges) and last 512 tokens (final statutory conviction/acquittal) of each judgment and pools them via attention+mean+max pooling.
  - **Best-Performing Method — Knowledge Distillation (KD) + Per-Statute Threshold Calibration (`0.3392` Macro-F1, Submission `1043558`)**:
    - Distills soft multi-label probabilities ($\tau_{\text{KD}}=2.0$) from our Dual-View Arch 4 teacher (trained with compulsory `LORMALinear`, $r=32, \alpha=64$, and positive-weighted BCE) into our compact Arch 1 student model.
    - Runs coordinate-wise threshold search over $\tau_k \in [0.10, 0.60]$ on the 10,181 validation judgments (`statute_calibrated_thresholds.npy`), assigning lower thresholds ($\tau_k \approx 0.12\text{--}0.22$) to rare statutes and stricter thresholds ($\tau_k \approx 0.45\text{--}0.55$) to frequent statutes. Per-statute calibration improved official test Macro-F1 by **+0.0753** (`0.2639` $\to$ `0.3392`).
- **Phase 5 Submissions (6 Total):**
  - `1043556` (`0.2639`): Fixed global threshold ($\tau = 0.50$) baseline with Dual-View LoRMA.
  - `1043564` (`0.2940`): Global scalar threshold optimization ($\tau = 0.28$) on validation set.
  - `1043557` (`0.3327`), `1043562` (`0.3334`), `1043566` (`0.3375`): Compulsory LoRMA ($r=32, \alpha=64$) and LoRA Dual-View models with per-statute calibrated thresholds.
  - **`1043558` (`0.3392` Macro-F1 — Best Phase-5 Submission)**: **Knowledge Distillation (KD)** (Arch 4 Dual-View teacher $\to$ Arch 1 student) + per-statute calibrated thresholds $\tau_k \in [0.10, 0.60]$.

---

### Phase 6: Legal Abstractive Summarization
- **Folder:** [`code/phase6/`](code/phase6/)
- **Task:** Generate abstractive summaries ($\sim 638$ median words) from long Indian court judgments ($\sim 3,128$ median words; $4.90:1$ compression ratio) across 6,677 clean training pairs (after filtering 1 empty target record `ID_87502`), 352 validation pairs, and 100 test documents, evaluated by **BERTScore-F1**.
- **Methodology & Approach (`phase6.ipynb`, `phase6-2.ipynb`, `phase6-3.ipynb`):**
  - **Salient Chunk-and-Merge Encoding**: Selects top-scoring TF-IDF/lead/conclusion chunks from the source judgment in chronological order up to 1,024 source tokens so both the opening factual background and final judicial holding are preserved in the encoder context.
  - **Best-Performing Method — LoRA on Arch 2 Seq2Seq + Beam Search (`0.8389` BERTScore-F1, Submission `1030574`)**:
    - Adapts the Phase-1 pretrained Arch 2 Pre-LN Encoder-Decoder (160.68M) with **LoRA** ($r=16, \alpha=32$) across encoder self-attention, decoder self-attention, and cross-attention projections ($\text{lr}=3\times 10^{-4}$, label smoothing $\epsilon=0.10$).
    - Decodes summaries using length-penalized beam search (beam width $B=4$, length penalty $1.0$, repetition penalty $1.25$, minimum length 200 tokens, maximum length 512 tokens, 4-gram blocking).
  - **Compulsory LoRMA, Houlsby Adapters & Knowledge Distillation**:
    - `phase6-3.ipynb`: Compulsory **LoRMA** ($r=32, \alpha=64$) causal prompt-continuation summarization on Arch 4 (`0.8086` BERTScore-F1 in `1029819`).
    - `phase6-2.ipynb`: **Houlsby Bottleneck Adapters** ($r=16$) on Arch 2 Seq2Seq (`0.8343` BERTScore-F1 in `1029788`/`1029803`).
    - `phase6.ipynb`: Mandatory sequence-level and token-level **Knowledge Distillation** (`0.8242` BERTScore-F1 in `1030555`).
- **Phase 6 Submissions (8 Total):**
  - `1029807` (`0.7612`), `1029814` (`0.7713`), `1029819` (`0.8086`): Arch 4 decoder-only causal continuation with Compulsory LoRMA (`phase6-3.ipynb`).
  - `1030538` (`0.8233`), `1030555` (`0.8242`): Sequence-level Knowledge Distillation student runs (`phase6.ipynb`).
  - `1029788` (`0.8343`), `1029803` (`0.8343`): Arch 2 Seq2Seq with Houlsby bottleneck adapters ($r=16$, `phase6-2.ipynb`).
  - **`1030574` (`0.8389` BERTScore-F1 — Best Phase-6 Submission)**: Arch 2 Pre-LN Seq2Seq (160.68M) + **LoRA** ($r=16, \alpha=32$) + chunk-and-merge source encoding + beam search ($B=4$).

---

### Phase 7: Prior Case Retrieval (PCR)
- **Folder:** [`code/phase7/`](code/phase7/)
- **Task:** Given a query case $q$ and a candidate pool $\mathcal{C}_q$ of 10 candidate precedents (250 train, 50 dev, 100 test queries; mean 4.62 gold positives per query), rank and return the top-$k$ ($k=7$) relevant prior cases, evaluated by **Micro-F1@7**.
- **Methodology & Approach (`phase7-start.ipynb`):**
  - **Full-Document Reconstruction & Symmetric Bi-Encoder**: Concatenates sentence arrays into complete documents, encodes Head+Tail windows (512+512 tokens), and trains a symmetric bi-encoder using **Multiple Negatives Ranking Loss** with in-batch negatives ($\tau=0.05$) and compulsory `LORMALinear` ($r=16, \alpha=32$) projections.
  - **Best-Performing Method — Knowledge Distillation (KD) + Hybrid Lexical Fusion + 2-Hop Citation Graph Diffusion (`0.3627` Micro-F1@7, Submission `1038781`)**:
    - **Knowledge Distillation**: Distills similarity distributions from the 201.44M Arch 4 Bi-Encoder teacher into the lightweight 88.21M Arch 1 student (`Phase1BiEncoder`, `UniversalBiEncoder_best.pt`).
    - **Hybrid Scoring with Background Subtraction**: Combines dense student cosine similarity $s_{\text{KD}}(q,c)$ with word $n$-gram TF-IDF, character $n$-gram TF-IDF, BM25 ($k_1=1.5, b=0.75$), and regex-extracted statutory/IPC/CrPC citation overlap, while subtracting $0.45\times$ the candidate's mean background score across all queries to suppress generic procedural boilerplate.
    - **2-Hop Citation Graph Diffusion**: Propagates relevance scores over the candidate-to-candidate similarity graph $W_{ij} = \exp(S_{\text{cand}}(c_i, c_j)/\tau_g)$ ($\tau_g=2.5$) for two steps ($\mathbf{s}^{(t+1)} = (1-\alpha_g)\mathbf{s}^{(0)} + \alpha_g W \mathbf{s}^{(t)}$) so indirectly cited landmark precedents are promoted into the top 7 (+0.0266 Micro-F1@7 over dense-only retrieval).
- **Phase 7 Submissions (7 Total):**
  - `1038668` (`0.3361`), `1038754` (`0.3361`): Dense Bi-Encoder baseline with Compulsory LoRMA ($r=16, \alpha=32$) before graph diffusion.
  - `1038764` (`0.3414`), `1038760` (`0.3455`): Dense Bi-Encoder + sparse BM25 and statutory regex overlap fusion.
  - `1038724` (`0.3491`): Arch 4 Bi-Encoder + 1-hop citation graph propagation.
  - `1038801` (`0.3568`): KD Bi-Encoder + 2-hop graph diffusion with uncalibrated background weight.
  - **`1038781` (`0.3627` Micro-F1@7 — Best Phase-7 Submission)**: **Knowledge Distillation (KD)** (Arch 4 teacher $\to$ Arch 1 `Phase1BiEncoder` student) + hybrid TF-IDF/BM25/statute fusion with $0.45\times$ background subtraction + 2-hop citation graph diffusion ($k=7$).

---

## 7. Complete Master Hyperparameter Table

| Hyperparameter | P1 (Pretraining) | P2 (L-NER) | P3 (Rhetorical Roles) | P4 (CJPE) | P5 (Statutes) | P6 (Summarization) | P7 (Retrieval) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Best Method** | Scratch Pretrain | **LoRMA** ($r=16,\alpha=32$) | **LoRMA** ($r=8,\alpha=16$) | **BitFit** + TF-IDF Blend | **KD** + Per-Statute $\tau_k$ | **LoRA** ($r=16,\alpha=32$) | **KD** + 2-Hop Graph |
| **Backbone / Setup** | Arch 4 (201.4M) | Arch 4 (201.4M) | Arch 1/4 Context | Arch 2 Enc (160.7M) | Arch 4 $\to$ Arch 1 | Arch 2 Seq2Seq (160.7M) | Arch 4 $\to$ Arch 1 |
| **Context Length** | 1,024 | 1,024 (stride 768) | 256 ($s_{t-1}, s_t$) | $4 \times 384$ zones | $2 \times 512$ views | 1,024 in / 512 out | $512+512$ / Full doc |
| **Optimizer** | Fused AdamW | AdamW | AdamW | AdamW + Ridge/LogReg | AdamW | AdamW | AdamW |
| **Peak Learning Rate** | $3 \times 10^{-4}$ | $5 \times 10^{-4}$ | $3 \times 10^{-4}$ | $1 \times 10^{-3}$ | $2 \times 10^{-4}$ | $3 \times 10^{-4}$ | $2 \times 10^{-4}$ |
| **Batch $\times$ Grad Accum.** | $8 \times 2$ | $4 \times 4$ | $16 \times 1$ | $16 \times 2$ | $16 \times 2$ | $8 \times 2$ | $16 \times 1$ |
| **Steps / Epochs** | 10,000 steps | 5 epochs | 6 epochs | 9 epochs | 550 steps | 5 epochs | 5 epochs |
| **Dropout / Weight Decay** | $0.0 / 0.10$ | $0.05 / 0.01$ | $0.10 / 0.01$ | $0.10 / 0.01$ | $0.10 / 0.01$ | $0.05 / 0.01$ | $0.10 / 0.01$ |
| **Decoding / Post-Proc.** | $p=0.9, T=0.7$, 4-gram | `correct_offsets()` | Focal $\gamma=2.0$ | Blend $\beta=0.40, \tau=0.48$ | $\tau_k \in [0.10, 0.60]$ | Beam $B=4$, pen $1.25$ | 2-Hop Graph, top-7 |

---

## 8. Experimental Setup & Instructions to Run the Code

### 8.1 Hardware & Software Environment
- **Hardware**: Single NVIDIA Tesla T4 GPU (16 GB VRAM), 2–4 vCPUs, 16–30 GB System RAM (standard Kaggle / Google Colab T4 environment or local CUDA workstation).
- **Software Dependencies**:
  ```bash
  pip install torch>=2.1.0 numpy pandas scikit-learn tokenizers>=0.15.0 \
              huggingface_hub ijson tqdm matplotlib seaborn scipy
  ```

### 8.2 Step-by-Step Execution Guide

1. **Phase 1 — Corpus Ingestion, 40k BPE Tokenizer & Foundation Pretraining (`code/phase1/`)**:
   - **Step 1 (Data Streaming & Splitting)**: Run [`code/phase1/cs781-phase1-data.ipynb`](code/phase1/cs781-phase1-data.ipynb) to stream `train_acts.json`, `train_supreme_court.json`, and 10% of `train_tribunals.json` via `ijson` into deterministic 98/2 CRC32 train/val `.jsonl` splits.
   - **Step 2 (OCR/Hindi Filtering & Binary Packing)**: Run [`code/phase1/phase1-tokens-v40k-clean.ipynb`](code/phase1/phase1-tokens-v40k-clean.ipynb) to strip OCR control characters and $\sim 1\%$ Hindi (Devanagari) script, train the 40,000-token Byte-Level BPE tokenizer (`tokenizer.json`), and write packed `uint16` `.bin` files.
   - **Step 3 (Pretraining Backbones)**:
     - Run [`code/phase1/train-arch4-t4-v4-1.ipynb`](code/phase1/train-arch4-t4-v4-1.ipynb) to train Architecture 4 (201.44M Hybrid Local-Global GQA Decoder) for 10,000 steps and save `arch4_hybrid_gqa_201m/best.pt`.
     - Run [`code/phase1/arch1_model.ipynb`](code/phase1/arch1_model.ipynb), [`code/phase1/arch2-model.ipynb`](code/phase1/arch2-model.ipynb), and [`code/phase1/arach3-train.ipynb`](code/phase1/arach3-train.ipynb) to train Comparative Architectures 1 (88.21M), 2 (160.68M), and 3 (242.54M).
   - **Step 4 (128-Word Continuation & Submission)**: Run [`code/phase1/train-arch4-t4-v4-1_genarate arch2.ipynb`](code/phase1/train-arch4-t4-v4-1_genarate arch2.ipynb) to run `FastGen` KV-cached decoding on the 1,000 test prompts and package `prediction.json` inside `submission.zip`.

2. **Phase 2 — Legal Named Entity Recognition (`code/phase2/`)**:
   - Place `tokenizer.json` and `arch4_hybrid_gqa_201m/best.pt` in the input path.
   - Run [`code/phase2/phase2-experiments.ipynb`](code/phase2/phase2-experiments.ipynb) and [`code/phase2/phase2-arch4-ner.ipynb`](code/phase2/phase2-arch4-ner.ipynb) to apply `correct_offsets()`, fine-tune **LoRMA** ($r=16, \alpha=32$) with class-weighted focal loss, and generate `submission.zip` (`0.5429` Strict Macro-F1).
   - Run [`code/phase2/phase2-experiments2.ipynb`](code/phase2/phase2-experiments2.ipynb) for the 12-layer $\to$ 6-layer Knowledge Distillation student experiment.

3. **Phase 3 — Rhetorical Role Prediction (`code/phase3/`)**:
   - Run [`code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb`](code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb) and [`code/phase3/kaggle-phase3-data.ipynb`](code/phase3/kaggle-phase3-data.ipynb) to construct predecessor sentence pairs ($s_{t-1} \to s_t$), train the **LoRMA** ($r=8, \alpha=16$) contextual classifier and KD student with focal loss ($\gamma=2.0$), and export `submission.zip` (`0.4807` Macro-F1).

4. **Phase 4 — Court Judgment Prediction (`code/phase4/`)**:
   - Run [`code/phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb`](code/phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb) to slice documents into 4 zones ($4\times 384$ tokens), fine-tune Arch 2 via **BitFit** (0.32M params), blend with 5-fold out-of-fold Tail TF-IDF Ridge/Logistic classifiers ($\beta=0.40, \tau=0.48$), and generate `submission.zip` (`0.7106` Macro-F1).
   - Run [`code/phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb`](code/phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb) for Compulsory LoRMA, [`code/phase4/exp2-arch3-adalora.ipynb`](code/phase4/exp2-arch3-adalora.ipynb) for AdaLoRA, and [`code/phase4/phase4-kd.ipynb`](code/phase4/phase4-kd.ipynb) for Arch 3 $\to$ Arch 1 Knowledge Distillation.

5. **Phase 5 — Legal Statute Identification (`code/phase5/`)**:
   - Run [`code/phase5/phase5.ipynb`](code/phase5/phase5.ipynb) to encode Dual-View windows ($2\times 512$ tokens), train the **Knowledge Distillation (KD)** student and Compulsory LoRMA ($r=32, \alpha=64$) models, calibrate per-statute thresholds $\tau_k \in [0.10, 0.60]$ on the 10,181 dev cases, and produce `submission.zip` (`0.3392` Macro-F1).

6. **Phase 6 — Legal Abstractive Summarization (`code/phase6/`)**:
   - Run [`code/phase6/phase6.ipynb`](code/phase6/phase6.ipynb) to adapt Arch 2 Pre-LN Seq2Seq (160.68M) with **LoRA** ($r=16, \alpha=32$) and KD on salient chunk-and-merge source inputs and decode with beam search ($B=4$) to produce `submission.zip` (`0.8389` BERTScore-F1).
   - Run [`code/phase6/phase6-2.ipynb`](code/phase6/phase6-2.ipynb) for Houlsby bottleneck adapters ($r=16$) and [`code/phase6/phase6-3.ipynb`](code/phase6/phase6-3.ipynb) for Compulsory LoRMA on Arch 4.

7. **Phase 7 — Prior Case Retrieval (`code/phase7/`)**:
   - Run [`code/phase7/phase7-start.ipynb`](code/phase7/phase7-start.ipynb) to train the Arch 4 Bi-Encoder teacher with Multiple Negatives Ranking Loss and Compulsory LoRMA ($r=16, \alpha=32$), distill into the Arch 1 **Knowledge Distillation (KD)** student (`Phase1BiEncoder`), fuse with BM25/TF-IDF/statutory regex scores and 2-hop citation graph diffusion, and generate `submission.zip` (`0.3627` Micro-F1@7).

### 8.3 Pre-Submission Verification Protocol
Every notebook runs an automated verification block before writing `submission.zip`:
- Confirms `prediction.json` is placed directly at the root of `submission.zip` without nested subdirectories.
- Verifies exact 1-to-1 key alignment (`set(pred.keys()) == set(test.keys())`) and zero `NaN`/empty outputs.

---

## 9. Complete Log of All 77 Official Codabench Submissions

| # | Phase | Submission ID | Date | Architecture | Adaptation / Method | Notebook File | Primary Metric | Score | Best? |
| :-: | :---: | :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| 1 | P1 | `964323` | 2026-10-06 | Arch 4 (201.4M) | **Scratch Pretrain + FastGen KV-Cache (4-gram block, $\theta_{\text{rep}}=1.15$)** | `code/phase1/train-arch4-t4-v4-1_genarate arch2.ipynb` | BERTScore-F1 | **0.8322** | **Yes** |
| 2 | P1 | `964202` | 2026-10-06 | Arch 4 (201.4M) | Top-$p$ sampling + length-tuned continuation | `code/phase1/train-arch4-t4-v4-1_genarate arch.ipynb` | BERTScore-F1 | 0.8305 | No |
| 3 | P1 | `964111` | 2026-10-06 | Arch 4 (201.4M) | Top-$p$ sampling ($p=0.90, T=0.70, \theta_{\text{rep}}=1.15$) | `code/phase1/train-arch4-t4-v4-1_genarate arch.ipynb` | BERTScore-F1 | 0.8319 | No |
| 4 | P1 | `963120` | 2026-10-06 | Arch 4 (201.4M) | Top-$p$ sampling ($p=0.90, T=0.75$) | `code/phase1/train-arch4-t4-v4-1_genarate arch.ipynb` | BERTScore-F1 | 0.8295 | No |
| 5 | P1 | `960793` | 2026-10-05 | Arch 4 (201.4M) | Intermediate checkpoint + top-$p$ sampling | `code/phase1/train-arch4-t4-v4-1_genarate arch.ipynb` | BERTScore-F1 | 0.8232 | No |
| 6 | P1 | `956293` | 2026-10-02 | Arch 1 (88.2M) | Standard Causal GPT Decoder baseline | `code/phase1/arch1_genarate.ipynb` | BERTScore-F1 | 0.8176 | No |
| 7 | P2 | `978765` | 2026-10-11 | Arch 4 (201.4M) | LoRA ($r=16, \alpha=32$) + window stride ablation | `code/phase2/phase2-arch4-ner.ipynb` | Strict Macro-F1 | 0.5023 | No |
| 8 | P2 | `978703` | 2026-10-11 | Arch 4 (201.4M) | **LoRMA ($r=16, \alpha=32$) + Focal Loss + `correct_offsets()`** | `code/phase2/phase2-experiments.ipynb` | Strict Macro-F1 | **0.5429** | **Yes** |
| 9 | P2 | `978669` | 2026-10-11 | Arch 4 (12L$\to$6L) | Knowledge Distillation Student (`[0,1,2,6,7,8]`) | `code/phase2/phase2-experiments2.ipynb` | Strict Macro-F1 | 0.5164 | No |
| 10 | P2 | `978571` | 2026-10-11 | Arch 4 (201.4M) | LoRA ($r=16, \alpha=32$) + `correct_offsets()` | `code/phase2/phase2-arch4-ner.ipynb` | Strict Macro-F1 | 0.5135 | No |
| 11 | P2 | `977217` | 2026-10-11 | Arch 4 (201.4M) | Unaligned threshold sensitivity test | `code/phase2/phase2-experiments.ipynb` | Strict Macro-F1 | 0.1725 | No |
| 12 | P2 | `977191` | 2026-10-11 | Arch 4 (201.4M) | BitFit bias-only token classification | `code/phase2/phase2-experiments.ipynb` | Strict Macro-F1 | 0.4514 | No |
| 13 | P2 | `977156` | 2026-10-11 | Arch 4 (201.4M) | Top-4 layer partial unfreezing | `code/phase2/phase2-experiments.ipynb` | Strict Macro-F1 | 0.4959 | No |
| 14 | P2 | `976907` | 2026-10-11 | Arch 4 (201.4M) | Full Fine-Tuning + `correct_offsets()` | `code/phase2/full-fine-tuning-for-phase-2-legal-ner-architect.ipynb` | Strict Macro-F1 | 0.5043 | No |
| 15 | P2 | `976644` | 2026-10-11 | Arch 1 (88.2M) | Exploratory window merge without offset fix | `code/phase2/phase2-experiments.ipynb` | Strict Macro-F1 | 0.1880 | No |
| 16 | P2 | `974822` | 2026-10-10 | Arch 4 (201.4M) | Raw BPE span decoding (pre-offset fix) | `code/phase2/phase2-arch4-ner.ipynb` | Strict Macro-F1 | 0.1261 | No |
| 17 | P2 | `974820` | 2026-10-10 | Arch 4 (201.4M) | Raw BPE greedy BIO decoding | `code/phase2/phase2-arch4-ner.ipynb` | Strict Macro-F1 | 0.1209 | No |
| 18 | P2 | `973804` | 2026-10-10 | Arch 4 (201.4M) | LoRA baseline without leading-space strip | `code/phase2/phase2-arch4-ner.ipynb` | Strict Macro-F1 | 0.3228 | No |
| 19 | P2 | `973794` | 2026-10-10 | Arch 4 (201.4M) | Partial whitespace trim baseline | `code/phase2/phase2-arch4-ner.ipynb` | Strict Macro-F1 | 0.3545 | No |
| 20 | P3 | `994340` | 2026-10-16 | Arch 4 (201.4M) | Gated Context Classifier + LoRA | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4521 | No |
| 21 | P3 | `994294` | 2026-10-16 | Arch 1/4 Context | **LoRMA ($r=8, \alpha=16$) + Predecessor Context ($s_{t-1}\to s_t$) + Focal Loss** | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | **0.4807** | **Yes** |
| 22 | P3 | `994291` | 2026-10-16 | Arch 4 (201.4M) | BitFit isolated sentence ablation | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.3778 | No |
| 23 | P3 | `994208` | 2026-10-16 | Arch 1 (88.2M) | Contextual LoRA ($r=8, \alpha=16$) + Focal ($\gamma=2.0$) | `code/phase3/kaggle-phase3-data.ipynb` | Macro-F1 | 0.4807 | No |
| 24 | P3 | `994049` | 2026-10-16 | Arch 4 $\to$ Arch 1 | Knowledge Distillation student ($\tau_{\text{KD}}=2.0$) | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4673 | No |
| 25 | P3 | `994028` | 2026-10-16 | Arch 4 $\to$ Arch 1 | Knowledge Distillation + context concatenation | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4679 | No |
| 26 | P3 | `993939` | 2026-10-16 | Arch 4 (201.4M) | LoRMA ($r=16, \alpha=32$) + Gated Context Head | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4607 | No |
| 27 | P3 | `993934` | 2026-10-16 | Arch 4 $\to$ Arch 1 | Teacher-Student KD soft-logit blend | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4679 | No |
| 28 | P3 | `993928` | 2026-10-16 | Arch 3 (242.5M) | Sliding-Window Contextual LoRA ($r=8$) | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4633 | No |
| 29 | P3 | `993899` | 2026-10-16 | Arch 2 (160.7M) | Pre-LN Encoder Contextual LoRA ($r=8$) | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4608 | No |
| 30 | P3 | `993892` | 2026-10-16 | Arch 4 (201.4M) | Top-2 Layer Partial Unfreezing + Context | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4593 | No |
| 31 | P3 | `993890` | 2026-10-16 | Arch 4 (201.4M) | Top-2 Layer Unfreezing + Label Smoothing | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4593 | No |
| 32 | P3 | `993466` | 2026-10-16 | Arch 1 (88.2M) | Contextual Full Fine-Tuning | `code/phase3/kaggle-phase3-data.ipynb` | Macro-F1 | 0.4564 | No |
| 33 | P3 | `993426` | 2026-10-16 | Arch 1 (88.2M) | Contextual BitFit | `code/phase3/kaggle-phase3-data.ipynb` | Macro-F1 | 0.4352 | No |
| 34 | P3 | `993404` | 2026-10-16 | Arch 4 (201.4M) | Unweighted Cross-Entropy single-sentence | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.3443 | No |
| 35 | P3 | `993259` | 2026-10-16 | Arch 3 (242.5M) | Single-sentence LoRA ($r=8$) | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.3701 | No |
| 36 | P3 | `993228` | 2026-10-16 | Arch 3 (242.5M) | Single-sentence LoRMA ($r=8$) | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.3701 | No |
| 37 | P3 | `993179` | 2026-10-16 | Arch 4 (201.4M) | Contextual LoRA ($r=16$) + inverse class weights | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4616 | No |
| 38 | P3 | `993009` | 2026-10-16 | Arch 2 (160.7M) | Contextual BitFit + Focal Loss | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.4262 | No |
| 39 | P3 | `992707` | 2026-10-16 | Arch 1 (88.2M) | Isolated sentence LoRA ($r=8$) | `code/phase3/kaggle-phase3-data.ipynb` | Macro-F1 | 0.3462 | No |
| 40 | P3 | `992701` | 2026-10-16 | Arch 1 (88.2M) | Isolated sentence LoRMA ($r=8$) | `code/phase3/kaggle-phase3-data.ipynb` | Macro-F1 | 0.3462 | No |
| 41 | P3 | `992698` | 2026-10-16 | Arch 1 (88.2M) | Isolated sentence Full Fine-Tuning | `code/phase3/kaggle-phase3-data.ipynb` | Macro-F1 | 0.3450 | No |
| 42 | P3 | `991240` | 2026-10-16 | Arch 4 (201.4M) | Frozen backbone + linear probe | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.3161 | No |
| 43 | P3 | `991035` | 2026-10-16 | Arch 2 (160.7M) | Isolated sentence encoder baseline | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.3254 | No |
| 44 | P3 | `990963` | 2026-10-16 | Arch 3 (242.5M) | Frozen sliding-window linear probe | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.2776 | No |
| 45 | P3 | `990876` | 2026-10-15 | Arch 4 (201.4M) | Initial Phase-3 transfer smoke test | `code/phase3/arch-4-phase-3-rhetorical-role-prediction.ipynb` | Macro-F1 | 0.3005 | No |
| 46 | P4 | `1010026` | 2026-10-26 | Arch 4 (201.4M) | Compulsory LoRMA 4-Zone ($r=16, \alpha=32$) | `code/phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb` | Macro-F1 | 0.6633 | No |
| 47 | P4 | `1010019` | 2026-10-26 | Arch 4 (201.4M) | Compulsory LoRMA 4-Zone ($\tau=0.50$) | `code/phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb` | Macro-F1 | 0.6618 | No |
| 48 | P4 | `1010013` | 2026-10-26 | Arch 2 (160.7M) | BitFit + Lexical Tail Blend ($\tau=0.50$) | `code/phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb` | Macro-F1 | 0.7093 | No |
| 49 | P4 | `1009979` | 2026-10-26 | Arch 2 (160.7M) | **BitFit (0.32M) + 5-Fold OOF Tail TF-IDF Blend ($\beta=0.40, \tau=0.48$)** | `code/phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb` | Macro-F1 | **0.7106** | **Yes** |
| 50 | P4 | `1008710` | 2026-10-21 | Arch 3 (242.5M) | AdaLoRA ($r_{\text{init}}=12 \to 4$) neural-only | `code/phase4/exp2-arch3-adalora.ipynb` | Macro-F1 | 0.6633 | No |
| 51 | P4 | `1008459` | 2026-10-21 | Arch 3 (242.5M) | AdaLoRA full-corpus + Tail lexical fusion | `code/phase4/phase-4-exp-2-2-arch3-adalora-full-corpus.ipynb` | Macro-F1 | 0.7027 | No |
| 52 | P4 | `1008332` | 2026-10-21 | Arch 3 (242.5M) | AdaLoRA + calibrated threshold ($\tau=0.47$) | `code/phase4/notebook1fa8d9d541.ipynb` | Macro-F1 | 0.7053 | No |
| 53 | P4 | `1006682` | 2026-10-21 | Arch 4 (201.4M) | Single 1,024-token Head truncation baseline | `code/phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb` | Macro-F1 | 0.5573 | No |
| 54 | P4 | `1006441` | 2026-10-21 | Arch 4 (201.4M) | Compulsory LoRMA (`Arch4HierarchicalLoRMANet`) + Tail blend | `code/phase4/p4-experiment-1-arch4-201m-compulsory-lorma.ipynb` | Macro-F1 | 0.6935 | No |
| 55 | P4 | `1006424` | 2026-10-21 | Arch 3 $\to$ Arch 1 | Knowledge Distillation student (88.2M) | `code/phase4/phase4-kd.ipynb` | Macro-F1 | 0.6878 | No |
| 56 | P4 | `1006385` | 2026-10-21 | Arch 2 (160.7M) | Layer-wise partial unfreezing (top-2 layers) | `code/phase4/p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb` | Macro-F1 | 0.6800 | No |
| 57 | P5 | `1043566` | 2026-11-09 | Arch 4 (201.4M) | Dual-View Compulsory LoRMA ($r=32, \alpha=64$) + Tuned $\tau_k$ | `code/phase5/phase5.ipynb` | Macro-F1 | 0.3375 | No |
| 58 | P5 | `1043564` | 2026-11-09 | Arch 4 (201.4M) | Dual-View LoRMA + uniform tuned threshold ($\tau=0.28$) | `code/phase5/phase5.ipynb` | Macro-F1 | 0.2940 | No |
| 59 | P5 | `1043562` | 2026-11-09 | Arch 4 (201.4M) | Dual-View LoRA ($r=32, \alpha=64$) + Tuned $\tau_k$ | `code/phase5/phase5.ipynb` | Macro-F1 | 0.3334 | No |
| 60 | P5 | `1043558` | 2026-11-09 | Arch 4 $\to$ Arch 1 | **Knowledge Distillation (KD) + Per-Statute Thresholds $\tau_k \in [0.10, 0.60]$** | `code/phase5/phase5.ipynb` | Macro-F1 | **0.3392** | **Yes** |
| 61 | P5 | `1043557` | 2026-11-09 | Arch 2 (160.7M) | BitFit Dual-View Encoder + Tuned $\tau_k$ | `code/phase5/phase5.ipynb` | Macro-F1 | 0.3327 | No |
| 62 | P5 | `1043556` | 2026-11-09 | Arch 4 (201.4M) | Dual-View LoRMA with uncalibrated global $\tau=0.50$ | `code/phase5/phase5.ipynb` | Macro-F1 | 0.2639 | No |
| 63 | P6 | `1030574` | 2026-11-03 | Arch 2 (160.7M) | **LoRA ($r=16, \alpha=32$) Seq2Seq + Salient Chunk-Merge + Beam Search ($B=4$)** | `code/phase6/phase6.ipynb` | BERTScore-F1 | **0.8389** | **Yes** |
| 64 | P6 | `1030555` | 2026-11-03 | Arch 2 $\to$ Student | Sequence-level Knowledge Distillation student | `code/phase6/phase6.ipynb` | BERTScore-F1 | 0.8242 | No |
| 65 | P6 | `1030538` | 2026-11-03 | Arch 2 (160.7M) | Token-level KD + greedy decoding | `code/phase6/phase6.ipynb` | BERTScore-F1 | 0.8233 | No |
| 66 | P6 | `1029819` | 2026-11-03 | Arch 4 (201.4M) | Compulsory LoRMA ($r=32, \alpha=64$) causal continuation | `code/phase6/phase6-3.ipynb` | BERTScore-F1 | 0.8086 | No |
| 67 | P6 | `1029814` | 2026-11-03 | Arch 4 (201.4M) | Compulsory LoRMA top-$p$ sampling ($p=0.90$) | `code/phase6/phase6-3.ipynb` | BERTScore-F1 | 0.7713 | No |
| 68 | P6 | `1029807` | 2026-11-03 | Arch 1 (88.2M) | Causal decoder prompt-continuation baseline | `code/phase6/phase6-3.ipynb` | BERTScore-F1 | 0.7612 | No |
| 69 | P6 | `1029803` | 2026-11-03 | Arch 2 (160.7M) | Houlsby Bottleneck Adapters ($r=16$) + Beam 4 | `code/phase6/phase6-2.ipynb` | BERTScore-F1 | 0.8343 | No |
| 70 | P6 | `1029788` | 2026-11-03 | Arch 2 (160.7M) | Houlsby Bottleneck Adapters ($r=16$) + length penalty 1.0 | `code/phase6/phase6-2.ipynb` | BERTScore-F1 | 0.8343 | No |
| 71 | P7 | `1038801` | 2026-11-07 | Arch 4 $\to$ Arch 1 | KD Bi-Encoder + 2-Hop Graph (uncalibrated bg weight) | `code/phase7/phase7-start.ipynb` | Micro-F1@7 | 0.3568 | No |
| 72 | P7 | `1038781` | 2026-11-07 | Arch 4 $\to$ Arch 1 | **Knowledge Distillation (KD) Bi-Encoder + Hybrid BM25/TF-IDF + 2-Hop Graph** | `code/phase7/phase7-start.ipynb` | Micro-F1@7 | **0.3627** | **Yes** |
| 73 | P7 | `1038764` | 2026-11-07 | Arch 4 (201.4M) | Compulsory LoRMA ($r=16, \alpha=32$) Bi-Encoder + BM25 | `code/phase7/phase7-start.ipynb` | Micro-F1@7 | 0.3414 | No |
| 74 | P7 | `1038760` | 2026-11-07 | Arch 4 (201.4M) | LoRMA Bi-Encoder + BM25 + Statutory Regex matchers | `code/phase7/phase7-start.ipynb` | Micro-F1@7 | 0.3455 | No |
| 75 | P7 | `1038754` | 2026-11-07 | Arch 4 (201.4M) | Dense Bi-Encoder with Compulsory LoRMA ($r=16, \alpha=32$) | `code/phase7/phase7-start.ipynb` | Micro-F1@7 | 0.3361 | No |
| 76 | P7 | `1038724` | 2026-11-07 | Arch 4 (201.4M) | LoRMA Bi-Encoder + 1-Hop Citation Graph propagation | `code/phase7/phase7-start.ipynb` | Micro-F1@7 | 0.3491 | No |
| 77 | P7 | `1038668` | 2026-11-07 | Arch 1 (88.2M) | Dense Bi-Encoder MNRL baseline (top-7 cosine) | `code/phase7/phase7-start.ipynb` | Micro-F1@7 | 0.3361 | No |
