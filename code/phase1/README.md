# Phase 1 — Legal Language Model Pretraining

- **Winning Architecture**: Architecture 4 (Hybrid Local-Global GQA Decoder, 201.44M parameters)
- **Official Score**: `0.8322` BERTScore-F1 (Submission `964323`, Rank 1)

## Notebooks
- `cs781-phase1-data.ipynb`: Streams Central/State Acts (18,826), Supreme Court judgments (55,549), and 10% sampled Tribunal orders into deterministic 98/2 splits.
- `phase1-tokens-v40k-clean.ipynb`: Filters OCR control noise and strips ~1% Hindi (Devanagari) text, trains the 40,000-token Byte-Level BPE tokenizer (`tokenizer.json`), and writes packed `uint16` `.bin` shards.
- `architecture-4-hybrid-local-global-grouped-query.ipynb`: Defines Architecture 4 (12 layers, $d_{\text{model}}=1024$, 16Q/8KV GQA, RoPE, SwiGLU, `L-L-G` local-global attention schedule).
- `train-arch4-t4-v4-1.ipynb`: Primary 10,000-step pretraining run on a single 16 GB NVIDIA T4 GPU (163.84M tokens).
- `train-arch4-t4-v4-1_genarate arch2.ipynb`, `train-arch4-t4-v4-1_genarate arch.ipynb`, `arch4-1-gen.ipynb`, `arch4ka2part.ipynb`: KV-cached generation (`FastGen`) with dynamic 4-gram blocking and repetition penalty ($\theta_{\text{rep}}=1.15$).
- `arch1_model.ipynb`, `arch1_genarate.ipynb`: Comparative Architecture 1 (88.21M Standard Causal GPT Decoder).
- `arch2-model.ipynb`, `arch2 FULL ENCODER-DECODER TRANSFORMER .ipynb`, `arch2-preln-eval-generate.ipynb`: Comparative Architecture 2 (160.68M Pre-LN Encoder-Decoder).
- `arach3-train.ipynb`, `arach3-inference.ipynb`: Comparative Architecture 3 (242.54M Pure Sliding-Window Decoder).
