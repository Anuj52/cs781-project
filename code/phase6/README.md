# Phase 6 — Legal Abstractive Summarization

- **Best-Performing Method**: **LoRA** ($r=16, \alpha=32$) on Architecture 2 Pre-LN Seq2Seq (160.68M) + Length-Penalized Beam Search ($B=4$)
- **Official Score**: `0.8389` BERTScore-F1 (Submission `1030574`, Rank 1)

## Notebooks
- `phase6.ipynb`: Winning **LoRA** ($r=16, \alpha=32$) Seq2Seq adaptation, mandatory Knowledge Distillation, and length-penalized beam search ($B=4$, length penalty $1.0$, repetition penalty $1.25$).
- `phase6-2.ipynb`: Arch 2 Pre-LN Encoder-Decoder with Houlsby bottleneck adapters ($r=16$) and salient chunk-and-merge cross-attention encoding.
- `phase6-3.ipynb`: Compulsory **LoRMA** ($r=32, \alpha=64$) causal prompt-continuation summarization on Arch 4.
