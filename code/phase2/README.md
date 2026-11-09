# Phase 2 — Legal Named Entity Recognition (L-NER)

- **Best-Performing Method**: **LoRMA** ($r=16, \alpha=32$) on Architecture 4 (201.44M) + 25-tag BIO Token Classification Head + `correct_offsets()`
- **Official Score**: `0.5429` Strict Macro-F1 (Submission `978703`, Rank 1)

## Notebooks
- `phase2-experiments.ipynb`: Compulsory **LoRMA** (`LORMALinear`), LoRA, BitFit, and leading-space offset correction (`correct_offsets()`).
- `phase2-arch4-ner.ipynb`: Arch 4 BIO sequence labeling, class-weighted focal loss, and overlapping window inference (1,024 window / 768 stride).
- `phase2-experiments2.ipynb`: Mandatory Knowledge Distillation (12-layer Arch 4 teacher $\to$ 6-layer student `[0,1,2,6,7,8]` preserving `L-L-G` schedule).
- `full-fine-tuning-for-phase-2-legal-ner-architect.ipynb`: Full fine-tuning baseline ablation.
