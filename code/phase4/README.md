# Phase 4 — Court Judgment Prediction (CJPE)

- **Best-Performing Method**: **BitFit** (Bias-Only Tuning, 0.32M params) on 4-Zone Arch 2 Encoder + Out-of-Fold Tail TF-IDF Ridge/Logistic Ensemble ($0.40 \times \text{Neural} + 0.60 \times \text{Lexical}$)
- **Official Score**: `0.7106` Macro-F1 (Submission `1009979`, Rank 1)

## Notebooks
- `p4-ex-3-arch2-bitfit-layer-wise-partial.ipynb`: Winning **BitFit** + 4-Zone document slicing (Head, Mid-1, Mid-2, Tail) + Tail word/char TF-IDF ensemble and validation threshold search ($\tau=0.48$).
- `p4-experiment-1-arch4-201m-compulsory-lorma.ipynb`: Compulsory **LoRMA** (`Arch4HierarchicalLoRMANet`) on 4-zone representations.
- `exp2-arch3-adalora.ipynb`, `phase-4-exp-2-2-arch3-adalora-full-corpus.ipynb`, `notebook1fa8d9d541.ipynb`: Arch 3 (242.5M) AdaLoRA SVD rank-allocation experiments.
- `phase4-kd.ipynb`: Mandatory Knowledge Distillation (242.5M Arch 3 AdaLoRA teacher $\to$ 88.2M Arch 1 student).
