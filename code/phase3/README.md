# Phase 3 — Rhetorical Role Prediction

- **Best-Performing Method**: **LoRMA** ($r=8, \alpha=16$) + Predecessor Context Conditioning ($s_{t-1} \to s_t$) + Class-Weighted Focal Loss ($\gamma=2.0$)
- **Official Score**: `0.4807` Macro-F1 (Submission `994294`, Rank 1)

## Notebooks
- `arch-4-phase-3-rhetorical-role-prediction.ipynb`: Compulsory **LoRMA** head adaptation, Gated Context Classifier ($s_{t-1}, s_t$), and mandatory Knowledge Distillation across Corporate Law (CL) and Income Tax (IT) judgments.
- `kaggle-phase3-data.ipynb`: Predecessor-conditioned training and multi-domain Corporate Law + Income Tax transfer pipeline.
