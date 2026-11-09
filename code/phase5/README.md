# Phase 5 — Legal Statute Identification (LSI)

- **Best-Performing Method**: **Knowledge Distillation (KD)** (Arch 4 Dual-View Teacher $\to$ Arch 1 Student) + Per-Statute Threshold Tuning ($\tau_k \in [0.10, 0.60]$)
- **Official Score**: `0.3392` Macro-F1 (Submission `1043558`, Rank 1)

## Notebooks
- `phase5.ipynb`: Complete Dual-View encoding (first 512 + last 512 tokens), mandatory **Knowledge Distillation (KD)**, compulsory **LoRMA** ($r=32, \alpha=64$), and per-statute coordinate threshold search on 10,181 validation cases.
