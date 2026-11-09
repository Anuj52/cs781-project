# Phase 7 — Prior Case Retrieval (PCR)

- **Best-Performing Method**: **Knowledge Distillation (KD)** (Arch 4 Bi-Encoder Teacher $\to$ Arch 1 Student `Phase1BiEncoder`) + Hybrid BM25/TF-IDF Fusion + 2-Hop Citation Graph Diffusion
- **Official Score**: `0.3627` Micro-F1@7 (Submission `1038781`, Rank 1)

## Notebooks
- `phase7-start.ipynb`: Complete symmetric bi-encoder contrastive training (Multiple Negatives Ranking Loss), mandatory **Knowledge Distillation (KD)** into Arch 1 student, compulsory **LoRMA** ($r=16, \alpha=32$), sparse BM25/TF-IDF and regex statutory overlap features, and 2-hop citation graph propagation.
