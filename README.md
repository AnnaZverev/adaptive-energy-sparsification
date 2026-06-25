# Adaptive Energy Sparsification (AES)

Deterministic information-theoretic regularization for hybrid neural networks in low-data regimes.

## Overview

This repository contains the implementation of **Adaptive Energy Sparsification (AES)** — a deterministic entropy-based regularization method for training deep spatio-temporal neural networks under limited data conditions.

The method introduces an adaptive constraint on the entropy of hidden representations, controlling the information capacity of deep layers without relying on stochastic sampling or variational approximations.

AES is motivated by the Information Bottleneck principle but avoids the instability of KL-based formulations by directly constraining the entropy of latent activations.

## Key Idea

Instead of penalizing weights or activations uniformly, AES enforces a **relative entropy constraint**:

- Early layer → reference information level
- Deep layers → compressed representation

The model is penalized only when the entropy of deep representations exceeds a dynamically defined threshold (a one-sided barrier). This realizes a **deterministic information bottleneck** without stochastic encoding.

## Features

- Deterministic alternative to the Variational Information Bottleneck (VIB)
- Integrates into standard hybrid architectures (3D-CNN / ResNet + LSTM)
- Designed for **small datasets**
- No KL divergence, no sampling
- Single penalty term, easy to drop into existing training pipelines

## Repository structure
.

├── adaptive_energy_sparsification_v3_1.ipynb   # main notebook: comparison, ablation, cross-validation

├── README.md

└── results/                                     # CSV tables produced by the notebook (optional)

├── table1_comparison.csv

├── table2_ablation.csv

└── table3_cv.csv

## How to run

The experiments are provided as a self-contained Colab notebook.

1. Open `adaptive_energy_sparsification_v3_1.ipynb` in Google Colab and select a GPU runtime (A100 recommended; T4 works with `batch_size = 2`).
2. Run the cells top to bottom. The notebook will:
   - download the ABIDE subset via `nilearn` (C-PAC pipeline);
   - cache decompressed volumes once to `.pt` tensors (subsequent epochs read from cache);
   - train the baseline and all regularizers, then run ablation and 5-fold cross-validation.
3. Results are written incrementally to `results.json` on Google Drive, so a runtime reset resumes from where it stopped instead of restarting. Set `USE_DRIVE = False` to keep results local.
4. Summary tables are exported as `table1_comparison.csv`, `table2_ablation.csv`, `table3_cv.csv`.

### Requirements

Installed automatically in the notebook:

- `torch`, `torchvision`
- `nilearn`, `nibabel`
- `scikit-learn`, `pandas`, `numpy`

## Experiment (fMRI / ABIDE)

- Dataset: ABIDE (150 subjects, multi-site)
- Preprocessing: C-PAC pipeline
- Model: 3D ResNet-18 + LSTM
- Sequence length: 30
- Optimizer: AdamW
- Penalty applied at layer3 (selected via sensitivity analysis)

### Results — fixed 80/20 split

| Model | Accuracy | ROC-AUC | F1 | Entropy (bits) |
|------|--------:|--------:|----:|--------------:|
| Baseline | 66.7% | 0.738 | 0.643 | 10.17 |
| L2 (weight decay) | 70.0% | 0.762 | 0.667 | 10.36 |
| Dropout | 63.3% | 0.762 | 0.667 | 9.71 |
| L1 (activations) | 66.7% | 0.760 | 0.545 | 10.97 |
| VIB | 70.0% | 0.724 | 0.667 | 11.47 |
| **AES** | **70.0%** | 0.711 | **0.710** | **0.27** |

### Results — 5-fold cross-validation (mean ± std)

| Model | Accuracy | ROC-AUC | F1 | Entropy (bits) |
|------|:-------:|:-------:|:--:|:-------------:|
| Baseline | 0.660 ± 0.053 | 0.701 ± 0.056 | 0.655 ± 0.028 | 6.26 ± 2.16 |
| VIB | 0.673 ± 0.053 | 0.743 ± 0.068 | 0.630 ± 0.085 | 9.19 ± 1.50 |
| **AES (layer3)** | **0.680 ± 0.040** | 0.679 ± 0.054 | **0.673 ± 0.093** | **0.45 ± 0.13** |

**Summary.** AES reduces the entropy of the latent representation by roughly an order of magnitude while keeping accuracy and F1 on par with the baseline (within one standard deviation). ROC-AUC is slightly lower than the baseline and VIB, reflecting a compression–ranking trade-off: AES optimizes representation compactness rather than probabilistic ranking.
