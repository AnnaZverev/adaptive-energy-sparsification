# Adaptive Energy Sparsification (AES)

Deterministic information-theoretic regularization for hybrid neural networks in low-data regimes.

## Overview

This repository contains the implementation of **Adaptive Energy Sparsification (AES)** - a deterministic entropy-based regularization method designed for training deep spatio-temporal neural networks under limited data conditions.

The method introduces an adaptive constraint on the entropy of hidden representations, effectively controlling the information capacity of deep layers without relying on stochastic sampling or variational approximations.

AES is motivated by the Information Bottleneck principle but avoids the instability of KL-based formulations by directly minimizing the marginal entropy of latent representations.

## Key Idea

Instead of penalizing weights or activations uniformly, AES enforces a **relative entropy constraint**:

- Early layer → reference information level  
- Deep layers → compressed representation  

The model is penalized only when the entropy of deep representations exceeds a dynamically defined threshold.

This creates a **deterministic information bottleneck** without stochastic encoding.

## Features

- Deterministic alternative to Variational Information Bottleneck (VIB)
- Works with standard architectures (CNN, ResNet, LSTM, Transformers)
- Designed for **small datasets**
- No KL divergence, no sampling
- Simple integration into existing training pipelines


## Experiment (fMRI / ABIDE)

- Dataset: ABIDE (150 subjects)
- Preprocessing: C-PAC pipeline
- Model: 3D ResNet-18 + LSTM
- Sequence length: 30
- Optimizer: AdamW

### Results

| Model | Accuracy | ROC-AUC | Entropy (bits) | Reduction |
|------|--------:|--------:|--------------:|----------:|
| Baseline | 63.3% | 0.649 | 5.23 | — |
| AES | 70.0% | 0.764 | 0.12 | 97.7% |

AES improves both classification performance and representation compactness.
