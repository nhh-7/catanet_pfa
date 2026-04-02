# Lightweight Image Super-Resolution Experimental Data (PSNR/SSIM)

This document compiles the quantitative experimental results of the CATANet and PFT models from their original papers on the lightweight image super-resolution task.

## 1. Citation Information

**CATANet:**
> [1] Zhou X, Li W, Li C, et al. Content-aware token aggregation network for lightweight image super-resolution[J]. arXiv preprint arXiv:2403.01166, 2024. (Accepted to CVPR 2024)

**PFT (Progressive Focused Transformer):**
> [2] Long W, Zhou X, Zhang L, et al. Progressive Focused Transformer for Single Image Super-Resolution[J]. arXiv preprint arXiv:2503.20337, 2025. (Accepted to CVPR 2025)

---

## 2. Experimental Data (Scale Factors x2, x3, x4)

The evaluation metrics are PSNR and SSIM on standard benchmarks: Set5, Set14, B100 (BSD100), Urban100, and Manga109.

### Scale Factor x2

| Method | Params | Set5 (PSNR/SSIM) | Set14 (PSNR/SSIM) | B100 (PSNR/SSIM) | Urban100 (PSNR/SSIM) | Manga109 (PSNR/SSIM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CATANet** [1] | 477K | 38.28 / 0.9617 | 33.99 / 0.9217 | 32.37 / 0.9023 | 33.09 / 0.9372 | 39.37 / 0.9784 |
| **CATANet**† [1] | 477K | 38.35 / 0.9620 | 34.11 / 0.9229 | 32.41 / 0.9027 | 33.33 / 0.9387 | 39.57 / 0.9788 |
| **PFT-light** [2] | 776K | 38.36 / 0.9620 | 34.19 / 0.9232 | 32.43 / 0.9030 | 33.67 / 0.9411 | 39.55 / 0.9792 |

### Scale Factor x3

| Method | Params | Set5 (PSNR/SSIM) | Set14 (PSNR/SSIM) | B100 (PSNR/SSIM) | Urban100 (PSNR/SSIM) | Manga109 (PSNR/SSIM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CATANet** [1] | 550K | 34.75 / 0.9300 | 30.67 / 0.8481 | 29.28 / 0.8101 | 29.04 / 0.8689 | 34.40 / 0.9499 |
| **CATANet**† [1] | 550K | 34.83 / 0.9307 | 30.73 / 0.8490 | 29.34 / 0.8111 | 29.24 / 0.8718 | 34.69 / 0.9512 |
| **PFT-light** [2] | 783K | 34.81 / 0.9305 | 30.75 / 0.8493 | 29.33 / 0.8116 | 29.43 / 0.8759 | 34.60 / 0.9510 |

### Scale Factor x4

| Method | Params | Set5 (PSNR/SSIM) | Set14 (PSNR/SSIM) | B100 (PSNR/SSIM) | Urban100 (PSNR/SSIM) | Manga109 (PSNR/SSIM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CATANet** [1] | 535K | 32.58 / 0.8998 | 28.90 / 0.7880 | 27.75 / 0.7427 | 26.87 / 0.8081 | 31.31 / 0.9183 |
| **CATANet**† [1] | 535K | 32.68 / 0.9009 | 28.98 / 0.7894 | 27.80 / 0.7437 | 27.04 / 0.8113 | 31.58 / 0.9206 |
| **PFT-light** [2] | 792K | 32.63 / 0.9005 | 28.92 / 0.7891 | 27.79 / 0.7445 | 27.20 / 0.8171 | 31.51 / 0.9204 |

*(Note: † indicates that self-ensemble strategy is used in testing)*

---

## 3. Computational Complexity Data

| Method | Scale | Params | Multi-Adds (FLOPs) | Test Image Size |
| :--- | :---: | :---: | :---: | :---: |
| **CATANet-L** | x4 | 535K | 46.8G | 3 x 256 x 256 |
| **PFT-light** | x4 | 792K | 69.6G | (Standard FLOPs eval) |

*(Note: CATANet provides FLOPs based on $3 \times 256 \times 256$ input size, while PFT's exact input size for FLOPs calculation might follow the standard $1280 \times 720$ output convention, but PFT-light's reported 69.6G FLOPs shows it is a lightweight model.)*