# Ablation Study Experimental Data (CATANet & PFT)

This document compiles the quantitative results from the ablation studies conducted in the CATANet (CVPR 2024) and PFT (CVPR 2025) original papers. These results demonstrate the effectiveness of their respective core components.

---

## 1. CATANet Ablation Studies [1]

All experiments for CATANet ablation studies were conducted under the $\times 4$ scale setting unless otherwise specified. Multi-Adds (FLOPs) were calculated based on a $3 \times 256 \times 256$ input size.

### Table 1: Attending to Consecutive Subgroups ($\times 4$)
This validates the overlap-window mechanism allowing Query to attend to adjacent Subgroups.

| Method | Params | Multi-Adds | Urban100 | Manga109 |
| :--- | :---: | :---: | :---: | :---: |
| Not Attend | 536K | 46.8G | 26.85 | 31.26 |
| **Attend (ours)** | 536K | 46.8G | **26.87** | **31.31** |

### Table 3: Effectiveness of IASA and IRCA ($\times 4$)
This validates the Intra-Group Self-Attention (IASA) and Inter-Group Cross-Attention (IRCA) branches inside the TAB module.

| Configuration | Params | Multi-Adds | Set5 | Set14 | B100 | Urban100 | Manga109 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (No TAB) | 366K | 37.3G | 32.26 | 28.63 | 27.68 | 26.46 | 30.81 |
| + IASA | 511K | 46.8G | 32.47 | 28.75 | 27.75 | 26.85 | 31.24 |
| **+ IASA + IRCA (ours)** | 535K | 46.8G | **32.58** | **28.90** | **27.75** | **26.87** | **31.31** |

### Table 4: Different Designs of Token Aggregation ($\times 2$)
Comparison of different clustering/grouping mechanisms.

| Method | Set5 | Set14 | B100 | Urban100 | Manga109 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Clustered Attention | 32.25 | 33.84 | 32.33 | 32.96 | 39.30 |
| TCformer | 38.06 | 33.87 | 32.32 | 32.90 | 39.17 |
| NLSA | 37.67 | 33.29 | 31.96 | 31.22 | 37.78 |
| **CATANet (ours)** | **38.28** | **33.99** | **32.37** | **33.09** | **39.37** |

### Table 5: Fusion Approach of IASA and IRCA ($\times 4$)
Comparison of feature fusion strategies.

| Method | Params | Multi-Adds | Set5 | Urban100 | Manga109 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Concat | 548K | 47.6G | 32.49 | 26.82 | 31.28 |
| **Add (ours)** | 535K | 46.8G | **32.58** | **26.87** | **31.31** |

---

## 2. PFT (Progressive Focused Transformer) Ablation Studies [2]

All experiments for PFT ablation studies were conducted on the PFT-light model for 250k iterations on the DIV2K dataset at the $\times 4$ scale.

### Table 3: Effectiveness of the Proposed Focused Attention ($\times 4$)
This compares standard self-attention, Top-k sparse attention, Progressive Attention (Hadamard product connection), and Progressive Focused Attention (SMM + inheritance).

| Method | FLOPs | Set5 | Urban100 | Manga109 |
| :--- | :---: | :---: | :---: | :---: |
| Vanilla Self-Attention | 70.4G | 32.28 | 26.26 | 30.62 |
| Top-k Attention | 70.4G | 32.30 | 26.29 | 30.67 |
| Progressive Attention | 70.4G | 32.35 | 26.41 | 30.78 |
| **Progressive Focused Attention (Ours)**| **50.9G** | **32.41** | **26.62** | **30.85** |

*(Note: PFA achieves the highest PSNR while reducing computation by 27.69% compared to Vanilla Self-Attention.)*

### Table 4: Impact of Focus Ratio $\alpha$ ($\times 4$)
Evaluated on the Urban100 dataset. $\alpha$ controls the focus degree as network depth increases.

| Focus ratio ($\alpha$) | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PSNR** | 26.48 | 26.58 | **26.62** | 26.61 | 26.56 |

*(Note: Model performance peaks when $\alpha$ is around 0.5.)*

### Table 5: Impact of Different Window Sizes ($\times 4$)
Evaluated on the Urban100 dataset. PFA dynamically filters irrelevant patches, allowing the use of large windows efficiently.

| Window size | Set5 | Urban100 | Manga109 |
| :--- | :---: | :---: | :---: |
| $8 \times 8$ | 32.33 | 26.40 | 30.71 |
| $16 \times 16$ | 32.41 | 26.62 | 30.85 |
| **$32 \times 32$** | **32.49** | **26.81** | **30.93** |

---

## 3. Citation References
> [1] Zhou X, Li W, Li C, et al. Content-aware token aggregation network for lightweight image super-resolution[J]. arXiv preprint arXiv:2403.01166, 2024.
> [2] Long W, Zhou X, Zhang L, et al. Progressive Focused Transformer for Single Image Super-Resolution[J]. arXiv preprint arXiv:2503.20337, 2025.