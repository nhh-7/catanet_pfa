# PFA-CATANet (Ours) 最终实验结果记录表

本文档用于记录 PFA-CATANet 模型训练及测试完成后的所有实验数据。表格已基于实验组设计方案及参考的 SOTA (CATANet / PFT-light) 数据预先构建，您只需在 `[TODO]` 占位符处填入 PFA-CATANet 的实际跑分数据即可。

***

## 1. Quantitative Comparison (与 SOTA 的定量对比)

**设置**: 训练集为 DIV2K，在 Y 通道上计算 PSNR/SSIM。

### 1.1 Scale Factor $\times 2$

| Method                 | Params |         Set5        |        Set14        |         B100        |       Urban100      |       Manga109      |
| :--------------------- | :----: | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: |
| CARN                   |  1592K |    37.76 / 0.9590   |    33.52 / 0.9166   |    32.09 / 0.8978   |    31.92 / 0.9256   |    38.36 / 0.9765   |
| ESRT                   |  677K  |    38.03 / 0.9600   |    33.75 / 0.9184   |    32.25 / 0.9001   |    32.58 / 0.9318   |    39.12 / 0.9774   |
| SwinIR-light           |  878K  |    38.14 / 0.9611   |    33.86 / 0.9206   |    32.31 / 0.9012   |    32.76 / 0.9340   |    39.12 / 0.9783   |
| CATANet                |  477K  |    38.28 / 0.9617   |    33.99 / 0.9217   |    32.37 / 0.9023   |    33.09 / 0.9372   |    39.37 / 0.9784   |
| PFT-light              |  776K  |    38.36 / 0.9620   |    34.19 / 0.9232   |    32.43 / 0.9030   |    33.67 / 0.9411   |    39.55 / 0.9792   |
| **PFA-CATANet (Ours)** |  477K  | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` |

### 1.2 Scale Factor $\times 3$

| Method                 |  Params  |         Set5        |        Set14        |         B100        |       Urban100      |       Manga109      |
| :--------------------- | :------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: |
| CARN                   |   1592K  |    34.29 / 0.9255   |    30.29 / 0.8407   |    29.06 / 0.8034   |    28.06 / 0.8493   |    33.43 / 0.9427   |
| ESRT                   |   770K   |    34.42 / 0.9268   |    30.43 / 0.8433   |    29.15 / 0.8063   |    28.46 / 0.8574   |    33.95 / 0.9455   |
| SwinIR-light           |   886K   |    34.62 / 0.9289   |    30.54 / 0.8463   |    29.20 / 0.8082   |    28.66 / 0.8624   |    33.98 / 0.9478   |
| CATANet                |   550K   |    34.75 / 0.9300   |    30.67 / 0.8481   |    29.28 / 0.8101   |    29.04 / 0.8689   |    34.40 / 0.9499   |
| PFT-light              |   783K   |    34.81 / 0.9305   |    30.75 / 0.8493   |    29.33 / 0.8116   |    29.43 / 0.8759   |    34.60 / 0.9510   |
| **PFA-CATANet (Ours)** | `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` |

### 1.3 Scale Factor $\times 4$

| Method                 |  Params  |         Set5        |        Set14        |         B100        |       Urban100      |       Manga109      |
| :--------------------- | :------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: |
| CARN                   |   1592K  |    32.13 / 0.8937   |    28.60 / 0.7806   |    27.58 / 0.7349   |    26.07 / 0.7837   |    30.42 / 0.9070   |
| ESRT                   |   751K   |    32.19 / 0.8947   |    28.69 / 0.7833   |    27.69 / 0.7379   |    26.39 / 0.7962   |    30.75 / 0.9100   |
| SwinIR-light           |   897K   |    32.44 / 0.8976   |    28.77 / 0.7858   |    27.69 / 0.7406   |    26.47 / 0.7980   |    30.92 / 0.9151   |
| CATANet                |   535K   |    32.58 / 0.8998   |    28.90 / 0.7880   |    27.75 / 0.7427   |    26.87 / 0.8081   |    31.31 / 0.9183   |
| PFT-light              |   792K   |    32.63 / 0.9005   |    28.92 / 0.7891   |    27.79 / 0.7445   |    27.20 / 0.8171   |    31.51 / 0.9204   |
| **PFA-CATANet (Ours)** | `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` | `[TODO]` / `[TODO]` |

***

## 2. Ablation Studies (消融实验)

**设置**: 训练集为 DIV2K，放大倍率 $\times 4$，迭代 250K 步 (快速评估设置)。

> *(注意: 如果您决定按照 CATANet 跑全量 500K DF2K 训练，请更新说明)*

### 2.1 核心组件的有效性验证 (Effectiveness of Core Components)

此实验旨在拆解 `Cluster-Isolation Mask` (隔离掩码) 和 `Progressive Attention` (注意力继承) 的独立与联合贡献。

| Model | Components                    |  Params  |   FLOPs  |   Set5   | Urban100 | Manga109 |
| :---- | :---------------------------- | :------: | :------: | :------: | :------: | :------: |
| A     | Baseline (原版 CATANet)         |   535K   |   46.8G  |   32.58  |   26.87  |   31.31  |
| B     | + Cluster-Isolation Mask      | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| C     | + PFA 继承 (无 Mask)             | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| **D** | **PFA-CATANet (Ours, A+B+C)** | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

### 2.2 Focus Ratio ($\alpha$) 超参数分析 (Impact of Focus Ratio)

探讨公式 $fused\_probs = \alpha \cdot prev\_attn + (1 - \alpha) \cdot current\_attn$ 中 $\alpha$ 取值的影响。

| Focus Ratio ($\alpha$) | 0.0 (No Inheritance) |   0.25   |    0.5   |   0.75   | 1.0 (Full Inheritance) |
| :--------------------- | :------------------: | :------: | :------: | :------: | :--------------------: |
| **PSNR (Urban100)**    |       `[TODO]`       | `[TODO]` | `[TODO]` | `[TODO]` |        `[TODO]`        |
| **PSNR (Manga109)**    |       `[TODO]`       | `[TODO]` | `[TODO]` | `[TODO]` |        `[TODO]`        |

### 2.3 注意力继承的跨度策略 (Attention Inheritance Strategy)

对比局部继承与全局渐进式聚焦的收益。

| Strategy                                         |   Set5   | Urban100 | Manga109 |
| :----------------------------------------------- | :------: | :------: | :------: |
| Strategy 1: Intra-Block (仅在 Block 内继承)           | `[TODO]` | `[TODO]` | `[TODO]` |
| **Strategy 2: Inter-Block (跨 Block 全局继承, Ours)** | `[TODO]` | `[TODO]` | `[TODO]` |

***

## 3. Computational Complexity (计算复杂度与推理延迟)

**设置**: 评估输入尺寸为 $3 \times 256 \times 256$，以公平对比 CATANet。Latency 使用同一张显卡测试。

| Model                  |  Params  | Multi-Adds (FLOPs) | Latency (ms) |
| :--------------------- | :------: | :----------------: | :----------: |
| SwinIR-light           |   897K   |        60.3G       |     158.1    |
| CATANet-L (Baseline)   |   535K   |        46.8G       |     86.0     |
| PFT-light              |   792K   |        69.6G       |   `[TODO]`   |
| **PFA-CATANet (Ours)** | `[TODO]` |      `[TODO]`      |   `[TODO]`   |

