# PFA-CATANet 超分辨率新算法论文实验设计方案

将融合了 **Progressive Focused Attention (PFA)** 与 **CATANet** 聚类机制的新模型视为一个全新的轻量级图像超分辨率（Lightweight Image Super-Resolution）算法。为了满足顶级会议/期刊（如 CVPR/ICCV/ECCV/TPAMI）的发表要求，本方案参考了 CATANet、PFT 等最新论文的标准实验范式，设计了详尽的实验论证逻辑。

本实验设计的核心目标是证明两点：
1. **Cluster-Isolation Mask** 成功解决了由于物理连续切分（Subgroup划分）导致的跨聚类噪声（Cross-Cluster Noise）问题。
2. **PFA 渐进式聚焦机制** 在深层网络中有效剔除了冗余相似度计算，提升了高频细节重建能力，且维持或降低了计算复杂度。

---

## 1. 实验设置 (Experimental Settings)

### 1.1 数据集 (Datasets)
*   **训练集 (Training Set)**: 使用标准的 **DF2K** 数据集（包含 DIV2K 的 800 张图像和 Flickr2K 的 2650 张图像）。
*   **测试集 (Test Sets)**: 使用 5 个标准基准测试集：**Set5, Set14, BSD100, Urban100, Manga109**。

### 1.2 评价指标 (Evaluation Metrics)
*   **定量指标**: **PSNR** 和 **SSIM**。为了保证公平对比，所有指标均在 YCbCr 颜色空间的 **Y 通道 (亮度通道)** 上计算。
*   **效率指标**: 参数量 (**Params**)、计算复杂度 (**FLOPs**，基于 $1280 \times 720$ 输出分辨率或 $256 \times 256$ 输入评估)、推理延迟 (**Latency** / Throughput，基于单张 RTX 3090/4090 或 V100 GPU 测算)。

### 1.3 训练细节 (Implementation Details)
*   **输入尺寸**: LR 图像裁剪为 $64 \times 64$ 的 Patch。
*   **Batch Size**: 32 或 64 (根据显存调整，保持与对比基线一致)。
*   **优化器**: Adam ($ \beta_1=0.9, \beta_2=0.999 $) 或 AdamW。
*   **学习率**: 初始学习率设为 $5 \times 10^{-4}$，采用 Cosine Annealing 或 MultiStepLR 衰减策略，总迭代次数 500K 步。
*   **数据增强**: 随机水平/垂直翻转、随机 $90^\circ$ 旋转。

---

## 2. 主实验：与最先进方法的对比 (Comparison with SOTA)

本节旨在确立 PFA-CATANet 在轻量级超分领域的 SOTA 地位。

### 2.1 定量对比 (Quantitative Comparison)
*   **对比方法**:
    *   *经典轻量级 CNN*: RCAN, CARN, IMDN, LAPAR, SAFMN 等。
    *   *Transformer 架构*: SwinIR-light, ESRT, ELAN, HAT-S, CATANet (原版基线), PFT-light (PFA原出处)。
*   **实验设置**: 在 $\times 2, \times 3, \times 4$ 放大倍率下，全面报告 PSNR/SSIM 结果。
*   **预期结果**: 在 Params 和 FLOPs 相似或更低的前提下，PFA-CATANet 在高频细节丰富的测试集（如 Urban100 和 Manga109）上取得最优（SOTA）的 PSNR/SSIM。

### 2.2 定性可视化对比 (Visual Comparison)
*   选取 Urban100 (丰富线条/几何结构) 和 Manga109 (丰富文本/纹理) 中的挑战性图像 (如 `img_004`, `img_092`, `YumeiroCooking`)。
*   提供局部放大的 Visual 结果，并附带 Error Map 差异图。
*   **预期结论**: 相比原版 CATANet 容易在边缘产生的模糊或伪影，PFA-CATANet 能恢复更锐利、无结构扭曲的边缘，证明跨聚类噪声被有效抑制，且 PFA 成功聚焦了相关纹理。

---

## 3. 核心消融实验 (Ablation Studies)

这是论文最关键的部分，用于证明我们提出的两大创新点的绝对有效性。所有消融实验建议在 **$\times 4$ 倍率，仅使用 DIV2K 训练 250K 步**的快速设置下进行验证（以节省算力）。

### 3.1 核心组件的有效性验证 (Effectiveness of Core Components)
设计四组模型进行逐步递进对比：
*   **Model A (Baseline)**: 原版 CATANet（不带 Mask，不带 PFA）。
*   **Model B (+ Cluster-Isolation Mask)**: 原版 CATANet 仅增加聚类隔离掩码，阻断强制 Subgroup 带来的不同聚类间的注意力计算。
*   **Model C (+ PFA 继承)**: 原版 CATANet 仅增加 Attention 状态跨层继承，不使用隔离掩码。
*   **Model D (Ours, + Mask + PFA)**: 完整的 PFA-CATANet。
*   **分析**: 预期 B 和 C 均优于 A，证明两个机制各自的有效性；D 取得最佳性能，证明物理隔离噪声与注意力聚焦在机制上的互补性。

### 3.2 Focus Ratio ($\alpha$) 超参数分析 (Impact of Focus Ratio)
PFA 中的公式 $fused\_probs = \alpha \cdot prev\_attn + (1 - \alpha) \cdot current\_attn$。
*   **实验组**: $\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$。
*   **分析**: 当 $\alpha=0$ 时等同于不继承；当 $\alpha=1.0$ 时深层完全放弃学习新特征。预期在 $\alpha = 0.5$ 或 $0.25$ 处取得峰值，说明“渐进式指导”能最大化收益。

### 3.3 注意力继承的跨度策略 (Attention Inheritance Strategy)
*   **Strategy 1 (Intra-Block)**: 仅在同一个 Block 的不同层之间传递 Attention Map。
*   **Strategy 2 (Inter-Block)**: 全局连续传递（当前代码实现的版本，从浅层一直传到最深层）。
*   **分析**: 证明长程的渐进式聚焦 (Progressive Focus) 比局部的注意力继承能更好地捕获图像的深层语义。

### 3.4 计算开销与稀疏性评估 (Computational Overhead & Sparsity)
*   如果我们的 Mask 在底层结合了稀疏矩阵乘法（SMM），需要对比稠密计算（Dense）与稀疏计算的 FLOPs 差异与实际推理速度 (FPS)。
*   即使不使用 SMM，也需要证明引入的 Mask 和 $\alpha$ 融合仅带来了极少量的运算（通常可忽略不计），但带来了显著的 PSNR 提升。

---

## 4. 模型分析与可视化 (Model Analysis and Visualization)

这一部分旨在深入“黑盒”，直观解释为什么新算法更好。

### 4.1 Attention Map 演变可视化 (Visualization of Progressive Focused Attention)
*   选取一张输入图像中的一个特定 Query Token（例如位于边缘的 Token）。
*   **可视化目标**: 提取浅层、中层、深层网络的 Attention Map（即 $fused\_probs$）并将其投射回原图。
*   **预期现象**: 在浅层，注意力可能比较分散；随着层数加深，借助 PFA 的继承机制和 Mask 的隔离机制，深层的注意力分布**极其锐利**地聚焦在与该 Query Token 纹理结构相似的区域，完全排除了背景噪声（相比原版 CATANet 散乱的注意力分布）。

### 4.2 聚类隔离效果可视化 (Effectiveness of Cluster-Isolation Mask)
*   将原版 CATANet 的 Attention Logits 与带有 Mask 后的 Attention Logits 进行对比热力图展示。
*   **分析**: 证明原版 CATANet 在 Subgroup 的边界处计算了大量毫无意义的、跨语义区域的相似度（即噪声），而我们的 Mask 实现了完美的物理切割。

### 4.3 性能-参数权衡图 (Performance vs. Complexity Trade-off)
*   绘制两张散点图：**PSNR vs. Parameters** 和 **PSNR vs. FLOPs**（在 Urban100 $\times 4$ 上）。
*   将 PFA-CATANet 与 SwinIR-light, HAT-S, PFT-light, 原版 CATANet 等模型标在图上。
*   **预期结果**: 我们的模型应处于图表的左上角（参数/FLOPs最少，PSNR最高），展现出一条陡峭的帕累托前沿 (Pareto Front)。

---

## 5. 论文 Storyline 与写作逻辑建议 (Storyline for Paper Writing)

*   **Motivation (动机)**: 近期的轻量级 SR 倾向于使用 Token 聚类（如 CATANet）或窗口聚合来降低 Attention 复杂度。但这种“硬切分”不可避免地将无关的 Token 强行分配到一起计算，引入了严重的交叉噪声。同时，独立计算每一层的 Attention 导致深层网络在寻找相似纹理时效率低下。
*   **Method (方法)**: 我们提出 PFA-CATANet。第一把斧子：**Cluster-Isolation Mask**，在不增加聚类开销的前提下，利用现成的聚类标签物理阻断跨类噪声。第二把斧子：**Progressive Focused Attention**，利用干净的浅层注意力图引导深层网络聚焦，避免了重复计算。
*   **Conclusion (结论)**: 这种结合不仅从理论上修复了聚类 Transformer 的结构缺陷，还在基准测试中用更少的计算成本刷新了轻量级 SR 的 SOTA 记录。