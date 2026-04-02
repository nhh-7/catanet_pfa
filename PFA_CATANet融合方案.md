# PFA-CATANet (Progressive Focused Content-Aware Token Aggregation) 详细架构与逻辑融合方案

本文档提供将 CVPR 2025 的 **Progressive Focused Attention (PFA)** (出自 PFT 论文) 融合进 **CATANet** 算法的详细架构级步骤与逻辑设计。本方案旨在解决 CATANet 在子组（Subgroup）划分时引入的跨聚类噪声问题，同时引入渐进式注意力聚焦，以在不增加（甚至降低）计算复杂度的情况下提升超分辨率性能。

---

## 1. 核心改进思想
1. **网络级注意力继承 (Attention Inheritance)**：让深层网络能够复用和聚焦浅层网络学到的有效注意力分布。
2. **聚类隔离掩码 (Cluster-Isolation Mask)**：在执行子组 (Subgroup) 注意力计算时，利用真实的聚类 ID 屏蔽被强制划分到同一个子组但属于不同聚类的 Token 之间的交互。

---

## 2. 详细融合步骤

### 步骤一：网络宏观层面的数据流改造（主干网络改造）
在 CATANet 的主干特征提取部分（即连续的残差组块 Residual Groups），我们需要建立一条额外的**注意力状态传递通道**。

1. **初始化全局状态**：在开始遍历所有的 `Token-Aggregation Block (TAB)` 之前，初始化一个变量 `prev_attn_map`（初始值为 None）。
2. **状态逐层传递**：在遍历每个 `TAB` 模块时，将 `prev_attn_map` 作为额外输入传入该模块。
3. **状态更新**：`TAB` 模块在计算完毕后，不仅要返回更新后的特征图 (Feature Map)，还要返回当前层计算出的最新 `current_attn_map`，并将其赋值给 `prev_attn_map`，以便传递给下一层。

### 步骤二：TAB 模块内部的信息提取与传递
TAB 模块是 CATANet 聚类的核心，我们需要在这里提取每个 Token 的真实聚类归属，并传递给注意力计算模块 (IASA)。

1. **获取聚类索引 (Cluster Assignment)**：
   在 CATA（Content-Aware Token Aggregation）阶段，模型会计算所有 Token 与全局聚类中心 (`x_means`) 的余弦相似度。通过 `argmax` 操作，我们可以获得每个 Token 所属的聚类中心 ID，记为 `x_belong_idx` (形状为 `[Batch, N]`)。
2. **聚类索引排序对齐**：
   CATANet 会根据聚类 ID 对所有的 Token 进行排序 (Argsort)，以便将属于同一聚类的 Token 物理上移动到相邻的位置。此时，**必须对 `x_belong_idx` 进行完全相同的排序操作**，得到 `sorted_belong_idx`。
   *关键点*：经过排序后，特征序列 `x` 和聚类标签 `sorted_belong_idx` 在序列维度上是一一对应的。
3. **接口扩展**：
   将 `sorted_belong_idx` 和上层传来的 `prev_attn_map` 一起传递给内部的 IASA (Intra-Group Self-Attention) 模块。

### 步骤三：IASA 模块内部的 PFA 逻辑实现（核心）
这是融合最核心的区域。我们需要在计算注意力矩阵（Attention Logits）时，引入 PFA 聚焦机制和聚类隔离机制。

#### 3.1 序列切分与 Padding 对齐 (数据对齐关键点)
由于图像总 Token 数 `N` 可能无法被 `group_size` (`gs`) 整除，且 CATANet 为了扩大感受野，允许 Query 访问相邻的 Key/Value。这里存在严格的数据对齐要求：
1. **Query 的对齐**：
   - 如果 `N` 无法被 `gs` 整除，需要对末尾的 Token 进行 Padding（通常采用翻转 Padding）。
   - 将 Padding 后的 Query 序列 Reshape 为 `[Batch, 组数(ng), Head数, gs, Dim]`。
2. **Key / Value 的对齐（重叠窗口）**：
   - 因为当前子组的 Query 要访问当前子组和**下一个子组**的 Key，所以 Key 的有效长度是 `2 * gs`。
   - 对 Key 进行 Padding 时，需要比 Query 多 Pad 一个 `gs` 的长度。
   - 使用滑动窗口操作 (Unfold) 将 Key 划分为重叠的块，Reshape 后，每个子组对应的 Key 长度为 `2 * gs`。

#### 3.2 Cluster-Isolation Mask 的生成与对齐
为了防止强制子组划分带来的跨聚类噪声，必须为 Attention Matrix 生成一个布尔掩码：
1. **Query 标签的 Padding 与 Reshape**：
   对 `sorted_belong_idx` 进行与 Query 完全相同的 Padding 和 Reshape，得到 `paded_idx_q` (形状对应于 `gs`)。
2. **Key 标签的 Padding 与 Unfold**：
   对 `sorted_belong_idx` 进行与 Key 完全相同的 Padding (多 Pad 一个 `gs`) 和 Unfold 操作，得到 `paded_idx_k` (形状对应于 `2 * gs`)。
3. **生成 Mask**：
   比较 `paded_idx_q` 和 `paded_idx_k`。当 Query Token 和 Key Token 的聚类 ID 严格相等时，掩码对应位置为 `True`，否则为 `False`。最终生成的 Mask 形状必须与注意力矩阵 `[Batch, ng, gs, 2*gs]` 严格对齐（可能需要广播 Head 维度）。

#### 3.3 渐进式聚焦注意力 (Progressive Focused Attention) 的计算
在获得 Q、K、V 和 Mask 后，执行 PFA 逻辑：
1. **计算初始 Logits**：计算 Q 和 K 的点积缩放结果，得到 `attn_logits`。
2. **应用隔离掩码**：利用上一步生成的 Mask，将 `attn_logits` 中掩码为 `False`（不同聚类）的位置强行替换为负无穷 (`-inf`)。
3. **计算 Softmax**：对 `attn_logits` 执行 Softmax，此时不同聚类的 Token 之间的注意力权重将严格为 0。
4. **融合前层 Attention (PFA 核心公式)**：
   - 如果传入了 `prev_attn_map`，则根据 PFT 的公式进行加权融合：`fused_probs = α * prev_attn_map + (1 - α) * 当前_Softmax_probs`。（$\alpha$ 为 Focus Ratio 超参数，建议设为 0.5 左右）。
   - **注意对齐与重归一化**：由于加权后概率和可能发生轻微偏移，且我们需要确保跨聚类位置依然为 0，需再次使用 Mask 将无关位置置 0，并在最后对概率分布进行重归一化（除以总和）。
5. **计算最终特征并保存状态**：
   - 将 `fused_probs` 与 Value 矩阵相乘，得到子组内的注意力输出特征。
   - 将当前的 `fused_probs` 作为 `current_attn_map` 从模块中返回，供下一层网络继承。

### 步骤四：与全局注意力 (IRCA) 分支的合并
CATANet 的 IASA 模块中除了计算上述的子组内部注意力外，还会让 Query 与全局聚类中心 (Global K/V) 计算一次全局交叉注意力。
这一部分**保持原样不变**。
最终，将 PFA 增强后的子组注意力输出与全局交叉注意力输出相加，再还原回原始的图像 Token 顺序，即可完成本层特征的提取。

---

## 3. 方案总结
通过上述架构级重构：
1. **数据流层面**增加了 `prev_attn_map` 的自顶向下传递，实现了 PFT 论文中的 Progressive Focus 机制，引导深层网络聚焦关键纹理。
2. **逻辑层面**巧妙利用 CATA 模块已经计算出的聚类 ID，构建了 `Cluster-Isolation Mask`，在特征空间上彻底“物理隔离”了不相关的 Token，修复了 CATANet 强制子组切分带来的理论缺陷。
3. 整个融合过程无需引入新的可学习参数（除了一个固定的标量超参 $\alpha$），完全符合 CATANet 轻量级超分的定位。