我最开始在代码里做过的核心修改
都在 catanet_arch.py、catanet_model.py、train.py 里，主要有这些：

把原来简单的跨层注意力线性混合，改成了“更完整的 PFA 逻辑”
原先是更像：
有 prev_attn_map
但很多时候 shape 对不上，实际继承常常不起作用
注意力融合只是简单插值
后来我改成了：

align_attn_map()：跨 block 对齐前一层注意力图
build_focus_mask()：基于前一层注意力做 top-k 聚焦筛选
乘性继承：attn_probs * prev_attn_map
再加 pfa_alpha 稳定混合
这部分是实现层面的变化，不是配置能关掉的。

新增了跨层注意力图对齐
这在之前实际上是没有真正打通的。
我加了 align_attn_map (line 46)，让不同 group_size 的 block 之间也能继承注意力。
这意味着：

现在跨层继承是真正生效的
而不是以前那种“很多层因为 shape 不同自动失效”
这个差异不是改几个数值超参能消掉的。

新增了 focus mask 机制
我加了 build_focus_mask (line 64)，让前层注意力直接参与本层 key 的筛选。
这意味着现在的 PFA 已经不是“只平滑注意力图”，而是“改变了当前层的可见连接范围”。

即使你把 focus_ratio 调到很大，它仍然是在这套新实现框架里运行。

加了 NaN 防护和 mask fallback
为了解决你训练时出现的 NaN，我又加了：
ensure_valid_mask()
focus_mask -> cluster mask -> forced keep-one 的回退逻辑
这是数值稳定性改造，也不是光调配置能撤回的。

把跨层继承改成了默认 detach
后来为了显存和稳定性，我又加了：
detach_prev_attn: true
这会让跨层注意力变成“引导信号”而不是端到端反传链。
这同样是实现层的差异，不过这个倒是已经暴露成配置开关了，可以切。

补了真实梯度累积
catanet_model.py (line 93) 和 train.py (line 188) 现在支持真实 accum_iter。
这不属于 PFA 本身，但会影响训练行为。
所以你现在改配置能达到什么
现在改配置，只能在“当前这套新实现”里切不同强度：

强 PFA
保守 PFA
超轻 PFA
它能做到的是：

让新实现更温和
让表现更接近原版 CATANet
做消融和调参
但它做不到：

回到“还没引入跨层对齐/聚焦筛选/数值兜底”时的实现状态
一句话总结

配置修改 = 调节“新 PFA 实现”的强弱
代码修改 = 改变“PFA 是怎么被实现的”