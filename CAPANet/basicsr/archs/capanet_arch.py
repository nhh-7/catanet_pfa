import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.archs.arch_util import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY


LEVEL_NAMES: Tuple[str, str, str] = ("hf", "mf", "lf")
LEVEL_SCALES: Dict[str, int] = {"hf": 1, "mf": 2, "lf": 4}
LEVEL_BASE_BETA: Dict[str, float] = {"lf": 0.35, "mf": 0.25, "hf": 0.10}
LEVEL_ETA_MAX: Dict[str, float] = {"lf": 0.2, "mf": 0.3, "hf": 0.5}


def exists(val) -> bool:
    return val is not None


def init_logit(prob: float) -> torch.Tensor:
    clipped = min(max(prob, 1e-4), 1.0 - 1e-4)
    return torch.logit(torch.tensor(clipped, dtype=torch.float32))


def depth_aware_beta_init(level: str, block_index: int, total_blocks: int) -> float:
    if total_blocks <= 1:
        t = 0.0
    else:
        t = float(block_index) / float(total_blocks - 1)
    return LEVEL_BASE_BETA[level] + 0.40 * t


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = 1) -> Tuple[torch.Tensor, int]:
    size = x.shape[dim]
    pad_n = (multiple - size % multiple) % multiple
    if pad_n == 0:
        return x, size

    tail = x.narrow(dim, max(size - pad_n, 0), min(pad_n, size))
    if tail.shape[dim] < pad_n:
        repeat_shape = [1] * x.dim()
        repeat_shape[dim] = math.ceil(pad_n / max(tail.shape[dim], 1))
        tail = tail.repeat(*repeat_shape).narrow(dim, 0, pad_n)
    tail = torch.flip(tail, dims=[dim])
    return torch.cat([x, tail], dim=dim), size


def repeat_each(x: torch.Tensor, factor: int, target_len: int) -> torch.Tensor:
    if factor == 1 and x.shape[1] == target_len:
        return x
    repeated = x.repeat_interleave(factor, dim=1)
    return repeated[:, :target_len, :]


def restore_sorted_tokens(x: torch.Tensor, idx_last: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    return out.scatter(dim=1, index=idx_last.expand_as(x), src=x)


def align_attention_state(
    prev_attn_state: Optional[torch.Tensor],
    target_shape: Sequence[int],
) -> Optional[torch.Tensor]:
    if prev_attn_state is None:
        return None

    if tuple(prev_attn_state.shape) == tuple(target_shape):
        return prev_attn_state

    target_batch, target_groups, target_heads, target_query, target_key = map(int, target_shape)
    prev_batch, _, prev_heads, _, _ = prev_attn_state.shape
    if prev_batch != target_batch:
        return None

    aligned = prev_attn_state
    if prev_heads != target_heads:
        aligned = aligned.mean(dim=2, keepdim=True).expand(-1, -1, target_heads, -1, -1)

    aligned = rearrange(aligned, "b ng h q k -> (b h) 1 ng q k")
    aligned = F.interpolate(
        aligned,
        size=(target_groups, target_query, target_key),
        mode="trilinear",
        align_corners=False,
    )
    aligned = rearrange(aligned, "(b h) 1 ng q k -> b ng h q k", b=target_batch, h=target_heads)
    aligned = aligned.clamp_min(0.0)
    return aligned / (aligned.sum(dim=-1, keepdim=True) + 1e-9)


def identity_routing(
    x: torch.Tensor, num_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, channels = x.shape
    idx_last = torch.arange(length, device=x.device).view(1, length, 1).expand(batch, -1, -1)
    labels = torch.zeros(batch, length, dtype=torch.long, device=x.device)
    scores = torch.ones(batch, length, dtype=x.dtype, device=x.device)
    prototype = F.normalize(x.mean(dim=1, keepdim=True), dim=-1)
    prototypes = prototype.expand(-1, num_tokens, -1)
    return x, idx_last, labels, scores, prototypes


def segment_pool_features(x: torch.Tensor, scores: torch.Tensor, scale: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale == 1:
        return x, scores

    x_pad, _ = pad_to_multiple(x, scale, dim=1)
    scores_pad, _ = pad_to_multiple(scores, scale, dim=1)
    bsz, padded_len, channels = x_pad.shape
    x_grouped = x_pad.reshape(bsz, padded_len // scale, scale, channels)
    score_grouped = scores_pad.reshape(bsz, padded_len // scale, scale)
    weights = score_grouped.unsqueeze(-1).clamp_min(1e-6)
    pooled_x = (x_grouped * weights).sum(dim=2) / weights.sum(dim=2)
    pooled_scores = score_grouped.mean(dim=2)
    return pooled_x, pooled_scores


def segment_pool_labels(labels: torch.Tensor, scores: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return labels

    labels_pad, _ = pad_to_multiple(labels, scale, dim=1)
    scores_pad, _ = pad_to_multiple(scores, scale, dim=1)
    bsz, padded_len = labels_pad.shape
    grouped_labels = labels_pad.reshape(bsz, padded_len // scale, scale)
    grouped_scores = scores_pad.reshape(bsz, padded_len // scale, scale)

    num_classes = int(labels_pad.max().item()) + 1
    one_hot = F.one_hot(grouped_labels.long(), num_classes=num_classes)
    counts = one_hot.sum(dim=2)
    score_sums = (one_hot * grouped_scores.unsqueeze(-1)).sum(dim=2)

    max_count = counts.max(dim=-1, keepdim=True).values
    count_mask = counts == max_count
    score_sums = score_sums.masked_fill(~count_mask, -1e9)

    max_score = score_sums.max(dim=-1, keepdim=True).values
    score_mask = score_sums == max_score

    positions = torch.arange(scale, device=labels.device).view(1, 1, scale, 1)
    first_positions = torch.where(one_hot.bool(), positions, scale).min(dim=2).values
    first_positions = first_positions.masked_fill(~score_mask, scale)
    return first_positions.argmin(dim=-1)


class DynamicPrototypeRouter(nn.Module):
    """Dynamic semantic prototype router with prototype-query refinement."""

    def __init__(self, dim: int, router_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.assign = nn.Linear(dim, num_tokens, bias=False)
        self.prototype_queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.refine_q = nn.Linear(dim, router_dim, bias=False)
        self.refine_k = nn.Linear(dim, router_dim, bias=False)
        self.refine_v = nn.Linear(dim, dim, bias=False)
        self.token_proj = nn.Linear(dim, router_dim, bias=False)
        self.prototype_proj = nn.Linear(dim, router_dim, bias=False)
        self.refine_gate = nn.Parameter(init_logit(0.25))
        self.prototype_norm = nn.LayerNorm(dim)
        self.scale = router_dim ** -0.5

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_patches, channels = x.shape
        embed = self.embed(x)

        assignment = F.softmax(self.assign(embed), dim=-1)
        proto_content = torch.einsum("bnm,bnc->bmc", assignment, x)
        proto_weight = assignment.sum(dim=1, keepdim=False).unsqueeze(-1).clamp_min(1e-6)
        proto_content = proto_content / proto_weight
        proto_content = F.normalize(self.prototype_norm(proto_content), dim=-1)

        query_seed = proto_content + self.prototype_queries.unsqueeze(0)
        q_proto = self.refine_q(query_seed)
        k_tokens = self.refine_k(embed)
        v_tokens = self.refine_v(x)
        refine_attn = F.softmax(torch.matmul(q_proto, k_tokens.transpose(-2, -1)) * self.scale, dim=-1)
        proto_refine = torch.matmul(refine_attn, v_tokens)

        gamma = torch.sigmoid(self.refine_gate)
        prototypes = F.normalize(self.prototype_norm(proto_content + gamma * proto_refine), dim=-1)

        token_features = F.normalize(self.token_proj(embed), dim=-1)
        prototype_features = F.normalize(self.prototype_proj(prototypes), dim=-1)
        scores = F.softmax(torch.matmul(token_features, prototype_features.transpose(-2, -1)) * self.scale, dim=-1)

        x_scores, belong_idx = torch.max(scores, dim=-1)
        sort_key = belong_idx.to(scores.dtype) + 0.5 * (1.0 - x_scores)
        sorted_idx = torch.argsort(sort_key, dim=-1)
        gather_idx = sorted_idx.unsqueeze(-1).expand(bsz, num_patches, channels)

        sorted_x = torch.gather(x, dim=1, index=gather_idx)
        sorted_belong_idx = torch.gather(belong_idx, dim=1, index=sorted_idx)
        sorted_scores = torch.gather(x_scores, dim=1, index=sorted_idx)
        idx_last = sorted_idx.unsqueeze(-1)
        return sorted_x, idx_last, sorted_belong_idx, sorted_scores, prototypes


class PrototypeCenterInteraction(nn.Module):
    """Per-level prototype projections for the independent global center branch."""

    def __init__(
        self,
        dim_splits: Dict[str, int],
        qk_dim_splits: Dict[str, int],
        head_splits: Dict[str, int],
        enabled_levels: Iterable[str],
    ):
        super().__init__()
        self.dim_splits = dim_splits
        self.qk_dim_splits = qk_dim_splits
        self.head_splits = head_splits
        self.enabled_levels = set(enabled_levels)
        self.k_projs = nn.ModuleDict()
        self.v_projs = nn.ModuleDict()
        for level in LEVEL_NAMES:
            if level in self.enabled_levels and self.dim_splits[level] > 0:
                self.k_projs[level] = nn.Linear(dim_splits[level], qk_dim_splits[level], bias=False)
                self.v_projs[level] = nn.Linear(dim_splits[level], dim_splits[level], bias=False)

    def forward(self, prototypes: torch.Tensor) -> Dict[str, Optional[Dict[str, torch.Tensor]]]:
        proto_levels = torch.split(prototypes, [self.dim_splits[level] for level in LEVEL_NAMES], dim=-1)
        outputs: Dict[str, Optional[Dict[str, torch.Tensor]]] = {}
        for level, proto_level in zip(LEVEL_NAMES, proto_levels):
            if level not in self.k_projs:
                outputs[level] = None
                continue
            heads = self.head_splits[level]
            k_global = rearrange(self.k_projs[level](proto_level), "b m (h d) -> b h m d", h=heads)
            v_global = rearrange(self.v_projs[level](proto_level), "b m (h d) -> b h m d", h=heads)
            outputs[level] = {"k": k_global, "v": v_global}
        return outputs


class ProgressiveFocusedSparseAttention(nn.Module):
    """PFSA with depth-aware prior fusion and gated prototype-center interaction."""

    def __init__(
        self,
        dim: int,
        qk_dim: int,
        heads: int,
        group_size: int,
        beta_init: float,
        use_sparse: bool = True,
        use_global: bool = True,
        focus_mode: str = "dynamic",
        fixed_focus_ratio: float = 0.5,
        r_base: float = 0.5,
        r_min: float = 0.25,
        r_max: float = 0.75,
        lambda_p: float = 0.25,
        lambda_v: float = 0.25,
        cross_cluster_ratio: float = 0.125,
        eta_max: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.group_size = group_size
        self.use_sparse = use_sparse
        self.use_global = use_global and eta_max > 0
        self.focus_mode = focus_mode
        self.fixed_focus_ratio = fixed_focus_ratio
        self.r_base = r_base
        self.r_min = r_min
        self.r_max = r_max
        self.lambda_p = lambda_p
        self.lambda_v = lambda_v
        self.cross_cluster_ratio = cross_cluster_ratio
        self.eta_max = eta_max

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.beta_logit = nn.Parameter(init_logit(beta_init))
        self.eta_logit = nn.Parameter(init_logit(0.5))

    @staticmethod
    def _tail_tokens(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if length == 0:
            return tensor[:, :0]
        total = tensor.shape[1]
        if length <= total:
            return tensor[:, total - length : total]
        repeat = math.ceil(length / total)
        repeat_shape = [1] * tensor.dim()
        repeat_shape[1] = repeat
        return tensor.repeat(*repeat_shape)[:, -length:]

    def build_windows(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        bsz, num_tokens, _ = q.shape
        group_size = min(num_tokens, self.group_size)
        num_groups = (num_tokens + group_size - 1) // group_size
        pad_n = num_groups * group_size - num_tokens

        q_pad = torch.cat((q, torch.flip(self._tail_tokens(q, pad_n), dims=[1])), dim=1) if pad_n > 0 else q
        k_pad = torch.cat((k, torch.flip(self._tail_tokens(k, pad_n + group_size), dims=[1])), dim=1)
        v_pad = torch.cat((v, torch.flip(self._tail_tokens(v, pad_n + group_size), dims=[1])), dim=1)

        q_windows = rearrange(q_pad, "b (ng gs) c -> b ng gs c", ng=num_groups, gs=group_size)
        k_windows = rearrange(k_pad.unfold(1, 2 * group_size, group_size), "b ng c ws -> b ng ws c")
        v_windows = rearrange(v_pad.unfold(1, 2 * group_size, group_size), "b ng c ws -> b ng ws c")

        label_pad = (
            torch.cat((labels, torch.flip(self._tail_tokens(labels, pad_n), dims=[1])), dim=1) if pad_n > 0 else labels
        )
        key_label_pad = torch.cat((labels, torch.flip(self._tail_tokens(labels, pad_n + group_size), dims=[1])), dim=1)
        score_pad = (
            torch.cat((scores, torch.flip(self._tail_tokens(scores, pad_n), dims=[1])), dim=1) if pad_n > 0 else scores
        )
        key_score_pad = torch.cat((scores, torch.flip(self._tail_tokens(scores, pad_n + group_size), dims=[1])), dim=1)

        q_labels = rearrange(label_pad, "b (ng gs) -> b ng gs", ng=num_groups, gs=group_size)
        k_labels = key_label_pad.unfold(1, 2 * group_size, group_size)
        q_scores = rearrange(score_pad, "b (ng gs) -> b ng gs", ng=num_groups, gs=group_size)
        k_scores = key_score_pad.unfold(1, 2 * group_size, group_size)
        return q_windows, k_windows, v_windows, q_labels, k_labels, q_scores, k_scores, group_size, num_groups

    def focus_ratio(self, q_labels: torch.Tensor, q_scores: torch.Tensor) -> torch.Tensor:
        if self.focus_mode == "fixed":
            return q_scores.new_full((q_scores.shape[0], q_scores.shape[1]), float(self.fixed_focus_ratio))

        mode_labels = torch.mode(q_labels, dim=-1).values
        purity = (q_labels == mode_labels.unsqueeze(-1)).float().mean(dim=-1)
        score_var = q_scores.float().var(dim=-1, unbiased=False)
        focus = self.r_base + self.lambda_p * purity - self.lambda_v * score_var
        return focus.clamp(self.r_min, self.r_max)

    @staticmethod
    def topk_mask(values: torch.Tensor, keep: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        ranks = torch.argsort(torch.argsort(values, dim=-1, descending=True), dim=-1)
        keep_view = keep.view(keep.shape[0], keep.shape[1], 1, 1, 1)
        return (ranks < keep_view) & candidate_mask

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        prev_attn_state: Optional[torch.Tensor] = None,
        global_kv: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_tokens, _ = x.shape
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q_windows, k_windows, v_windows, q_labels, k_labels, q_scores, k_scores, _, num_groups = self.build_windows(
            q, k, v, labels, scores
        )

        q_windows = rearrange(q_windows, "b ng gs (h d) -> b ng h gs d", h=self.heads)
        k_windows = rearrange(k_windows, "b ng ws (h d) -> b ng h ws d", h=self.heads)
        v_windows = rearrange(v_windows, "b ng ws (h d) -> b ng h ws d", h=self.heads)

        same_cluster_mask = (q_labels.unsqueeze(-1) == k_labels.unsqueeze(-2)).unsqueeze(2)
        scale = q_windows.shape[-1] ** -0.5
        logits = torch.matmul(q_windows, k_windows.transpose(-2, -1)) * scale
        attn_cur = F.softmax(logits, dim=-1)

        aligned_prev_attn = align_attention_state(prev_attn_state, attn_cur.shape)
        if aligned_prev_attn is not None:
            beta = torch.sigmoid(self.beta_logit)
            attn_hat = beta * aligned_prev_attn + (1.0 - beta) * attn_cur
            attn_hat = attn_hat / (attn_hat.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            attn_hat = attn_cur

        if self.use_sparse:
            focus = self.focus_ratio(q_labels, q_scores)
            keep = torch.ceil(focus * attn_hat.shape[-1]).long().clamp(min=1, max=attn_hat.shape[-1])
            cross_keep = torch.round(keep.float() * self.cross_cluster_ratio).long()
            cross_keep = torch.where(keep > 1, cross_keep.clamp(min=1), torch.zeros_like(cross_keep))
            same_keep = (keep - cross_keep).clamp(min=1)

            same_scores = attn_hat.masked_fill(~same_cluster_mask, -1e4)
            topk_same_mask = self.topk_mask(same_scores, same_keep, same_cluster_mask)

            cross_cluster_mask = ~same_cluster_mask
            pair_confidence = q_scores.unsqueeze(2).unsqueeze(-1) * k_scores.unsqueeze(2).unsqueeze(-2)
            cross_scores = attn_hat * pair_confidence
            topk_cross_mask = self.topk_mask(cross_scores, cross_keep, cross_cluster_mask)

            final_mask = topk_same_mask | topk_cross_mask
            sparse_logits = logits.masked_fill(~final_mask, -1e4)
            attn_local = F.softmax(sparse_logits, dim=-1)
            attn_local = attn_local.masked_fill(~final_mask, 0.0)
            attn_local = attn_local / (attn_local.sum(dim=-1, keepdim=True) + 1e-9)
            current_attn_state = attn_local
        else:
            attn_local = attn_hat
            current_attn_state = attn_hat

        out_local = torch.matmul(attn_local, v_windows)

        if self.use_global and global_kv is not None:
            k_global = global_kv["k"].unsqueeze(1).expand(bsz, num_groups, -1, -1, -1)
            v_global = global_kv["v"].unsqueeze(1).expand(bsz, num_groups, -1, -1, -1)
            out_global = F.scaled_dot_product_attention(q_windows, k_global, v_global)
            eta = self.eta_max * torch.sigmoid(self.eta_logit)
            out = out_local + eta * out_global
        else:
            out = out_local

        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :num_tokens, :]
        return self.proj(out), current_attn_state


class LowToHighMultiLevelReconstruction(nn.Module):
    """LMR orchestration over low/mid/high frequency branches."""

    def __init__(
        self,
        dim: int,
        qk_dim: int,
        heads: int,
        group_size: int,
        block_index: int,
        total_blocks: int,
        level_head_split: Sequence[int] = (2, 1, 1),
        focus_mode: str = "dynamic",
        fixed_focus_ratio: float = 0.5,
        use_sparse_pfsa: bool = True,
        global_branch_levels: Iterable[str] = ("mf", "hf"),
    ):
        super().__init__()
        if sum(level_head_split) != heads:
            raise ValueError(f"level_head_split {level_head_split} must sum to heads={heads}.")
        if len(level_head_split) != 3:
            raise ValueError("level_head_split must contain three entries for hf/mf/lf.")

        head_dim = dim // heads
        qk_head_dim = qk_dim // heads
        self.head_splits = {level: level_head_split[idx] for idx, level in enumerate(LEVEL_NAMES)}
        self.dim_splits = {level: self.head_splits[level] * head_dim for level in LEVEL_NAMES}
        self.qk_dim_splits = {level: self.head_splits[level] * qk_head_dim for level in LEVEL_NAMES}
        self.split_sizes = [self.dim_splits[level] for level in LEVEL_NAMES]
        self.active_levels = [level for level in LEVEL_NAMES if self.dim_splits[level] > 0]

        self.global_levels = set(global_branch_levels)
        self.global_branch = PrototypeCenterInteraction(
            self.dim_splits, self.qk_dim_splits, self.head_splits, enabled_levels=self.global_levels
        )
        self.lf_to_mf_proj = self._build_projection(self.dim_splits["lf"], self.dim_splits["mf"])
        self.mf_to_hf_proj = self._build_projection(self.dim_splits["mf"], self.dim_splits["hf"])
        self.lf_to_hf_proj = self._build_projection(self.dim_splits["lf"], self.dim_splits["hf"])

        self.level_attn = nn.ModuleDict()
        for level in self.active_levels:
            beta_init = depth_aware_beta_init(level, block_index, total_blocks)
            eta_max = LEVEL_ETA_MAX[level] if level in self.global_levels else 0.0
            self.level_attn[level] = ProgressiveFocusedSparseAttention(
                dim=self.dim_splits[level],
                qk_dim=self.qk_dim_splits[level],
                heads=self.head_splits[level],
                group_size=max(1, group_size // LEVEL_SCALES[level]),
                beta_init=beta_init,
                use_sparse=use_sparse_pfsa,
                use_global=level in self.global_levels,
                focus_mode=focus_mode,
                fixed_focus_ratio=fixed_focus_ratio,
                eta_max=eta_max,
            )

        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.proj = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _build_projection(in_dim: int, out_dim: int) -> Optional[nn.Linear]:
        if in_dim <= 0 or out_dim <= 0:
            return None
        return nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        sorted_x: torch.Tensor,
        idx_last: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        prototypes: torch.Tensor,
        prev_attn_state: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        prev_attn_state = prev_attn_state or {}
        level_inputs = {
            level: split for level, split in zip(LEVEL_NAMES, torch.split(sorted_x, self.split_sizes, dim=-1))
        }
        global_kv = self.global_branch(prototypes)

        pooled_labels = {"hf": labels}
        pooled_scores = {"hf": scores}
        if "mf" in self.active_levels:
            level_inputs["mf"], pooled_scores["mf"] = segment_pool_features(level_inputs["mf"], scores, scale=2)
            pooled_labels["mf"] = segment_pool_labels(labels, scores, scale=2)
        if "lf" in self.active_levels:
            level_inputs["lf"], pooled_scores["lf"] = segment_pool_features(level_inputs["lf"], scores, scale=4)
            pooled_labels["lf"] = segment_pool_labels(labels, scores, scale=4)

        outputs: Dict[str, Optional[torch.Tensor]] = {level: None for level in LEVEL_NAMES}
        current_attn_state: Dict[str, Optional[torch.Tensor]] = {level: None for level in LEVEL_NAMES}

        if "lf" in self.active_levels:
            outputs["lf"], current_attn_state["lf"] = self.level_attn["lf"](
                level_inputs["lf"],
                pooled_labels["lf"],
                pooled_scores["lf"],
                prev_attn_state.get("lf"),
                global_kv.get("lf"),
            )

        if "mf" in self.active_levels:
            mf_input = level_inputs["mf"]
            if outputs["lf"] is not None:
                lf_to_mf = self.lf_to_mf_proj(outputs["lf"]) if self.lf_to_mf_proj is not None else outputs["lf"]
                mf_input = mf_input + self.gamma_1 * repeat_each(lf_to_mf, factor=2, target_len=mf_input.shape[1])
            outputs["mf"], current_attn_state["mf"] = self.level_attn["mf"](
                mf_input,
                pooled_labels["mf"],
                pooled_scores["mf"],
                prev_attn_state.get("mf"),
                global_kv.get("mf"),
            )

        if "hf" in self.active_levels:
            hf_input = level_inputs["hf"]
            if outputs["mf"] is not None:
                mf_to_hf = self.mf_to_hf_proj(outputs["mf"]) if self.mf_to_hf_proj is not None else outputs["mf"]
                hf_input = hf_input + self.gamma_2 * repeat_each(mf_to_hf, factor=2, target_len=hf_input.shape[1])
            elif outputs["lf"] is not None:
                up_factor = max(1, hf_input.shape[1] // max(outputs["lf"].shape[1], 1))
                lf_to_hf = self.lf_to_hf_proj(outputs["lf"]) if self.lf_to_hf_proj is not None else outputs["lf"]
                hf_input = hf_input + self.gamma_2 * repeat_each(lf_to_hf, factor=up_factor, target_len=hf_input.shape[1])
            outputs["hf"], current_attn_state["hf"] = self.level_attn["hf"](
                hf_input,
                pooled_labels["hf"],
                pooled_scores["hf"],
                prev_attn_state.get("hf"),
                global_kv.get("hf"),
            )

        fusion_parts = []
        if outputs["hf"] is not None:
            fusion_parts.append(outputs["hf"])
        if outputs["mf"] is not None:
            fusion_parts.append(repeat_each(outputs["mf"], factor=2, target_len=sorted_x.shape[1]))
        if outputs["lf"] is not None:
            fusion_parts.append(repeat_each(outputs["lf"], factor=4, target_len=sorted_x.shape[1]))

        fused = self.proj(torch.cat(fusion_parts, dim=-1))
        fused = restore_sorted_tokens(fused, idx_last)
        return fused, current_attn_state


def patch_divide(x: torch.Tensor, step: int, patch_size: int) -> Tuple[torch.Tensor, int, int]:
    """Crop image into patches."""
    batch, channels, height, width = x.size()
    if height == patch_size and width == patch_size:
        step = patch_size
    crop_x = []
    num_h = 0
    for i in range(0, height + step - patch_size, step):
        top = i
        down = i + patch_size
        if down > height:
            top = height - patch_size
            down = height
        num_h += 1
        for j in range(0, width + step - patch_size, step):
            left = j
            right = j + patch_size
            if right > width:
                left = width - patch_size
                right = width
            crop_x.append(x[:, :, top:down, left:right])
    num_w = len(crop_x) // num_h
    crop_x = torch.stack(crop_x, dim=0)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()
    return crop_x, num_h, num_w


def patch_reverse(crop_x: torch.Tensor, x: torch.Tensor, step: int, patch_size: int) -> torch.Tensor:
    """Reverse patches into image."""
    _, _, height, width = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, height + step - patch_size, step):
        top = i
        down = i + patch_size
        if down > height:
            top = height - patch_size
            down = height
        for j in range(0, width + step - patch_size, step):
            left = j
            right = j + patch_size
            if right > width:
                left = width - patch_size
                right = width
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, height + step - patch_size, step):
        top = i
        down = i + patch_size - step
        if top + patch_size > height:
            top = height - patch_size
        output[:, :, top:down, :] /= 2
    for j in range(step, width + step - patch_size, step):
        left = j
        right = j + patch_size - step
        if left + patch_size > width:
            left = width - patch_size
        output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class dwconv(nn.Module):
    def __init__(self, hidden_features: int, kernel_size: int = 5):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                dilation=1,
                groups=hidden_features,
            ),
            nn.GELU(),
        )
        self.hidden_features = hidden_features

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class ConvFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        kernel_size: int = 5,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        return self.fc2(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, qk_dim: int):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)


class LRSA(nn.Module):
    def __init__(self, dim: int, qk_dim: int, mlp_dim: int, heads: int = 1):
        super().__init__()
        self.layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads, qk_dim)), PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        step = patch_size - 2
        crop_x, _, _ = patch_divide(x, step, patch_size)
        batch, num_patch, _, patch_h, patch_w = crop_x.shape
        crop_x = rearrange(crop_x, "b n c h w -> (b n) (h w) c")

        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, "(b n) (h w) c -> b n c h w", b=batch, n=num_patch, w=patch_w)

        x = patch_reverse(crop_x, x, step, patch_size)
        _, _, height, width = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = ff(x, x_size=(height, width)) + x
        return rearrange(x, "b (h w) c -> b c h w", h=height)


class CAPABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        qk_dim: int,
        mlp_dim: int,
        heads: int,
        num_tokens: int = 8,
        group_size: int = 128,
        block_index: int = 0,
        total_blocks: int = 1,
        level_head_split: Sequence[int] = (2, 1, 1),
        focus_mode: str = "dynamic",
        fixed_focus_ratio: float = 0.5,
        use_sparse_pfsa: bool = True,
        global_branch_levels: Iterable[str] = ("mf", "hf"),
        routing_mode: str = "dpr",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.routing_mode = routing_mode
        self.num_tokens = num_tokens
        self.dpr = DynamicPrototypeRouter(dim, qk_dim, num_tokens)
        self.lmr = LowToHighMultiLevelReconstruction(
            dim=dim,
            qk_dim=qk_dim,
            heads=heads,
            group_size=group_size,
            block_index=block_index,
            total_blocks=total_blocks,
            level_head_split=level_head_split,
            focus_mode=focus_mode,
            fixed_focus_ratio=fixed_focus_ratio,
            use_sparse_pfsa=use_sparse_pfsa,
            global_branch_levels=global_branch_levels,
        )
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(
        self, x: torch.Tensor, prev_attn_state: Optional[Dict[str, Optional[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        _, _, height, width = x.shape
        x_tokens = rearrange(x, "b c h w -> b (h w) c")
        residual = x_tokens
        normed_x = self.norm(x_tokens)
        if self.routing_mode == "identity":
            sorted_x, idx_last, sorted_labels, x_scores, prototypes = identity_routing(normed_x, self.num_tokens)
        else:
            sorted_x, idx_last, sorted_labels, x_scores, prototypes = self.dpr(normed_x)
        y, current_attn_state = self.lmr(sorted_x, idx_last, sorted_labels, x_scores, prototypes, prev_attn_state)
        y = rearrange(y, "b (h w) c -> b c h w", h=height).contiguous()
        y = self.conv1x1(y)
        x_tokens = residual + rearrange(y, "b c h w -> b (h w) c")
        x_tokens = self.mlp(x_tokens, x_size=(height, width)) + x_tokens
        return rearrange(x_tokens, "b (h w) c -> b c h w", h=height), current_attn_state


TAB = CAPABlock


@ARCH_REGISTRY.register()
class CAPANet(nn.Module):
    setting = dict(dim=40, block_num=8, qk_dim=36, mlp_dim=96, heads=4, patch_size=[16, 20, 24, 28, 16, 20, 24, 28])

    def __init__(
        self,
        in_chans: int = 3,
        n_iters: Sequence[int] = (5, 5, 5, 5, 5, 5, 5, 5),
        num_tokens: Sequence[int] = (16, 32, 64, 128, 16, 32, 64, 128),
        group_size: Sequence[int] = (256, 128, 64, 32, 256, 128, 64, 32),
        upscale: int = 4,
        level_head_split: Sequence[int] = (2, 1, 1),
        focus_mode: str = "dynamic",
        fixed_focus_ratio: float = 0.5,
        use_sparse_pfsa: bool = True,
        global_branch_levels: Iterable[str] = ("mf", "hf"),
        attn_state_mode: str = "layered",
        routing_mode: str = "dpr",
    ):
        super().__init__()
        self.dim = self.setting["dim"]
        self.block_num = self.setting["block_num"]
        self.patch_size = self.setting["patch_size"]
        self.qk_dim = self.setting["qk_dim"]
        self.mlp_dim = self.setting["mlp_dim"]
        self.upscale = upscale
        self.heads = self.setting["heads"]

        self.n_iters = n_iters
        self.num_tokens = list(num_tokens)
        self.group_size = list(group_size)
        self.level_head_split = tuple(level_head_split)
        self.focus_mode = focus_mode
        self.fixed_focus_ratio = fixed_focus_ratio
        self.use_sparse_pfsa = use_sparse_pfsa
        self.global_branch_levels = tuple(global_branch_levels)
        self.attn_state_mode = attn_state_mode
        self.routing_mode = routing_mode

        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        for idx in range(self.block_num):
            self.blocks.append(
                nn.ModuleList(
                    [
                        CAPABlock(
                            self.dim,
                            self.qk_dim,
                            self.mlp_dim,
                            self.heads,
                            num_tokens=self.num_tokens[idx],
                            group_size=self.group_size[idx],
                            block_index=idx,
                            total_blocks=self.block_num,
                            level_head_split=self.level_head_split,
                            focus_mode=self.focus_mode,
                            fixed_focus_ratio=self.fixed_focus_ratio,
                            use_sparse_pfsa=self.use_sparse_pfsa,
                            global_branch_levels=self.global_branch_levels,
                            routing_mode=self.routing_mode,
                        ),
                        LRSA(self.dim, self.qk_dim, self.mlp_dim, self.heads),
                    ]
                )
            )
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1))

        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale in (2, 3):
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale**2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.last_conv = nn.Conv2d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _build_prev_state(
        self,
        attn_state: Dict[str, Optional[torch.Tensor]],
        shared_state: Optional[torch.Tensor],
    ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        if self.attn_state_mode == "none":
            return None
        if self.attn_state_mode == "shared":
            return {level: shared_state for level in LEVEL_NAMES}
        return attn_state

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        attn_state = {level: None for level in LEVEL_NAMES}
        shared_state: Optional[torch.Tensor] = None
        for idx in range(self.block_num):
            residual = x
            global_attn, local_attn = self.blocks[idx]
            prev_state = self._build_prev_state(attn_state, shared_state)
            x, current_state = global_attn(x, prev_state)
            x = local_attn(x, self.patch_size[idx])
            x = residual + self.mid_convs[idx](x)

            if self.attn_state_mode == "shared":
                shared_state = next((current_state[level] for level in LEVEL_NAMES if current_state[level] is not None), None)
                attn_state = {level: shared_state for level in LEVEL_NAMES}
            else:
                attn_state = current_state
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False) if self.upscale != 1 else x
        x = self.first_conv(x)
        x = self.forward_features(x) + x

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))
        return self.last_conv(out) + base

    def __repr__(self) -> str:
        num_parameters = sum(param.numel() for param in self.parameters())
        return f"#Params of {self._get_name()}: {num_parameters / 10 ** 3:<.4f} [K]"


@ARCH_REGISTRY.register()
class CATANet(CAPANet):
    """Compatibility alias around the CAPANet implementation."""

    pass


if __name__ == "__main__":
    model = CAPANet(upscale=3).cuda()
    x = torch.randn(2, 3, 128, 128).cuda()
    print(model)
    print(model(x).shape)
