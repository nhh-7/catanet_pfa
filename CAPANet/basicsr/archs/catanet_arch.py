import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_
import math

def exists(val):
    return val is not None

def is_empty(t):
    return t.nelement() == 0

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))
    
    
def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def center_iter(x, means, buckets = None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means
    
class IASA(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size
        
    
    def forward(self, normed_x, idx_last, k_global, v_global, sorted_belong_idx=None, prev_attn_map=None):
        x = normed_x
        B, N, _ = x.shape
       
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))
   
        gs = min(N, self.group_size)  # group size
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N
        
        paded_q = torch.cat((q, torch.flip(q[:,N-pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d",ng=ng,h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2,2*gs,gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2,2*gs,gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d",h=self.heads)

        # Step 3.2: Cluster-Isolation Mask
        if sorted_belong_idx is not None:
            paded_idx_q_full = torch.cat((sorted_belong_idx, torch.flip(sorted_belong_idx[:, N-pad_n:N], dims=[-1])), dim=-1)
            paded_idx_q = rearrange(paded_idx_q_full, "b (ng gs) -> b ng gs", ng=ng, gs=gs)
            
            paded_idx_k_full = torch.cat((sorted_belong_idx, torch.flip(sorted_belong_idx[:, N-pad_n-gs:N], dims=[-1])), dim=-1)
            paded_idx_k = paded_idx_k_full.unfold(-1, 2*gs, gs)
            
            mask = (paded_idx_q.unsqueeze(-1) == paded_idx_k.unsqueeze(-2))
            mask = mask.unsqueeze(2) # b ng 1 gs 2*gs
        else:
            mask = None

        # Step 3.3: Progressive Focused Attention
        scale = paded_q.shape[-1] ** -0.5
        attn_logits = torch.matmul(paded_q, paded_k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
            
        attn_probs = F.softmax(attn_logits, dim=-1)
        
        alpha = 0.5
        if prev_attn_map is not None and prev_attn_map.shape == attn_probs.shape:
            fused_probs = alpha * prev_attn_map + (1 - alpha) * attn_probs
            if mask is not None:
                fused_probs = fused_probs.masked_fill(~mask, 0.0)
                fused_probs = fused_probs / (fused_probs.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            fused_probs = attn_probs
            
        current_attn_map = fused_probs
        out1 = torch.matmul(fused_probs, paded_v)
        
        
        k_global = k_global.reshape(1,1,*k_global.shape).expand(B,ng,-1,-1,-1)
        v_global = v_global.reshape(1,1,*v_global.shape).expand(B,ng,-1,-1,-1)
       
        out2 = F.scaled_dot_product_attention(paded_q,k_global,v_global)
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]
 
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)
    
        return out, current_attn_map
    
class IRCA(nn.Module):
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
      
    def forward(self, normed_x, x_means):
        x = normed_x
        if self.training:
            x_global = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))
        else:
            x_global = x_means

        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        return k,v, x_global.detach()
    

class DynamicPrototypeRouter(nn.Module):
    """Dynamic semantic prototype router.

    The router keeps CATANet's content-aware token organization spirit, but
    replaces dataset-level static centers with image-conditioned prototypes.
    It returns routing metadata used by PFSA and LMR.
    """

    def __init__(self, dim, router_dim, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.prototype_logits = nn.Linear(dim, num_tokens, bias=False)
        self.prototype_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.token_proj = nn.Linear(dim, router_dim, bias=False)
        self.prototype_key = nn.Linear(dim, router_dim, bias=False)
        self.scale = router_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        proto_weights = self.prototype_logits(x).transpose(1, 2)
        proto_weights = F.softmax(proto_weights, dim=-1)
        prototypes = torch.matmul(proto_weights, x)
        prototypes = self.prototype_proj(prototypes)

        token_features = F.normalize(self.token_proj(x), dim=-1)
        prototype_features = F.normalize(self.prototype_key(prototypes), dim=-1)
        scores = torch.matmul(token_features, prototype_features.transpose(-2, -1)) * self.scale
        scores = F.softmax(scores, dim=-1)

        x_scores, belong_idx = torch.max(scores, dim=-1)
        sort_key = belong_idx.float() * 2.0 - x_scores
        sorted_idx = torch.argsort(sort_key, dim=-1)
        gather_idx = sorted_idx.unsqueeze(-1).expand(B, N, C)

        sorted_x = torch.gather(x, dim=1, index=gather_idx)
        sorted_belong_idx = torch.gather(belong_idx, dim=1, index=sorted_idx)
        sorted_scores = torch.gather(x_scores, dim=1, index=sorted_idx)
        idx_last = sorted_idx.unsqueeze(-1)

        return sorted_x, idx_last, sorted_belong_idx, sorted_scores, prototypes


def pad_to_multiple(x, multiple, dim=1):
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


def segment_pool_features(x, scores, scale):
    if scale == 1:
        return x, scores

    x_pad, original_len = pad_to_multiple(x, scale, dim=1)
    scores_pad, _ = pad_to_multiple(scores, scale, dim=1)
    B, padded_len, C = x_pad.shape
    x_grouped = x_pad.reshape(B, padded_len // scale, scale, C)
    score_grouped = scores_pad.reshape(B, padded_len // scale, scale)
    weights = score_grouped.unsqueeze(-1).clamp_min(1e-6)
    pooled_x = (x_grouped * weights).sum(dim=2) / weights.sum(dim=2)
    pooled_scores = score_grouped.mean(dim=2)
    return pooled_x, pooled_scores


def segment_pool_labels(labels, scores, scale):
    if scale == 1:
        return labels

    labels_pad, _ = pad_to_multiple(labels, scale, dim=1)
    scores_pad, _ = pad_to_multiple(scores, scale, dim=1)
    B, padded_len = labels_pad.shape
    grouped_labels = labels_pad.reshape(B, padded_len // scale, scale)
    grouped_scores = scores_pad.reshape(B, padded_len // scale, scale)

    values, _ = torch.mode(grouped_labels, dim=-1)
    tied = grouped_labels == values.unsqueeze(-1)
    has_mode = tied.any(dim=-1)
    best_score_idx = torch.argmax(grouped_scores, dim=-1)
    fallback = torch.gather(grouped_labels, dim=-1, index=best_score_idx.unsqueeze(-1)).squeeze(-1)
    return torch.where(has_mode, values, fallback)


def upsample_tokens(x, target_len):
    current_len = x.shape[1]
    if current_len == target_len:
        return x
    repeat = math.ceil(target_len / current_len)
    return x.repeat_interleave(repeat, dim=1)[:, :target_len, :]


def restore_sorted_tokens(x, idx_last):
    out = torch.zeros_like(x)
    return out.scatter(dim=1, index=idx_last.expand_as(x), src=x)


class ProgressiveFocusedSparseAttention(nn.Module):
    """Progressive focused sparse attention.

    PFSA combines current content similarity, previous attention priors, DPR
    routing labels, and a content-aware focus ratio to keep only high-value
    token connections.
    """

    def __init__(self, dim, qk_dim, heads, group_size, num_tokens,
                 use_global=True, r_base=0.5, r_min=0.25, r_max=0.75,
                 lambda_p=0.25, lambda_v=0.25, beta=0.5):
        super().__init__()
        self.heads = heads
        self.group_size = group_size
        self.num_tokens = num_tokens
        self.use_global = use_global
        self.r_base = r_base
        self.r_min = r_min
        self.r_max = r_max
        self.lambda_p = lambda_p
        self.lambda_v = lambda_v

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.beta_logit = nn.Parameter(torch.logit(torch.tensor(beta)))

    def build_windows(self, x, labels, scores):
        B, N, _ = x.shape
        gs = min(N, self.group_size)
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N

        def tail_tokens(t, length):
            if length == 0:
                return t[:, :0]
            if length <= N:
                return t[:, N - length:N]
            repeat = math.ceil(length / N)
            repeat_shape = [1] * t.dim()
            repeat_shape[1] = repeat
            return t.repeat(*repeat_shape)[:, -length:]

        q_pad = torch.cat((x, torch.flip(tail_tokens(x, pad_n), dims=[1])), dim=1) if pad_n > 0 else x
        k_tail = torch.flip(tail_tokens(x, pad_n + gs), dims=[1])
        k_pad = torch.cat((x, k_tail), dim=1)

        q_windows = rearrange(q_pad, "b (ng gs) c -> b ng gs c", ng=ng, gs=gs)
        k_windows = k_pad.unfold(1, 2 * gs, gs)
        k_windows = rearrange(k_windows, "b ng c gs2 -> b ng gs2 c")

        label_pad = torch.cat((labels, torch.flip(tail_tokens(labels, pad_n), dims=[1])), dim=1) if pad_n > 0 else labels
        key_label_pad = torch.cat((labels, torch.flip(tail_tokens(labels, pad_n + gs), dims=[1])), dim=1)
        score_pad = torch.cat((scores, torch.flip(tail_tokens(scores, pad_n), dims=[1])), dim=1) if pad_n > 0 else scores

        q_labels = rearrange(label_pad, "b (ng gs) -> b ng gs", ng=ng, gs=gs)
        k_labels = key_label_pad.unfold(1, 2 * gs, gs)
        q_scores = rearrange(score_pad, "b (ng gs) -> b ng gs", ng=ng, gs=gs)
        return q_windows, k_windows, q_labels, k_labels, q_scores, gs, ng

    def focus_ratio(self, q_labels, q_scores):
        mode_labels = torch.mode(q_labels, dim=-1).values
        purity = (q_labels == mode_labels.unsqueeze(-1)).float().mean(dim=-1)
        score_var = q_scores.float().var(dim=-1, unbiased=False)
        focus = self.r_base + self.lambda_p * purity - self.lambda_v * score_var
        return focus.clamp(self.r_min, self.r_max)

    def forward(self, x, labels, scores, prev_attn_state=None, prototypes=None):
        B, N, C = x.shape
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q_windows, k_windows, q_labels, k_labels, q_scores, gs, ng = self.build_windows(q, labels, scores)
        _, v_windows, _, _, _, _, _ = self.build_windows(v, labels, scores)

        q_windows = rearrange(q_windows, "b ng gs (h d) -> b ng h gs d", h=self.heads)
        k_windows = rearrange(k_windows, "b ng gs2 (h d) -> b ng h gs2 d", h=self.heads)
        v_windows = rearrange(v_windows, "b ng gs2 (h d) -> b ng h gs2 d", h=self.heads)

        cluster_mask = q_labels.unsqueeze(-1) == k_labels.unsqueeze(-2)
        cluster_mask = cluster_mask.unsqueeze(2)

        scale = q_windows.shape[-1] ** -0.5
        logits = torch.matmul(q_windows, k_windows.transpose(-2, -1)) * scale
        logits = logits.masked_fill(~cluster_mask, -1e4)
        attn_cur = F.softmax(logits, dim=-1)
        attn_cur = attn_cur.masked_fill(~cluster_mask, 0.0)
        attn_cur = attn_cur / (attn_cur.sum(dim=-1, keepdim=True) + 1e-9)

        if prev_attn_state is not None and prev_attn_state.shape == attn_cur.shape:
            beta = torch.sigmoid(self.beta_logit)
            attn = beta * prev_attn_state + (1.0 - beta) * attn_cur
            attn = attn.masked_fill(~cluster_mask, 0.0)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            attn = attn_cur

        focus = self.focus_ratio(q_labels, q_scores)
        keep = torch.ceil(focus * attn.shape[-1]).long().clamp(min=1, max=attn.shape[-1])
        ranks = torch.argsort(torch.argsort(attn, dim=-1, descending=True), dim=-1)
        topk_mask = ranks < keep.view(B, ng, 1, 1, 1)
        sparse_mask = topk_mask & cluster_mask

        attn_sparse = attn.masked_fill(~sparse_mask, 0.0)
        attn_sparse = attn_sparse / (attn_sparse.sum(dim=-1, keepdim=True) + 1e-9)
        out = torch.matmul(attn_sparse, v_windows)

        if self.use_global and prototypes is not None:
            k_global = self.to_k(prototypes)
            v_global = self.to_v(prototypes)
            k_global = rearrange(k_global, "b m (h d) -> b h m d", h=self.heads)
            v_global = rearrange(v_global, "b m (h d) -> b h m d", h=self.heads)
            k_global = k_global.unsqueeze(1).expand(B, ng, -1, -1, -1)
            v_global = v_global.unsqueeze(1).expand(B, ng, -1, -1, -1)
            out = out + F.scaled_dot_product_attention(q_windows, k_global, v_global)

        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]
        return self.proj(out), attn_sparse


class LowToHighMultiLevelReconstruction(nn.Module):
    """Low-to-high multi-level reconstruction with PFSA at each level."""

    def __init__(self, dim, qk_dim, heads, group_size, num_tokens):
        super().__init__()
        head_dim = dim // heads
        qk_head_dim = qk_dim // heads
        self.hf_dim = head_dim * 2
        self.mf_dim = head_dim
        self.lf_dim = dim - self.hf_dim - self.mf_dim
        self.hf_qk_dim = qk_head_dim * 2
        self.mf_qk_dim = qk_head_dim
        self.lf_qk_dim = qk_dim - self.hf_qk_dim - self.mf_qk_dim

        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.lf_attn = ProgressiveFocusedSparseAttention(
            self.lf_dim, self.lf_qk_dim, 1, max(1, group_size // 4),
            num_tokens, use_global=False)
        self.mf_attn = ProgressiveFocusedSparseAttention(
            self.mf_dim, self.mf_qk_dim, 1, max(1, group_size // 2),
            num_tokens, use_global=True)
        self.hf_attn = ProgressiveFocusedSparseAttention(
            self.hf_dim, self.hf_qk_dim, 2, group_size,
            num_tokens, use_global=True)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, sorted_x, idx_last, labels, scores, prototypes, prev_attn_state=None):
        prev_attn_state = prev_attn_state or {}
        x_hf, x_mf, x_lf = torch.split(
            sorted_x, [self.hf_dim, self.mf_dim, self.lf_dim], dim=-1)
        p_hf, p_mf, p_lf = torch.split(
            prototypes, [self.hf_dim, self.mf_dim, self.lf_dim], dim=-1)

        x_mf, scores_mf = segment_pool_features(x_mf, scores, scale=2)
        labels_mf = segment_pool_labels(labels, scores, scale=2)
        x_lf, scores_lf = segment_pool_features(x_lf, scores, scale=4)
        labels_lf = segment_pool_labels(labels, scores, scale=4)

        y_lf, attn_lf = self.lf_attn(
            x_lf, labels_lf, scores_lf, prev_attn_state.get("lf"), p_lf)
        lf_to_mf = upsample_tokens(y_lf, x_mf.shape[1])
        x_mf = x_mf + self.gamma_1 * lf_to_mf

        y_mf, attn_mf = self.mf_attn(
            x_mf, labels_mf, scores_mf, prev_attn_state.get("mf"), p_mf)
        mf_to_hf = upsample_tokens(y_mf, x_hf.shape[1])
        x_hf = x_hf + self.gamma_2 * mf_to_hf

        y_hf, attn_hf = self.hf_attn(
            x_hf, labels, scores, prev_attn_state.get("hf"), p_hf)

        y_lf = upsample_tokens(y_lf, sorted_x.shape[1])
        y_mf = upsample_tokens(y_mf, sorted_x.shape[1])
        y = self.proj(torch.cat([y_hf, y_mf, y_lf], dim=-1))
        y = restore_sorted_tokens(y, idx_last)
        current_attn_state = {"lf": attn_lf, "mf": attn_mf, "hf": attn_hf}
        return y, current_attn_state

    
class TAB(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay = 0.999):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.dpr = DynamicPrototypeRouter(dim, qk_dim, num_tokens)
        self.lmr = LowToHighMultiLevelReconstruction(dim, qk_dim, heads, group_size, num_tokens)
        self.mlp = PreNorm(dim, ConvFFN(dim,mlp_dim))
        self.conv1x1 = nn.Conv2d(dim,dim,1, bias=False)

    
    def forward(self, x, prev_attn_state=None):
        _,_,h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        normed_x = self.norm(x)
        sorted_x, idx_last, sorted_belong_idx, x_scores, prototypes = self.dpr(normed_x)
        y, current_attn_state = self.lmr(
            sorted_x, idx_last, sorted_belong_idx, x_scores,
            prototypes, prev_attn_state)
        y = rearrange(y,'b (h w) c->b c h w',h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x
        return rearrange(x, 'b (h w) c->b c h w',h=h), current_attn_state
        
        
        

def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        output (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
       
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
      
        out = F.scaled_dot_product_attention(q,k,v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class LRSA(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, qk_dim, mlp_dim,heads=1):
        super().__init__()
     

        self.layer = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, qk_dim)),
                PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        
        return x


    
@ARCH_REGISTRY.register()
class CATANet(nn.Module):
    setting = dict(dim=40, block_num=8, qk_dim=36, mlp_dim=96, heads=4, 
                     patch_size=[16, 20, 24, 28, 16, 20, 24, 28])

    def __init__(self,in_chans=3,n_iters=[5,5,5,5,5,5,5,5],
                 num_tokens=[16,32,64,128,16,32,64,128],
                 group_size=[256,128,64,32,256,128,64,32],
                 upscale: int = 4):
        super().__init__()
        
    
        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.patch_size = self.setting['patch_size']
        self.qk_dim = self.setting['qk_dim']
        self.mlp_dim = self.setting['mlp_dim']
        self.upscale = upscale
        self.heads = self.setting['heads']
        
        


        self.n_iters = n_iters
        self.num_tokens = num_tokens
        self.group_size = group_size
    
        #-----------1 shallow--------------
        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)

        #----------2 deep--------------
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
   
        for i in range(self.block_num):
          
            self.blocks.append(nn.ModuleList([TAB(self.dim, self.qk_dim, self.mlp_dim,
                                                                 self.heads, self.n_iters[i], 
                                                                 self.num_tokens[i],self.group_size[i]), 
                                              LRSA(self.dim, self.qk_dim, 
                                                             self.mlp_dim,self.heads)]))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim,3,1,1))
            
        #----------3 reconstruction---------
        
      
     
        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
    
        self.last_conv = nn.Conv2d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        prev_attn_state = None
        for i in range(self.block_num):
            residual = x
      
            global_attn,local_attn = self.blocks[i]
            
            x, prev_attn_state = global_attn(x, prev_attn_state)
            
            x = local_attn(x, self.patch_size[i])
            
            x = residual + self.mid_convs[i](x)
        return x
        
    def forward(self, x):
        
        if self.upscale != 1: 
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else: 
            base = x
        x = self.first_conv(x)
        
   
        x = self.forward_features(x) + x
    
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))
        out = self.last_conv(out) + base
       
    
        return out
    
    
    def __repr__(self):
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        return '#Params of {}: {:<.4f} [K]'.format(self._get_name(),
                                                      num_parameters / 10 ** 3) 
  
  

@ARCH_REGISTRY.register()
class CAPANet(CATANet):
    """CAPANet registry alias for the DPR + PFSA + LMR implementation."""

    pass



if __name__ == '__main__':


    model = CAPANet(upscale=3).cuda()
    x = torch.randn(2, 3, 128, 128).cuda()
    print(model)
 
  
