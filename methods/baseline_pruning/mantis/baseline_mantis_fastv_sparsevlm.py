import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .baseline_mantis_algorithms import clamp_keep_count


def _safe_scores(scores: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)


def fastv_visual_scores(
    attentions: torch.Tensor,
    visual_positions: torch.Tensor,
    seq_len: int,
    query_positions: torch.Tensor = None,
) -> torch.Tensor:
    # FastV paper step: after the first K decoder layers, estimate each visual
    # token's importance from attention paid by every other current token. This
    # deliberately includes any action/output tokens present in the sequence and
    # does not split visual tokens by camera/view.
    attn = attentions.float()
    if attn.dim() == 4:
        attn = attn.mean(dim=1)
    if attn.dim() != 3:
        raise ValueError(f"Unsupported attention shape for FastV: {tuple(attn.shape)}")
    device = attn.device
    if query_positions is None:
        query_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        query_mask[visual_positions.to(device=device, dtype=torch.long)] = False
        query_positions = torch.where(query_mask)[0]
    else:
        query_positions = query_positions.to(device=device, dtype=torch.long)
    if query_positions.numel() <= 0:
        query_positions = torch.arange(seq_len, device=device, dtype=torch.long)
    score = attn.index_select(1, query_positions).index_select(2, visual_positions.to(device)).mean(dim=1)
    return _safe_scores(score)


def select_top_visual_positions(
    scores: torch.Tensor,
    visual_positions: torch.Tensor,
    target_token_num: int,
) -> Tuple[torch.Tensor, int]:
    total = int(visual_positions.numel())
    keep = clamp_keep_count(target_token_num, total)
    if keep >= total:
        selected = visual_positions.unsqueeze(0).expand(scores.shape[0], -1)
        return selected.to(device=scores.device, dtype=torch.long), keep
    _, topk = torch.topk(scores, keep, dim=1)
    base = visual_positions.to(device=scores.device, dtype=torch.long).unsqueeze(0).expand(scores.shape[0], -1)
    selected = base.gather(1, topk)
    return torch.sort(selected, dim=1).values, keep


def sparsevlm_select_text_raters(
    hidden_states: torch.Tensor,
    visual_positions: torch.Tensor,
    text_positions: torch.Tensor,
    max_raters: int = 32,
) -> torch.Tensor:
    # SparseVLM paper step: first find text tokens that are genuinely related to
    # image content. Only text/query positions are eligible here, never action
    # tokens appended outside the language instruction.
    if text_positions.numel() <= 0:
        return text_positions.to(device=hidden_states.device, dtype=torch.long)
    visual = hidden_states.index_select(1, visual_positions.to(hidden_states.device)).float()
    text = hidden_states.index_select(1, text_positions.to(hidden_states.device)).float()
    affinity = torch.matmul(F.normalize(text, dim=-1), F.normalize(visual, dim=-1).transpose(1, 2)).amax(dim=-1)
    mean_affinity = _safe_scores(affinity.mean(dim=0, keepdim=True))[0]
    k = max(1, min(int(max_raters), int(text_positions.numel())))
    _, topk = torch.topk(mean_affinity, k=k, dim=0)
    return torch.sort(text_positions.to(device=hidden_states.device, dtype=torch.long).gather(0, topk), dim=0).values


def sparsevlm_visual_scores(
    attentions: torch.Tensor,
    visual_positions: torch.Tensor,
    rater_positions: torch.Tensor,
) -> torch.Tensor:
    # SparseVLM paper step: use selected image-related text raters to score
    # visual tokens at each decoder layer.
    attn = attentions.float()
    if attn.dim() == 4:
        attn = attn.mean(dim=1)
    if rater_positions.numel() <= 0:
        return torch.ones(attn.shape[0], visual_positions.numel(), device=attn.device)
    score = attn.index_select(1, rater_positions.to(attn.device)).index_select(2, visual_positions.to(attn.device)).mean(dim=1)
    return _safe_scores(score)


def _effective_rank(tokens: torch.Tensor, tolerance: float) -> int:
    if tokens.shape[0] <= 1:
        return int(tokens.shape[0])
    centered = tokens.float() - tokens.float().mean(dim=0, keepdim=True)
    try:
        singular = torch.linalg.svdvals(centered)
    except RuntimeError:
        return int(tokens.shape[0])
    if singular.numel() <= 0:
        return 1
    threshold = max(float(tolerance), 1e-8) * float(singular.max().item())
    return max(1, int((singular > threshold).sum().item()))


def sparsevlm_adaptive_counts(
    hidden_states: torch.Tensor,
    visual_positions: torch.Tensor,
    target_token_num: int,
    remaining_pruning_layers: int,
    *,
    rank_tolerance: float = 1e-4,
    min_prune_tokens: int = 1,
) -> Tuple[int, int, Dict[str, float]]:
    # SparseVLM paper step: decide how many tokens to delete adaptively from
    # visual rank/redundancy, then project the plan toward the explicit final
    # budget required by the benchmark sweep.
    current = int(visual_positions.numel())
    target = clamp_keep_count(target_token_num, current)
    if current <= target:
        return current, 0, {"sparsevlm_effective_rank": float(current), "sparsevlm_redundancy": 0.0}

    visual = hidden_states[0].index_select(0, visual_positions.to(hidden_states.device))
    rank = _effective_rank(visual, rank_tolerance)
    redundancy = max(0.0, 1.0 - float(rank) / float(max(1, current)))
    remaining_layers = max(1, int(remaining_pruning_layers))
    budget_gap = current - target
    projected_drop = int(math.ceil(float(budget_gap) / float(remaining_layers)))
    rank_drop = int(round(float(current) * min(0.5, redundancy) / float(remaining_layers)))
    drop = max(int(min_prune_tokens), projected_drop, rank_drop)
    drop = max(1, min(budget_gap, drop))
    return current - drop, drop, {"sparsevlm_effective_rank": float(rank), "sparsevlm_redundancy": float(redundancy)}


def sparsevlm_merge_rescued_tokens(
    hidden_states: torch.Tensor,
    global_indices: torch.Tensor,
    dropped_positions: torch.Tensor,
    dropped_scores: torch.Tensor,
    *,
    recycle_top_ratio: float = 0.5,
    cluster_ratio: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # SparseVLM paper step: recycle a subset of removed visual tokens, cluster
    # them by feature similarity, and merge each cluster into one contextual
    # token that continues through later decoder layers.
    if dropped_positions.numel() <= 0:
        return hidden_states[:, :0], global_indices[:, :0]
    bsz = hidden_states.shape[0]
    recycle = int(math.ceil(float(dropped_positions.numel()) * max(0.0, min(1.0, float(recycle_top_ratio)))))
    recycle = max(0, min(int(dropped_positions.numel()), recycle))
    if recycle <= 0:
        return hidden_states[:, :0], global_indices[:, :0]

    _, rescue_order = torch.topk(dropped_scores, k=recycle, dim=1)
    base = dropped_positions.to(device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    rescued_pos = base.gather(1, rescue_order)
    rescued_hidden = hidden_states.gather(1, rescued_pos.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)))
    rescued_global = global_indices.gather(1, rescued_pos)

    clusters = int(math.ceil(float(recycle) * max(0.0, min(1.0, float(cluster_ratio)))))
    clusters = max(1, min(recycle, clusters))
    if clusters >= recycle:
        return rescued_hidden.contiguous(), rescued_global.contiguous()

    outputs = []
    output_globals = []
    for batch_idx in range(bsz):
        tokens = rescued_hidden[batch_idx : batch_idx + 1]
        anchor_idx = torch.linspace(0, recycle - 1, clusters, device=hidden_states.device).round().long().unique()
        anchor_idx = anchor_idx[:clusters]
        anchors = tokens.index_select(1, anchor_idx)
        sim = torch.matmul(F.normalize(tokens.float(), dim=-1), F.normalize(anchors.float(), dim=-1).transpose(1, 2))
        assign = sim.argmax(dim=-1)[0]
        merged = []
        merged_global = []
        for cluster_idx in range(int(anchor_idx.numel())):
            mask = assign == cluster_idx
            if not bool(mask.any()):
                mask[anchor_idx[cluster_idx]] = True
            merged.append(tokens[:, mask].mean(dim=1, keepdim=True))
            merged_global.append(rescued_global[batch_idx : batch_idx + 1, mask].min(dim=1, keepdim=True).values)
        outputs.append(torch.cat(merged, dim=1))
        output_globals.append(torch.cat(merged_global, dim=1))
    return torch.cat(outputs, dim=0).to(hidden_states.dtype).contiguous(), torch.cat(output_globals, dim=0).contiguous()


__all__ = [
    "fastv_visual_scores",
    "select_top_visual_positions",
    "sparsevlm_adaptive_counts",
    "sparsevlm_merge_rescued_tokens",
    "sparsevlm_select_text_raters",
    "sparsevlm_visual_scores",
]
