import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def clamp_keep_count(target: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(1, min(int(target), int(total)))


def split_visionzip_budget(target_token_num: int, total_tokens: int, dominant_ratio: float) -> Tuple[int, int, bool]:
    # VisionZip outputs K dominant patch tokens + M contextual tokens + 1 CLS token.
    # The public experiment interface gives only final token count, so K/M are
    # deterministically split 70/30 by default while preserving the +1 CLS term.
    keep = clamp_keep_count(target_token_num, total_tokens)
    has_cls = total_tokens > 1
    patch_budget = max(0, keep - (1 if has_cls else 0))
    dominant = int(round(patch_budget * float(dominant_ratio)))
    dominant = max(0, min(patch_budget, dominant))
    contextual = max(0, patch_budget - dominant)
    return dominant, contextual, has_cls


def cls_patch_scores(attentions: torch.Tensor, patch_count: int) -> torch.Tensor:
    # FasterVLM uses [CLS] -> patch attention from the vision encoder, averaged
    # over heads. This score is computed before projector/LLM prefill.
    if attentions is None:
        return torch.ones(1, patch_count)
    attn = attentions.float()
    if attn.dim() == 4:
        score = attn[:, :, 0, 1 : 1 + patch_count].mean(dim=1)
    elif attn.dim() == 3:
        score = attn[:, 0, 1 : 1 + patch_count]
    else:
        score = attn.reshape(attn.shape[0], -1)[:, :patch_count]
    if score.shape[-1] < patch_count:
        pad = torch.zeros(score.shape[0], patch_count - score.shape[-1], device=score.device, dtype=score.dtype)
        score = torch.cat([score, pad], dim=-1)
    return torch.nan_to_num(score[:, :patch_count], nan=0.0, posinf=0.0, neginf=0.0)


def fastervlm_prune_features(
    selected_features: torch.Tensor,
    attentions: torch.Tensor,
    target_token_num: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = int(selected_features.shape[1])
    keep = clamp_keep_count(target_token_num, total)
    if keep >= total:
        return selected_features, {"visual_tokens_before": total, "visual_tokens_after": total, "reduction_ratio": 0.0}

    score = cls_patch_scores(attentions, total).to(device=selected_features.device)
    _, topk = torch.topk(score, k=keep, dim=1)
    topk = torch.sort(topk, dim=1).values
    pruned = selected_features.gather(1, topk.unsqueeze(-1).expand(-1, -1, selected_features.size(-1))).contiguous()
    reduction_ratio = 1.0 - float(keep) / float(max(1, total))
    return pruned, {"visual_tokens_before": total, "visual_tokens_after": keep, "reduction_ratio": reduction_ratio}


def _cosine_contextual_tokens(features: torch.Tensor, remaining_idx: torch.Tensor, contextual_num: int) -> torch.Tensor:
    if contextual_num <= 0 or remaining_idx.numel() <= 0:
        return features[:, :0]
    contextual_num = min(int(contextual_num), int(remaining_idx.numel()))
    remaining = features.index_select(1, remaining_idx)
    if contextual_num >= remaining.shape[1]:
        return remaining

    step = max(1, remaining.shape[1] // contextual_num)
    anchors = torch.arange(0, remaining.shape[1], step, device=features.device)[:contextual_num]
    anchor_tokens = remaining.index_select(1, anchors)
    rest_mask = torch.ones(remaining.shape[1], dtype=torch.bool, device=features.device)
    rest_mask[anchors] = False
    rest = remaining[:, rest_mask]
    if rest.numel() == 0:
        return anchor_tokens

    rest_norm = F.normalize(rest.float(), dim=-1)
    anchor_norm = F.normalize(anchor_tokens.float(), dim=-1)
    assign = torch.bmm(rest_norm, anchor_norm.transpose(1, 2)).argmax(dim=-1)
    one_hot = torch.zeros(rest.shape[0], rest.shape[1], contextual_num, device=features.device, dtype=features.dtype)
    one_hot.scatter_(2, assign.unsqueeze(-1), 1)
    counts = one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    merged = torch.bmm(one_hot.transpose(1, 2), rest) / counts
    return (anchor_tokens + merged).to(features.dtype)


def visionzip_prune_features(
    selected_features: torch.Tensor,
    attentions: torch.Tensor,
    target_token_num: int,
    dominant_ratio: float = 0.7,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = int(selected_features.shape[1])
    keep = clamp_keep_count(target_token_num, total)
    if keep >= total:
        return selected_features, {"visual_tokens_before": total, "visual_tokens_after": total, "dominant_tokens": total, "contextual_tokens": 0}

    # Mantis/LLaVA usually passes patch features to the projector after removing
    # the vision CLS token. The paper's K+M+1 budget is therefore mapped to a
    # K/M split over the available projected visual tokens, preserving the
    # experiment-facing final keep count.
    dominant_num = int(round(keep * float(dominant_ratio)))
    dominant_num = max(0, min(keep, dominant_num))
    contextual_num = max(0, keep - dominant_num)
    dominant_num = min(dominant_num, total)
    contextual_num = min(contextual_num, max(0, keep - dominant_num))

    score = cls_patch_scores(attentions, total).to(device=selected_features.device)
    _, dominant_idx = torch.topk(score, k=max(1, dominant_num), dim=1)
    dominant_idx = torch.sort(dominant_idx, dim=1).values
    dominant = selected_features.gather(1, dominant_idx.unsqueeze(-1).expand(-1, -1, selected_features.size(-1)))

    outputs: List[torch.Tensor] = []
    for batch_idx in range(selected_features.shape[0]):
        all_idx = torch.arange(total, device=selected_features.device)
        mask = torch.ones(total, dtype=torch.bool, device=selected_features.device)
        mask[dominant_idx[batch_idx]] = False
        contextual = _cosine_contextual_tokens(
            selected_features[batch_idx : batch_idx + 1],
            all_idx[mask],
            contextual_num,
        )
        outputs.append(torch.cat([dominant[batch_idx : batch_idx + 1], contextual], dim=1))

    pruned = torch.cat(outputs, dim=0).contiguous()
    if pruned.shape[1] > keep:
        pruned = pruned[:, :keep].contiguous()
    return pruned, {
        "visual_tokens_before": total,
        "visual_tokens_after": int(pruned.shape[1]),
        "dominant_tokens": int(dominant.shape[1]),
        "contextual_tokens": int(max(0, pruned.shape[1] - dominant.shape[1])),
    }


def solve_pyramiddrop_lambda(initial_tokens: int, target_avg_tokens: int, stages: int) -> float:
    # Find lambda such that mean(N * lambda^1 ... N * lambda^S) ~= target.
    initial_tokens = max(1, int(initial_tokens))
    target_avg_tokens = max(1, min(int(target_avg_tokens), initial_tokens))
    stages = max(1, int(stages))
    if target_avg_tokens >= initial_tokens:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(48):
        mid = (lo + hi) * 0.5
        avg = sum(initial_tokens * (mid ** (idx + 1)) for idx in range(stages)) / float(stages)
        if avg < target_avg_tokens:
            lo = mid
        else:
            hi = mid
    return float(hi)


def pyramiddrop_keep_schedule(initial_tokens: int, target_avg_tokens: int, stages: int) -> Tuple[List[int], float]:
    lam = solve_pyramiddrop_lambda(initial_tokens, target_avg_tokens, stages)
    counts = []
    current = int(initial_tokens)
    for idx in range(max(1, int(stages))):
        target = int(round(initial_tokens * (lam ** (idx + 1))))
        target = max(1, min(current, target))
        counts.append(target)
        current = target
    return counts, lam


def estimate_flops_reduction(prefill_len: int, before_visual: int, after_visual: int) -> float:
    dense = max(1.0, float(prefill_len) ** 2)
    pruned_len = max(1.0, float(prefill_len - max(0, before_visual - after_visual)))
    return max(0.0, min(1.0, 1.0 - (pruned_len * pruned_len) / dense))


__all__ = [
    "clamp_keep_count",
    "estimate_flops_reduction",
    "fastervlm_prune_features",
    "pyramiddrop_keep_schedule",
    "visionzip_prune_features",
]
