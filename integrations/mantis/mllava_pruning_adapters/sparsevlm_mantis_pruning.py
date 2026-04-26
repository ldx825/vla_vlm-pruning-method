import math
import time
from typing import Any, Dict, Optional, Tuple

import torch

from .fastv_mantis_pruning import estimate_fastv_flops_reduction
from .sparsevlm_mantis_common import MantisSparseVLMConfig


def select_visual_relevant_text_tokens(
    hidden_states: torch.Tensor,
    visual_positions: torch.Tensor,
    text_positions: torch.Tensor,
    *,
    threshold_scale: float = 1.0,
) -> torch.Tensor:
    if visual_positions.numel() <= 0 or text_positions.numel() <= 0:
        return torch.empty(0, dtype=torch.long, device=hidden_states.device)

    # SparseVLM Sec. 3.2: select text raters by visual-text correlation.
    # The paper computes visual-to-text relevance from hidden-state affinity;
    # here only prompt text positions from the Mantis layout are candidates, so
    # generated/action tokens never enter the rater pool.
    visual_hidden = hidden_states.index_select(1, visual_positions).float()
    text_hidden = hidden_states.index_select(1, text_positions).float()
    affinity = torch.matmul(visual_hidden, text_hidden.transpose(1, 2))
    relevance = torch.softmax(affinity, dim=-1).mean(dim=1)[0]
    relevance = torch.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
    threshold = relevance.mean() * float(threshold_scale)
    selected = text_positions[relevance > threshold]
    if selected.numel() <= 0:
        selected = text_positions[torch.argmax(relevance).view(1)]
    return selected.to(dtype=torch.long)


def score_visual_tokens_by_text_raters(
    attn_weights: torch.Tensor,
    scorer_positions: torch.Tensor,
    visual_positions: torch.Tensor,
) -> torch.Tensor:
    if scorer_positions.numel() <= 0 or visual_positions.numel() <= 0:
        return torch.empty((attn_weights.shape[0], 0), dtype=torch.float32, device=attn_weights.device)
    if attn_weights.shape[2] == int(scorer_positions.numel()):
        attn_queries = attn_weights
    else:
        attn_queries = attn_weights.index_select(2, scorer_positions)
    score = attn_queries.index_select(3, visual_positions).mean(dim=1).mean(dim=1)
    return torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _effective_rank(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() <= 0:
        return torch.tensor(1.0, device=matrix.device)
    try:
        sigmas = torch.linalg.svdvals(torch.nan_to_num(matrix.float(), nan=0.0, posinf=0.0, neginf=0.0))
        sigma_sum = sigmas.sum()
        if sigma_sum <= 1e-9:
            return torch.tensor(1.0, device=matrix.device)
        probs = sigmas / sigma_sum
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))
        return torch.nan_to_num(torch.exp(entropy), nan=1.0, posinf=float(matrix.shape[0]), neginf=1.0)
    except Exception:
        return torch.tensor(1.0, device=matrix.device)


def decide_sparsevlm_keep_count(
    config: MantisSparseVLMConfig,
    visual_hidden: torch.Tensor,
    scores: torch.Tensor,
    *,
    current_visual_tokens: int,
    layer_idx: int,
    remaining_layers: int,
) -> Tuple[int, Dict[str, Any]]:
    if current_visual_tokens <= 0:
        return 0, {}

    # SparseVLM Sec. 3.4: use rank as a redundancy proxy. A low effective
    # rank means many visual tokens carry similar information, so this layer can
    # prune more aggressively; a high rank keeps a larger ROI.
    rank = _effective_rank(visual_hidden[0])
    rank_ratio = float(torch.clamp(rank / max(1.0, float(current_visual_tokens)), 0.0, 1.0).item())
    rank_ratio = max(0.0, min(1.0, rank_ratio * float(config.rank_scale)))
    min_ratio = max(0.01, min(1.0, float(config.min_keep_ratio)))
    max_ratio = max(min_ratio, min(1.0, float(config.max_keep_ratio)))
    adaptive_ratio = min_ratio + (max_ratio - min_ratio) * rank_ratio
    raw_keep = int(round(float(current_visual_tokens) * adaptive_ratio))

    score = torch.nan_to_num(scores[0].float(), nan=0.0, posinf=0.0, neginf=0.0)
    score_sum = float(torch.clamp(score.sum(), min=0.0).item())
    if score_sum > 1e-9:
        probs = torch.clamp(score, min=0.0) / score_sum
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))
        attention_rank = float(torch.exp(entropy).item())
        raw_keep = max(raw_keep, int(round(attention_rank)))
    else:
        attention_rank = 1.0

    # Budget-constrained projection requested for Mantis sweeps: SparseVLM is
    # adaptive, but this cap makes final retained visual tokens comparable with
    # fixed-budget baselines without pretending the intermediate rule is fixed.
    projected_cap: Optional[int] = None
    if config.budget_projection and int(config.target_token_num) > 0:
        final_budget = max(1, min(int(config.target_token_num), int(current_visual_tokens)))
        remaining = max(1, int(remaining_layers))
        if current_visual_tokens > final_budget:
            projected_cap = int(math.ceil(final_budget * ((current_visual_tokens / final_budget) ** ((remaining - 1) / remaining))))
            raw_keep = min(raw_keep, projected_cap)

    keep_count = max(1, min(int(current_visual_tokens), int(raw_keep)))
    return keep_count, {
        "effective_rank": float(rank.item()),
        "rank_ratio": float(rank_ratio),
        "attention_effective_rank": float(attention_rank),
        "adaptive_keep_ratio": float(adaptive_ratio),
        "projected_cap": projected_cap,
    }


def _cluster_and_merge_tokens(tokens: torch.Tensor, cluster_num: int) -> torch.Tensor:
    batch_size, num_tokens, channels = tokens.shape
    cluster_num = max(1, min(int(cluster_num), int(num_tokens)))
    if num_tokens <= cluster_num:
        return tokens

    # SparseVLM token recycling follows density-peak clustering: rescue the
    # most valuable dropped tokens, form compact groups, and average each group
    # into one replacement token.
    x1 = tokens.unsqueeze(2)
    x2 = tokens.unsqueeze(1)
    dist_matrix = torch.norm((x1 - x2).float(), dim=-1, p=2) / math.sqrt(max(1, channels))
    k = max(1, min(cluster_num, num_tokens))
    dist_nearest, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
    density_mask = (density[:, None, :] > density[:, :, None]).to(dtype=dist_matrix.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist_to_higher, _ = (dist_matrix * density_mask + dist_max * (1.0 - density_mask)).min(dim=-1)
    center_score = dist_to_higher * density
    _, centers = torch.topk(center_score, k=cluster_num, dim=-1)

    center_tokens = tokens.gather(1, centers.unsqueeze(-1).expand(-1, -1, channels))
    center_dist = torch.cdist(tokens.float(), center_tokens.float())
    assignments = center_dist.argmin(dim=-1)
    merged = tokens.new_zeros(batch_size, cluster_num, channels)
    counts = tokens.new_zeros(batch_size, cluster_num, 1)
    merged.scatter_add_(1, assignments.unsqueeze(-1).expand(-1, -1, channels), tokens)
    counts.scatter_add_(1, assignments.unsqueeze(-1), torch.ones(batch_size, num_tokens, 1, device=tokens.device, dtype=tokens.dtype))
    return merged / counts.clamp_min(1e-6)


def select_and_recycle_visual_tokens(
    outputs_hidden: torch.Tensor,
    visual_positions: torch.Tensor,
    scores: torch.Tensor,
    keep_count: int,
    config: MantisSparseVLMConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    batch_size = outputs_hidden.shape[0]
    device = outputs_hidden.device
    current_visual_tokens = int(visual_positions.numel())
    keep_count = max(1, min(int(keep_count), current_visual_tokens))

    _, topk = torch.topk(scores, keep_count, dim=1)
    base = visual_positions.unsqueeze(0).expand(batch_size, -1)
    selected_positions = torch.sort(base.gather(1, topk), dim=1).values

    dropped_mask = torch.ones((batch_size, current_visual_tokens), dtype=torch.bool, device=device)
    dropped_mask.scatter_(1, topk, False)
    dropped_count = int(dropped_mask[0].sum().item()) if batch_size > 0 else 0
    if config.disable_recycling or dropped_count <= 0:
        empty = outputs_hidden.new_empty((batch_size, 0, outputs_hidden.shape[-1]))
        return selected_positions, empty, {"recycled_tokens": 0, "recycle_source_tokens": 0}

    # SparseVLM Sec. 3.5: recycle a valuable subset from the pruned pool instead
    # of discarding all dropped information. We choose the highest-scoring
    # dropped tokens, then compress roughly every N of them into one cluster.
    dropped_scores = scores.masked_fill(~dropped_mask, -float("inf"))
    recycle_source_count = int(round(dropped_count * float(config.recycle_ratio)))
    recycle_source_count = max(int(config.min_recycle_tokens), recycle_source_count)
    recycle_source_count = min(dropped_count, recycle_source_count)
    if recycle_source_count <= 0:
        empty = outputs_hidden.new_empty((batch_size, 0, outputs_hidden.shape[-1]))
        return selected_positions, empty, {"recycled_tokens": 0, "recycle_source_tokens": 0}

    _, recycle_local = torch.topk(dropped_scores, recycle_source_count, dim=1)
    recycle_positions = base.gather(1, recycle_local)
    recycle_tokens = outputs_hidden.gather(
        1,
        recycle_positions.unsqueeze(-1).expand(-1, -1, outputs_hidden.shape[-1]),
    )
    cluster_num = int(math.ceil(recycle_source_count / max(1, int(config.recycle_cluster_size))))
    recycled_tokens = _cluster_and_merge_tokens(recycle_tokens, cluster_num)
    return selected_positions, recycled_tokens, {
        "recycled_tokens": int(recycled_tokens.shape[1]),
        "recycle_source_tokens": int(recycle_source_count),
    }


def build_sparsevlm_sequence(
    outputs_hidden: torch.Tensor,
    always_keep_positions: torch.Tensor,
    selected_visual_positions: torch.Tensor,
    recycled_tokens: torch.Tensor,
    *,
    seq_len: int,
    visual_insert_position: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, hidden_size = outputs_hidden.shape
    keep_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=outputs_hidden.device)
    if always_keep_positions.numel() > 0:
        keep_mask[:, always_keep_positions] = True
    if selected_visual_positions.numel() > 0:
        keep_mask.scatter_(1, selected_visual_positions, True)
    base = torch.arange(seq_len, device=outputs_hidden.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    full_indices = base[keep_mask].view(batch_size, -1)
    gathered = outputs_hidden.gather(1, full_indices.unsqueeze(-1).expand(-1, -1, hidden_size)).contiguous()
    if recycled_tokens.numel() <= 0:
        return gathered, full_indices
    insert_at = int((full_indices[0] < int(visual_insert_position)).sum().item())
    new_hidden = torch.cat([gathered[:, :insert_at], recycled_tokens, gathered[:, insert_at:]], dim=1).contiguous()
    return new_hidden, full_indices


def estimate_sparsevlm_flops_reduction(
    *,
    config: Any,
    num_layers: int,
    layer_records: list,
    prefill_seq_len: Optional[int],
) -> Optional[float]:
    if not layer_records or prefill_seq_len is None:
        return None
    try:
        baseline_tokens = int(prefill_seq_len)
        total_baseline = 0.0
        total_sparse = 0.0
        current_tokens = baseline_tokens
        record_by_layer = {int(record["layer_idx"]): record for record in layer_records}
        for layer_idx in range(int(num_layers)):
            before = current_tokens
            if layer_idx in record_by_layer:
                before = int(record_by_layer[layer_idx]["tokens_before"])
                after = int(record_by_layer[layer_idx]["tokens_after"])
            else:
                after = before
            layer_reduction = estimate_fastv_flops_reduction(
                config=config,
                num_layers=1,
                start_layer=0,
                tokens_before=baseline_tokens,
                tokens_after=after,
            )
            layer_cost_ratio = 1.0 - float(layer_reduction or 0.0)
            total_baseline += 1.0
            total_sparse += layer_cost_ratio
            current_tokens = after
        return float(max(0.0, min(1.0, 1.0 - total_sparse / max(1e-9, total_baseline))))
    except Exception:
        return None


def record_sparsevlm_stats(
    runtime_stats: Dict[str, Any],
    *,
    config: MantisSparseVLMConfig,
    layer_idx: int,
    visual_before: int,
    visual_after: int,
    keep_count: int,
    recycle_stats: Dict[str, Any],
    rank_stats: Dict[str, Any],
    budget_elapsed_t0: float,
) -> None:
    runtime_stats["budget_schedule_seconds"] = float(
        runtime_stats.get("budget_schedule_seconds", 0.0) + max(0.0, time.perf_counter() - budget_elapsed_t0)
    )
    runtime_stats["configured_visual_budget_cap"] = int(config.target_token_num)
    runtime_stats["raw_total_budget"] = float(keep_count)
    runtime_stats["total_budget"] = int(visual_after)
    runtime_stats["target_keep_per_view"] = [int(visual_after)]
    runtime_stats["selected_pruning_mode"] = "sparsevlm_progressive_unified_visual"
    runtime_stats.setdefault("sparsevlm_layer_records", [])
    runtime_stats["sparsevlm_layer_records"].append(
        {
            "layer_idx": int(layer_idx),
            "visual_before": int(visual_before),
            "visual_kept": int(keep_count),
            "visual_after": int(visual_after),
            "recycled_tokens": int(recycle_stats.get("recycled_tokens", 0) or 0),
            "recycle_source_tokens": int(recycle_stats.get("recycle_source_tokens", 0) or 0),
            **rank_stats,
        }
    )


__all__ = [
    "build_sparsevlm_sequence",
    "decide_sparsevlm_keep_count",
    "estimate_sparsevlm_flops_reduction",
    "record_sparsevlm_stats",
    "score_visual_tokens_by_text_raters",
    "select_and_recycle_visual_tokens",
    "select_visual_relevant_text_tokens",
]
