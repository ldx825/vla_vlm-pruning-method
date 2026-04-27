import time
from typing import Any, Dict, Optional

import torch


def select_fastv_unified_visual_tokens(
    attn_weights: torch.Tensor,
    visual_positions: torch.Tensor,
    keep_visual_tokens: int,
    batch_size: int,
    device: torch.device,
    query_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if visual_positions.numel() <= 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)

    keep_visual_tokens = max(1, min(int(keep_visual_tokens), int(visual_positions.numel())))

    # FastV paper Sec. 4.1: rank visual tokens by the average attention score
    # they receive at layer K, then keep the top visual tokens for all later
    # layers. For the BLINK/VLA baseline requested here, every image patch from
    # every view is one unified visual pool rather than a per-view budget.
    visual_key_attn = attn_weights.index_select(3, visual_positions).mean(dim=1)

    # Paper-aligned importance signal, adapted to Mantis action decoding:
    # a visual token's score aggregates attention from all other tokens in the
    # current sequence, so answer/action tokens and non-selected visual tokens
    # both contribute. The self row is excluded to avoid a token being kept only
    # because it attends to itself under the causal mask.
    if query_positions is None:
        query_positions = torch.arange(attn_weights.shape[2], dtype=torch.long, device=device)
    else:
        query_positions = query_positions.to(device=device, dtype=torch.long)
    if int(query_positions.numel()) == int(visual_key_attn.shape[1]):
        other_token_mask = query_positions.unsqueeze(1) != visual_positions.to(device=device).unsqueeze(0)
        other_token_weight = other_token_mask.to(dtype=visual_key_attn.dtype)
        denominator = other_token_weight.sum(dim=0).clamp_min(1.0)
        score = (visual_key_attn * other_token_weight.unsqueeze(0)).sum(dim=1) / denominator.unsqueeze(0)
    else:
        score = visual_key_attn.mean(dim=1)
    score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)

    _, topk = torch.topk(score, keep_visual_tokens, dim=1)
    base = visual_positions.unsqueeze(0).expand(batch_size, -1)
    selected = base.gather(1, topk)
    return torch.sort(selected, dim=1).values


def estimate_fastv_flops_reduction(
    *,
    config: Any,
    num_layers: int,
    start_layer: int,
    tokens_before: int,
    tokens_after: int,
) -> Optional[float]:
    try:
        language_config = getattr(config, "text_config", None) or getattr(config, "language_config", None) or config
        hidden_size = int(getattr(language_config, "hidden_size", 0) or 0)
        intermediate_size = int(getattr(language_config, "intermediate_size", 0) or 0)
        if hidden_size <= 0 or intermediate_size <= 0:
            return None

        n = float(max(1, int(tokens_before)))
        n_hat = float(max(1, int(tokens_after)))
        total_layers = float(max(1, int(num_layers)))
        # The patch prunes after the selected decoder layer has already run.
        # Therefore layers [0, start_layer] use the original sequence length,
        # and the reduced sequence starts at the following layer.
        k = float(max(0, min(int(start_layer) + 1, int(num_layers))))

        # FastV paper Eq. 5: MHA + FFN layer cost is
        # 4*n*d^2 + 2*n^2*d + 2*n*d*m, before/after the pruning layer K.
        def layer_cost(seq_len: float) -> float:
            d = float(hidden_size)
            m = float(intermediate_size)
            return 4.0 * seq_len * d * d + 2.0 * seq_len * seq_len * d + 2.0 * seq_len * d * m

        baseline = total_layers * layer_cost(n)
        fastv_cost = k * layer_cost(n) + max(0.0, total_layers - k) * layer_cost(n_hat)
        if baseline <= 0.0:
            return None
        return float(max(0.0, min(1.0, 1.0 - fastv_cost / baseline)))
    except Exception:
        return None


def record_fastv_schedule_stats(
    runtime_stats: Dict[str, Any],
    *,
    start_layer: int,
    visual_tokens_before: int,
    keep_visual_tokens: int,
    budget_elapsed: float,
) -> None:
    runtime_stats["configured_visual_budget_cap"] = int(keep_visual_tokens)
    runtime_stats["raw_total_budget"] = float(keep_visual_tokens)
    runtime_stats["total_budget"] = int(keep_visual_tokens)
    runtime_stats["target_keep_per_view"] = [int(keep_visual_tokens)]
    runtime_stats["keep_count_schedule"] = [[int(keep_visual_tokens)]]
    runtime_stats["pruning_layer_indices"] = [int(start_layer)]
    runtime_stats["candidate_pruning_layer_indices"] = [int(start_layer)]
    runtime_stats["active_pruning_layer_indices"] = [int(start_layer)]
    runtime_stats["selected_pruning_mode"] = "fastv_oneshot_unified_visual"
    runtime_stats["dense_schedule_length"] = 1
    runtime_stats["compressed_schedule_length"] = 1
    runtime_stats["dense_pruning_layer_indices"] = [int(start_layer)]
    runtime_stats["dense_keep_count_schedule"] = [[int(keep_visual_tokens)]]
    runtime_stats["fastv_visual_tokens_before"] = int(visual_tokens_before)
    runtime_stats["fastv_visual_tokens_after"] = int(keep_visual_tokens)
    runtime_stats["budget_schedule_seconds"] = float(
        runtime_stats.get("budget_schedule_seconds", 0.0) + max(0.0, time.perf_counter() - budget_elapsed)
    )


__all__ = [
    "estimate_fastv_flops_reduction",
    "record_fastv_schedule_stats",
    "select_fastv_unified_visual_tokens",
]
