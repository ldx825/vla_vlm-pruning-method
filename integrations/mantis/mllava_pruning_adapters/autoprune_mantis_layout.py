from typing import Any, Dict, List, Optional

import torch


def build_runtime_stats() -> Dict[str, Any]:
    return {
        "prefill_seq_len": None,
        "layout_ready": 0,
        "layout_query_tokens": 0,
        "layout_num_views": 0,
        "layout_view_lengths": [],
        "layout_has_image_prior": 0,
        "layout_cache_hits": 0,
        "layout_recompute_count": 0,
        "configured_max_pruning_layers": None,
        "pruning_layer_indices": [],
        "candidate_pruning_layer_indices": [],
        "active_pruning_layer_indices": [],
        "adaptive_layer_policy_enabled": 0,
        "adaptive_layer_choices": [],
        "adaptive_layer_count": None,
        "dense_schedule_length": None,
        "compressed_schedule_length": None,
        "dense_pruning_layer_indices": [],
        "dense_keep_count_schedule": [],
        "selected_pruning_mode": None,
        "oneshot_fallback_used": 0,
        "complexity_signal": None,
        "complexity_score": None,
        "nmi_per_view": [],
        "kl_per_view": [],
        "effective_rank": None,
        "invalid_metric_detected": 0,
        "budget_fallback_used": 0,
        "budget_clamped_by_target": 0,
        "configured_visual_budget_cap": None,
        "raw_total_budget": None,
        "total_budget": None,
        "target_keep_per_view": [],
        "keep_count_schedule": [],
        "pruned_layers": 0,
        "last_tokens_before": None,
        "last_tokens_after": None,
        "patch_total_seconds": 0.0,
        "patch_overhead_seconds": 0.0,
        "layer_forward_seconds": 0.0,
        "alignment_seconds": 0.0,
        "layout_seconds": 0.0,
        "probe_seconds": 0.0,
        "metric_seconds": 0.0,
        "rank_seconds": 0.0,
        "budget_schedule_seconds": 0.0,
        "topk_seconds": 0.0,
        "gather_seconds": 0.0,
        "layer_call_count": 0,
        "prefill_layer_calls": 0,
        "decode_layer_calls": 0,
        "decode_fastpath_layer_calls": 0,
        "cache_enabled_layer_calls": 0,
        "seq_len_one_layer_calls": 0,
        "seq_len_gt_one_layer_calls": 0,
        "prefill_patch_total_seconds": 0.0,
        "decode_patch_total_seconds": 0.0,
        "prefill_patch_overhead_seconds": 0.0,
        "decode_patch_overhead_seconds": 0.0,
        "prefill_layer_forward_seconds": 0.0,
        "decode_layer_forward_seconds": 0.0,
        "generated_token_steps": 0,
        "max_seq_len_seen": 0,
    }


def prepare_input_layout_from_merge(
    input_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    image_token_index: int,
    visual_tokens_per_view: int,
) -> Dict[str, Any]:
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] < 1:
        return {}
    if visual_tokens_per_view <= 0:
        return {}

    row = input_ids[0]
    if attention_mask is not None:
        valid_mask = attention_mask[0].to(dtype=torch.bool)
        row = row[valid_mask]

    cursor = 0
    last_media_end = -1
    text_positions: List[int] = []
    view_indices: List[torch.Tensor] = []

    for token in row.tolist():
        if int(token) == int(image_token_index):
            positions = torch.arange(cursor, cursor + visual_tokens_per_view, dtype=torch.long)
            view_indices.append(positions)
            cursor += int(visual_tokens_per_view)
            last_media_end = int(cursor - 1)
        else:
            text_positions.append(int(cursor))
            cursor += 1

    if not view_indices or last_media_end < 0:
        return {}

    query_positions = [pos for pos in text_positions if pos > last_media_end]
    if not query_positions:
        return {}

    return {
        "ready": True,
        "view_indices": [indices.contiguous() for indices in view_indices],
        "view_priors": [None for _ in view_indices],
        "all_visual_indices": torch.cat(view_indices, dim=0).contiguous(),
        "query_positions": torch.tensor(query_positions, dtype=torch.long),
        "full_sequence_length": int(cursor),
    }


def summarize_layout(layout: Dict[str, Any]) -> Dict[str, Any]:
    if not layout.get("ready"):
        return {}
    view_indices = layout.get("view_indices", [])
    query_positions = layout.get("query_positions", torch.empty(0, dtype=torch.long))
    return {
        "layout_ready": 1,
        "layout_query_tokens": int(query_positions.numel()),
        "layout_num_views": int(len(view_indices)),
        "layout_view_lengths": [int(indices.numel()) for indices in view_indices],
        "layout_has_image_prior": 0,
    }


def locate_current_positions(
    global_indices: Optional[torch.Tensor],
    original_positions: Optional[torch.Tensor],
) -> torch.Tensor:
    if global_indices is None or original_positions is None or original_positions.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    current_globals = global_indices[0]
    original_positions = original_positions.to(device=current_globals.device, dtype=current_globals.dtype)
    mask = torch.isin(current_globals, original_positions)
    return torch.nonzero(mask, as_tuple=False).flatten()


def resolve_layout_state(
    input_layout: Dict[str, Any],
    global_indices: Optional[torch.Tensor],
    seq_len: int,
    device: torch.device,
) -> Dict[str, Any]:
    if not input_layout.get("ready"):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return {
            "query_positions": empty,
            "view_positions": [],
            "visual_positions": empty,
            "always_keep_positions": torch.arange(seq_len, device=device, dtype=torch.long),
        }

    query_positions = locate_current_positions(global_indices, input_layout["query_positions"]).to(device=device)
    visual_positions = locate_current_positions(global_indices, input_layout["all_visual_indices"]).to(device=device)
    view_positions = [
        locate_current_positions(global_indices, view_indices).to(device=device)
        for view_indices in input_layout["view_indices"]
    ]

    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    if visual_positions.numel() > 0:
        keep_mask[visual_positions] = False
    always_keep_positions = torch.nonzero(keep_mask, as_tuple=False).flatten()

    return {
        "query_positions": query_positions,
        "view_positions": view_positions,
        "view_prior_values": [None for _ in view_positions],
        "visual_positions": visual_positions,
        "always_keep_positions": always_keep_positions,
    }


__all__ = [
    "build_runtime_stats",
    "locate_current_positions",
    "prepare_input_layout_from_merge",
    "resolve_layout_state",
    "summarize_layout",
]
