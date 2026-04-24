from typing import Any, Dict, List, Optional

import torch


def build_runtime_stats() -> Dict[str, Any]:
    return {
        "prefill_seq_len": None,
        "layer_call_count": 0,
        "prefill_layer_calls": 0,
        "decode_layer_calls": 0,
        "seq_len_one_layer_calls": 0,
        "seq_len_gt_one_layer_calls": 0,
        "cache_enabled_layer_calls": 0,
        "decode_fastpath_layer_calls": 0,
        "generated_token_steps": 0,
        "patch_total_seconds": 0.0,
        "patch_overhead_seconds": 0.0,
        "layer_forward_seconds": 0.0,
        "prefill_patch_total_seconds": 0.0,
        "prefill_patch_overhead_seconds": 0.0,
        "prefill_layer_forward_seconds": 0.0,
        "decode_patch_total_seconds": 0.0,
        "decode_patch_overhead_seconds": 0.0,
        "decode_layer_forward_seconds": 0.0,
        "alignment_seconds": 0.0,
        "layout_seconds": 0.0,
        "layout_recompute_count": 0,
        "layout_cache_hits": 0,
        "metric_seconds": 0.0,
        "rank_seconds": 0.0,
        "budget_schedule_seconds": 0.0,
        "topk_seconds": 0.0,
        "gather_seconds": 0.0,
        "probe_seconds": 0.0,
        "pruned_layers": 0,
        "last_tokens_before": 0,
        "last_tokens_after": 0,
        "max_seq_len_seen": 0,
    }


def _compute_text_positions(input_ids: torch.Tensor, image_token_index: int, visual_tokens_per_view: int) -> torch.Tensor:
    special = input_ids == int(image_token_index)
    return torch.cumsum((special.to(torch.long) * (int(visual_tokens_per_view) - 1) + 1), dim=-1) - 1


def prepare_input_layout_from_merge(
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    image_token_index: int,
    visual_tokens_per_view: int,
) -> Dict[str, Any]:
    # Mirror Llava/Mantis _merge_input_ids_with_image_features so pruning
    # operates on real post-merge positions rather than guessed image spans.
    ids = input_ids[0]
    mask = attention_mask[0].bool() if attention_mask is not None else torch.ones_like(ids, dtype=torch.bool)
    text_positions = _compute_text_positions(input_ids[:1], image_token_index, visual_tokens_per_view)[0]
    max_embed_dim = int((ids == image_token_index).sum().item() * (visual_tokens_per_view - 1) + ids.numel())
    nb_image_pad = max_embed_dim - 1 - int(text_positions[-1].item())
    left_padding = not bool(torch.sum(input_ids[:, -1] == 0).item())
    if left_padding:
        text_positions = text_positions + nb_image_pad

    valid_text_positions = text_positions[(ids != image_token_index) & mask].to(torch.long)
    marker_positions = text_positions[(ids == image_token_index) & mask].to(torch.long)

    occupied = torch.zeros(max_embed_dim, dtype=torch.bool, device=input_ids.device)
    if valid_text_positions.numel() > 0:
        occupied[valid_text_positions] = True
    image_candidates = ~occupied
    image_candidates &= image_candidates.cumsum(-1) - 1 >= int(nb_image_pad)
    image_positions = torch.where(image_candidates)[0].to(torch.long)

    view_positions: List[torch.Tensor] = []
    offset = 0
    for _ in range(int(marker_positions.numel())):
        view_positions.append(image_positions[offset : offset + int(visual_tokens_per_view)])
        offset += int(visual_tokens_per_view)

    always_keep = torch.ones(max_embed_dim, dtype=torch.bool, device=input_ids.device)
    if image_positions.numel() > 0:
        always_keep[image_positions] = False

    return {
        "seq_len": int(max_embed_dim),
        "visual_tokens_per_view": int(visual_tokens_per_view),
        "view_positions": view_positions,
        "all_visual_positions": image_positions,
        "always_keep_positions": torch.where(always_keep)[0].to(torch.long),
        "query_positions": valid_text_positions,
        "num_views": int(len(view_positions)),
        "total_visual_tokens": int(image_positions.numel()),
    }


def resolve_layout_state(
    input_layout: Dict[str, Any],
    global_indices: Optional[torch.Tensor],
    seq_len: int,
    device: torch.device,
) -> Dict[str, Any]:
    if not input_layout:
        return {
            "view_positions": [],
            "all_visual_positions": torch.empty(0, dtype=torch.long, device=device),
            "always_keep_positions": torch.arange(seq_len, dtype=torch.long, device=device),
            "query_positions": torch.empty(0, dtype=torch.long, device=device),
        }

    if global_indices is None:
        reverse = torch.arange(seq_len, device=device, dtype=torch.long)
    else:
        current = global_indices[0].to(device=device, dtype=torch.long)
        reverse = torch.full((int(input_layout["seq_len"]),), -1, dtype=torch.long, device=device)
        reverse[current] = torch.arange(current.numel(), device=device, dtype=torch.long)

    def map_positions(pos: torch.Tensor) -> torch.Tensor:
        if pos.numel() == 0:
            return pos.to(device=device, dtype=torch.long)
        mapped = reverse[pos.to(device=device, dtype=torch.long)]
        return mapped[mapped >= 0].to(torch.long)

    view_positions = [map_positions(view) for view in input_layout.get("view_positions", [])]
    all_visual = map_positions(input_layout.get("all_visual_positions", torch.empty(0, dtype=torch.long)))
    always_keep = map_positions(input_layout.get("always_keep_positions", torch.empty(0, dtype=torch.long)))
    query_positions = map_positions(input_layout.get("query_positions", torch.empty(0, dtype=torch.long)))

    return {
        "view_positions": view_positions,
        "all_visual_positions": all_visual,
        "always_keep_positions": always_keep,
        "query_positions": query_positions,
    }


def summarize_layout(input_layout: Dict[str, Any]) -> Dict[str, Any]:
    view_positions = input_layout.get("view_positions", [])
    return {
        "num_views": int(len(view_positions)),
        "view_lengths": [int(view.numel()) for view in view_positions],
        "total_visual_tokens": int(input_layout.get("total_visual_tokens", 0) or 0),
    }


__all__ = [
    "build_runtime_stats",
    "prepare_input_layout_from_merge",
    "resolve_layout_state",
    "summarize_layout",
]
