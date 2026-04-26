from typing import Any, Dict, Optional, Tuple

import torch

from .fastv_mantis_pruning import estimate_fastv_flops_reduction


def split_visionzip_budget(
    target_token_num: int,
    visual_tokens_before: int,
    dominant_ratio: float = 0.7,
) -> Tuple[int, int, int]:
    target = max(1, min(int(target_token_num), int(visual_tokens_before)))
    if target >= int(visual_tokens_before):
        return int(visual_tokens_before), 0, int(visual_tokens_before)

    ratio = max(0.0, min(1.0, float(dominant_ratio)))
    dominant = int(round(target * ratio))
    dominant = max(1, min(dominant, target))
    contextual = max(0, int(target - dominant))
    return int(dominant), int(contextual), int(target)


def _score_dominant_tokens(
    vision_attn: Optional[torch.Tensor],
    image_features: torch.Tensor,
    visual_tokens_before: int,
) -> torch.Tensor:
    batch_images, _, _ = image_features.shape
    if vision_attn is not None and vision_attn.ndim == 4:
        key_len = int(vision_attn.shape[-1])
        # VisionZip Step 2: use the final visual-layer [CLS] -> patch attention
        # as the text-agnostic token importance signal. In Mantis default mode
        # CLS is already removed before the projector, so patch columns start at
        # 1 when the attention matrix still contains CLS.
        if key_len >= visual_tokens_before + 1:
            cls_to_patch = vision_attn[:, :, 0, 1 : 1 + visual_tokens_before]
        elif key_len >= visual_tokens_before:
            cls_to_patch = vision_attn[:, :, 0, :visual_tokens_before]
        else:
            cls_to_patch = None
        if cls_to_patch is not None:
            scores = torch.nan_to_num(cls_to_patch.mean(dim=1).float(), nan=0.0, posinf=0.0, neginf=0.0)
            if scores.shape[0] == 1 and batch_images > 1:
                scores = scores.expand(batch_images, -1)
            if scores.shape[0] == batch_images:
                return scores

    # Fallback keeps the method physically pruning even if a specific vision
    # backbone does not expose attentions. It ranks by feature norm, while the
    # contextual merge below still follows VisionZip's token compression step.
    return torch.nan_to_num(image_features.float().norm(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)


def build_visionzip_features(
    image_features: torch.Tensor,
    vision_attn: Optional[torch.Tensor],
    *,
    target_token_num: int,
    dominant_ratio: float,
    token_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if image_features.ndim != 3:
        return image_features, {"pruned": False}

    batch_images, visual_tokens_before, hidden_size = image_features.shape
    dominant_count, contextual_count, target = split_visionzip_budget(
        target_token_num,
        visual_tokens_before,
        dominant_ratio,
    )
    if target >= visual_tokens_before:
        return image_features, {
            "pruned": False,
            "dominant_count": int(visual_tokens_before),
            "contextual_count": 0,
            "target": int(visual_tokens_before),
        }

    if token_scores is not None and token_scores.shape[:2] == image_features.shape[:2]:
        scores = torch.nan_to_num(token_scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        scores = _score_dominant_tokens(vision_attn, image_features, visual_tokens_before)
    dominant_idx = torch.topk(scores, dominant_count, dim=1).indices
    dominant_idx = torch.sort(dominant_idx.to(device=image_features.device, dtype=torch.long), dim=1).values

    all_idx = torch.arange(visual_tokens_before, device=image_features.device).unsqueeze(0).expand(batch_images, -1)
    dominant_mask = torch.zeros((batch_images, visual_tokens_before), dtype=torch.bool, device=image_features.device)
    dominant_mask.scatter_(1, dominant_idx, True)
    non_dominant_idx = all_idx[~dominant_mask].view(batch_images, visual_tokens_before - dominant_count)

    # VisionZip Step 3: keep dominant tokens unchanged.
    dominant_tokens = image_features.gather(
        1,
        dominant_idx.unsqueeze(-1).expand(-1, -1, hidden_size),
    )

    if contextual_count <= 0 or non_dominant_idx.shape[1] <= 0:
        return dominant_tokens.contiguous(), {
            "pruned": True,
            "dominant_count": int(dominant_count),
            "contextual_count": 0,
            "target": int(dominant_count),
        }

    non_dominant_tokens = image_features.gather(
        1,
        non_dominant_idx.unsqueeze(-1).expand(-1, -1, hidden_size),
    )
    contextual_count = min(int(contextual_count), int(non_dominant_tokens.shape[1]))

    # VisionZip Steps 4-5: do not discard all remaining tokens. Select a small
    # set of contextual anchors, then merge each discarded token into its most
    # similar contextual anchor with a mean aggregation. The paper uses visual
    # key similarity; at this hook point Mantis exposes the selected visual
    # hidden states, which serve as the key-space proxy after vision encoding.
    anchor_scores = scores.gather(1, non_dominant_idx)
    context_local_idx = torch.topk(anchor_scores, contextual_count, dim=1).indices
    context_local_idx = torch.sort(context_local_idx.to(dtype=torch.long), dim=1).values
    contextual_tokens = non_dominant_tokens.gather(
        1,
        context_local_idx.unsqueeze(-1).expand(-1, -1, hidden_size),
    ).clone()

    if non_dominant_tokens.shape[1] > contextual_count:
        normalized_rest = torch.nn.functional.normalize(non_dominant_tokens.float(), dim=-1)
        normalized_context = torch.nn.functional.normalize(contextual_tokens.float(), dim=-1)
        assign = torch.matmul(normalized_rest, normalized_context.transpose(1, 2)).argmax(dim=-1)
        contextual_sum = torch.zeros_like(contextual_tokens.float())
        contextual_count_tensor = torch.zeros(
            (batch_images, contextual_count, 1),
            dtype=torch.float32,
            device=image_features.device,
        )
        contextual_sum.scatter_add_(
            1,
            assign.unsqueeze(-1).expand(-1, -1, hidden_size),
            non_dominant_tokens.float(),
        )
        contextual_count_tensor.scatter_add_(
            1,
            assign.unsqueeze(-1),
            torch.ones((batch_images, non_dominant_tokens.shape[1], 1), dtype=torch.float32, device=image_features.device),
        )
        contextual_tokens = (contextual_sum / contextual_count_tensor.clamp_min(1.0)).to(dtype=image_features.dtype)

    pruned_features = torch.cat([dominant_tokens, contextual_tokens], dim=1).contiguous()
    return pruned_features, {
        "pruned": True,
        "dominant_count": int(dominant_count),
        "contextual_count": int(contextual_count),
        "target": int(pruned_features.shape[1]),
    }


def record_visionzip_stats(
    runtime_stats: Dict[str, Any],
    *,
    visual_tokens_before: int,
    visual_tokens_after: int,
    dominant_count: int,
    contextual_count: int,
) -> None:
    runtime_stats["configured_visual_budget_cap"] = int(visual_tokens_after)
    runtime_stats["raw_total_budget"] = float(visual_tokens_after)
    runtime_stats["total_budget"] = int(visual_tokens_after)
    runtime_stats["target_keep_per_view"] = [int(visual_tokens_after)]
    runtime_stats["keep_count_schedule"] = [[int(visual_tokens_after)]]
    runtime_stats["dense_keep_count_schedule"] = [[int(visual_tokens_after)]]
    runtime_stats["dense_schedule_length"] = 1
    runtime_stats["compressed_schedule_length"] = 1
    runtime_stats["dense_pruning_layer_indices"] = [0]
    runtime_stats["selected_pruning_mode"] = "visionzip_dominant_contextual_unified"
    runtime_stats["visionzip_visual_tokens_before"] = int(visual_tokens_before)
    runtime_stats["visionzip_visual_tokens_after"] = int(visual_tokens_after)
    runtime_stats["visionzip_dominant_tokens"] = int(dominant_count)
    runtime_stats["visionzip_contextual_tokens"] = int(contextual_count)
    runtime_stats["visionzip_formula_final_keep_k_m_plus_cls"] = int(dominant_count + contextual_count + 1)


__all__ = [
    "build_visionzip_features",
    "estimate_fastv_flops_reduction",
    "record_visionzip_stats",
    "split_visionzip_budget",
]
