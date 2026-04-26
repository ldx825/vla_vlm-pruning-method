from typing import Optional

import torch

from .fastv_mantis_pruning import estimate_fastv_flops_reduction


def select_fastervlm_visual_tokens(
    vision_attn: Optional[torch.Tensor],
    *,
    visual_tokens_before: int,
    keep_visual_tokens: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    scores = score_fastervlm_cls_to_patch_attention(
        vision_attn,
        visual_tokens_before=visual_tokens_before,
        device=device,
    )
    if scores is None:
        return None

    keep_visual_tokens = max(1, min(int(keep_visual_tokens), int(visual_tokens_before)))
    _, topk = torch.topk(scores, keep_visual_tokens, dim=1)
    return torch.sort(topk.to(device=device, dtype=torch.long), dim=1).values


def score_fastervlm_cls_to_patch_attention(
    vision_attn: Optional[torch.Tensor],
    *,
    visual_tokens_before: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if vision_attn is None or vision_attn.ndim != 4 or visual_tokens_before <= 0:
        return None

    key_len = int(vision_attn.shape[-1])

    # FasterVLM: rank image patches by final-layer visual encoder attention
    # from [CLS] to each patch. Mantis normally drops [CLS] before projection,
    # so when the attention matrix still contains CLS, patch columns start at 1.
    if key_len >= visual_tokens_before + 1:
        cls_to_patch = vision_attn[:, :, 0, 1 : 1 + visual_tokens_before]
    elif key_len >= visual_tokens_before:
        cls_to_patch = vision_attn[:, :, 0, :visual_tokens_before]
    else:
        return None

    return torch.nan_to_num(cls_to_patch.mean(dim=1).float(), nan=0.0, posinf=0.0, neginf=0.0).to(device=device)


def select_fastervlm_unified_visual_tokens(
    vision_attn: Optional[torch.Tensor],
    *,
    visual_tokens_before: int,
    keep_visual_tokens: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    scores = score_fastervlm_cls_to_patch_attention(
        vision_attn,
        visual_tokens_before=visual_tokens_before,
        device=device,
    )
    if scores is None:
        return None

    total_visual_tokens = int(scores.numel())
    if total_visual_tokens <= 0:
        return None

    keep_visual_tokens = max(1, min(int(keep_visual_tokens), total_visual_tokens))
    flat_scores = scores.reshape(1, total_visual_tokens)
    _, topk = torch.topk(flat_scores, keep_visual_tokens, dim=1)
    # Sort flattened image-major indices so the retained patches keep the same
    # sequence order as the original multi-image visual stream after global top-k.
    return torch.sort(topk.to(device=device, dtype=torch.long), dim=1).values


__all__ = [
    "estimate_fastv_flops_reduction",
    "score_fastervlm_cls_to_patch_attention",
    "select_fastervlm_unified_visual_tokens",
    "select_fastervlm_visual_tokens",
]
