import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def _parse_env_flag(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_env_int(name: str, default: int) -> int:
    value = os.environ.get(name, None)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_env_float(name: str, default: float) -> float:
    value = os.environ.get(name, None)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_env_int_tuple(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    value = os.environ.get(name, None)
    if value is None:
        return default
    parsed = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            parsed.append(int(chunk))
        except ValueError:
            continue
    if not parsed:
        return default
    return tuple(parsed)


@dataclass
class MantisAutoPruneConfig:
    enabled: bool = False
    start_layer: int = 0
    max_pruning_layers: int = 6
    enable_adaptive_layer_policy: bool = False
    enable_oneshot_fallback: bool = False
    adaptive_layer_choices: Tuple[int, ...] = (1, 3, 6)
    adaptive_layer_mu: float = 4.0
    adaptive_layer_tau: float = 0.5
    oneshot_fallback_max_layers: int = 3
    oneshot_fallback_budget_margin: float = 0.15
    oneshot_fallback_frontload_ratio: float = 0.5
    scaffold_only: bool = False
    input_control_only: bool = False
    target_token_num: int = 128
    budget_mode: str = "absolute"
    absolute_auto_bounds: bool = True
    absolute_anchor_mode: str = "none"
    absolute_budget_scale: float = 1.0
    b_min: int = 64
    b_max: int = 128
    mu: float = 4.0
    tau: float = 0.5
    coverage_min: float = 0.82
    coverage_max: float = 0.95
    x0: float = 12.0
    k_shape: float = 0.3
    min_keep_per_view: int = 24
    allocation_temp: float = 0.4
    enable_image_prior: bool = False
    image_prior_weight: float = 1.0
    warm_keep_factor: float = 2.5
    monitor_only: bool = False
    disable_token_prune: bool = False
    disable_progressive_prune: bool = False
    disable_kv_cache: bool = False
    debug: bool = False


class AdaptiveBudgetStrategy:
    def __init__(self, b_min: int, b_max: int, mu: float, tau: float):
        self.b_min = int(b_min)
        self.b_max = int(b_max)
        self.mu = float(mu)
        self.tau = float(tau)

    def decide_budget(self, view_nmis: torch.Tensor, effective_rank: torch.Tensor) -> torch.Tensor:
        signal = compute_complexity_signal(view_nmis, effective_rank, default=self.mu)
        value = torch.clamp((signal - self.mu) / max(self.tau, 1e-6), min=-20.0, max=20.0)
        budget = self.b_min + (self.b_max - self.b_min) * torch.sigmoid(value)
        return torch.nan_to_num(budget, nan=float(self.b_min), posinf=float(self.b_max), neginf=float(self.b_min))


def compute_complexity_signal(
    view_nmis: torch.Tensor,
    effective_rank: torch.Tensor,
    *,
    default: float,
) -> torch.Tensor:
    view_nmis = torch.nan_to_num(view_nmis, nan=1e-6, posinf=1e6, neginf=1e-6)
    effective_rank = torch.nan_to_num(effective_rank, nan=1.0, posinf=1e6, neginf=1.0)
    nmi_avg = torch.clamp(view_nmis.mean(), min=1e-6)
    return torch.nan_to_num(effective_rank / nmi_avg, nan=float(default), posinf=1e6, neginf=0.0)


def logistic_complexity_score(signal: torch.Tensor, mu: float, tau: float) -> torch.Tensor:
    value = torch.clamp((signal - float(mu)) / max(float(tau), 1e-6), min=-20.0, max=20.0)
    return torch.sigmoid(value)


def calculate_mi_and_kl(attn: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    attn = torch.nan_to_num(attn.float(), nan=0.0, posinf=0.0, neginf=0.0)
    num_text = attn.shape[-2]
    num_vision = attn.shape[-1]

    p_x = 1.0 / max(1, num_text)
    p_xy = attn * p_x
    p_y = p_xy.sum(dim=-2)

    ratio = attn / (p_y.unsqueeze(-2) + eps)
    mi = (p_xy * torch.log2(torch.clamp(ratio, min=eps))).sum(dim=(-2, -1))

    uniform = 1.0 / max(1, num_vision)
    kl = (p_y * torch.log2(torch.clamp(p_y / uniform, min=eps))).sum(dim=-1)
    return (
        torch.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0),
    )


def resolve_mantis_attn_implementation(
    requested: Optional[str],
    autoprune_config: Optional[MantisAutoPruneConfig],
) -> Optional[str]:
    if requested is not None:
        return requested
    if autoprune_config is not None and autoprune_config.enabled:
        return "sdpa"
    return None


def build_mantis_autoprune_config_from_env(prefix: str = "MANTIS_AUTOPRUNE_") -> MantisAutoPruneConfig:
    input_control_only = _parse_env_flag(os.environ.get(f"{prefix}INPUT_CONTROL_ONLY", None), default=False)
    enabled = _parse_env_flag(os.environ.get(f"{prefix}ENABLED", None), default=False) or input_control_only
    return MantisAutoPruneConfig(
        enabled=enabled,
        start_layer=_parse_env_int(f"{prefix}START_LAYER", 0),
        max_pruning_layers=_parse_env_int(f"{prefix}MAX_PRUNING_LAYERS", 6),
        enable_adaptive_layer_policy=_parse_env_flag(
            os.environ.get(f"{prefix}ENABLE_ADAPTIVE_LAYER_POLICY", None),
            default=False,
        ),
        enable_oneshot_fallback=_parse_env_flag(
            os.environ.get(f"{prefix}ENABLE_ONESHOT_FALLBACK", None),
            default=False,
        ),
        adaptive_layer_choices=_parse_env_int_tuple(f"{prefix}ADAPTIVE_LAYER_CHOICES", (1, 3, 6)),
        adaptive_layer_mu=_parse_env_float(f"{prefix}ADAPTIVE_LAYER_MU", 4.0),
        adaptive_layer_tau=_parse_env_float(f"{prefix}ADAPTIVE_LAYER_TAU", 0.5),
        oneshot_fallback_max_layers=_parse_env_int(f"{prefix}ONESHOT_FALLBACK_MAX_LAYERS", 3),
        oneshot_fallback_budget_margin=_parse_env_float(f"{prefix}ONESHOT_FALLBACK_BUDGET_MARGIN", 0.15),
        oneshot_fallback_frontload_ratio=_parse_env_float(f"{prefix}ONESHOT_FALLBACK_FRONTLOAD_RATIO", 0.5),
        scaffold_only=_parse_env_flag(os.environ.get(f"{prefix}SCAFFOLD_ONLY", None), default=False),
        input_control_only=input_control_only,
        target_token_num=_parse_env_int(f"{prefix}TARGET_TOKEN_NUM", 128),
        budget_mode=(os.environ.get(f"{prefix}BUDGET_MODE", "absolute").strip().lower() or "absolute"),
        absolute_auto_bounds=_parse_env_flag(
            os.environ.get(f"{prefix}ABSOLUTE_AUTO_BOUNDS", None),
            default=True,
        ),
        absolute_anchor_mode=(os.environ.get(f"{prefix}ABSOLUTE_ANCHOR_MODE", "none").strip().lower() or "none"),
        absolute_budget_scale=_parse_env_float(f"{prefix}ABSOLUTE_BUDGET_SCALE", 1.0),
        b_min=_parse_env_int(f"{prefix}B_MIN", 64),
        b_max=_parse_env_int(f"{prefix}B_MAX", 128),
        mu=_parse_env_float(f"{prefix}MU", 4.0),
        tau=_parse_env_float(f"{prefix}TAU", 0.5),
        coverage_min=_parse_env_float(f"{prefix}COVERAGE_MIN", 0.82),
        coverage_max=_parse_env_float(f"{prefix}COVERAGE_MAX", 0.95),
        x0=_parse_env_float(f"{prefix}X0", 12.0),
        k_shape=_parse_env_float(f"{prefix}K_SHAPE", 0.3),
        min_keep_per_view=_parse_env_int(f"{prefix}MIN_KEEP_PER_VIEW", 24),
        allocation_temp=_parse_env_float(f"{prefix}ALLOCATION_TEMP", 0.4),
        enable_image_prior=_parse_env_flag(os.environ.get(f"{prefix}ENABLE_IMAGE_PRIOR", None), default=False),
        image_prior_weight=_parse_env_float(f"{prefix}IMAGE_PRIOR_WEIGHT", 1.0),
        warm_keep_factor=_parse_env_float(f"{prefix}WARM_KEEP_FACTOR", 2.5),
        monitor_only=_parse_env_flag(os.environ.get(f"{prefix}MONITOR_ONLY", None), default=False),
        disable_token_prune=_parse_env_flag(os.environ.get(f"{prefix}DISABLE_TOKEN_PRUNE", None), default=False),
        disable_progressive_prune=_parse_env_flag(
            os.environ.get(f"{prefix}DISABLE_PROGRESSIVE_PRUNE", None),
            default=False,
        ),
        disable_kv_cache=not _parse_env_flag(os.environ.get(f"{prefix}KEEP_KV_CACHE", None), default=False),
        debug=_parse_env_flag(os.environ.get(f"{prefix}DEBUG", None), default=False),
    )


def estimate_mantis_multiview_token_budget(model_or_config, num_views: int = 1) -> Dict[str, int]:
    config = getattr(model_or_config, "config", model_or_config)
    vision_config = getattr(config, "vision_config", None)
    image_size = int(getattr(vision_config, "image_size", 0) or 0)
    patch_size = int(getattr(vision_config, "patch_size", 1) or 1)
    patch_side = max(1, image_size // max(1, patch_size))
    visual_tokens_per_view = int(patch_side * patch_side)

    total_views = max(1, int(num_views))
    total_visual_tokens = int(total_views * visual_tokens_per_view)

    return {
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "patch_side": int(patch_side),
        "visual_tokens_per_view": int(visual_tokens_per_view),
        "views": int(total_views),
        "total_visual_tokens": int(total_visual_tokens),
        "net_seq_growth_per_view": int(max(0, visual_tokens_per_view - 1)),
        "total_net_seq_growth": int(max(0, total_visual_tokens - total_views)),
    }


__all__ = [
    "AdaptiveBudgetStrategy",
    "MantisAutoPruneConfig",
    "build_mantis_autoprune_config_from_env",
    "calculate_mi_and_kl",
    "compute_complexity_signal",
    "estimate_mantis_multiview_token_budget",
    "logistic_complexity_score",
    "resolve_mantis_attn_implementation",
]
