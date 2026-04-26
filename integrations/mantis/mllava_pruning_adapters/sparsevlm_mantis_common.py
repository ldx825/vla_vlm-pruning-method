import os
from dataclasses import dataclass
from typing import Optional, Tuple


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
    return tuple(parsed) if parsed else default


@dataclass
class MantisSparseVLMConfig:
    enabled: bool = False
    start_layer: int = 2
    pruning_layers: Tuple[int, ...] = (2, 6, 15)
    target_token_num: int = 128
    min_keep_ratio: float = 0.20
    max_keep_ratio: float = 0.85
    rank_scale: float = 1.0
    scorer_threshold_scale: float = 1.0
    recycle_ratio: float = 0.30
    recycle_cluster_size: int = 10
    min_recycle_tokens: int = 1
    budget_projection: bool = True
    scaffold_only: bool = False
    disable_token_prune: bool = False
    disable_recycling: bool = False
    disable_kv_cache: bool = False
    debug: bool = False


def build_mantis_sparsevlm_config_from_env(prefix: str = "MANTIS_SPARSEVLM_") -> MantisSparseVLMConfig:
    return MantisSparseVLMConfig(
        enabled=_parse_env_flag(os.environ.get(f"{prefix}ENABLED", None), default=False),
        start_layer=_parse_env_int(f"{prefix}START_LAYER", 2),
        pruning_layers=_parse_env_int_tuple(f"{prefix}PRUNING_LAYERS", (2, 6, 15)),
        target_token_num=_parse_env_int(f"{prefix}TARGET_TOKEN_NUM", 128),
        min_keep_ratio=_parse_env_float(f"{prefix}MIN_KEEP_RATIO", 0.20),
        max_keep_ratio=_parse_env_float(f"{prefix}MAX_KEEP_RATIO", 0.85),
        rank_scale=_parse_env_float(f"{prefix}RANK_SCALE", 1.0),
        scorer_threshold_scale=_parse_env_float(f"{prefix}SCORER_THRESHOLD_SCALE", 1.0),
        recycle_ratio=_parse_env_float(f"{prefix}RECYCLE_RATIO", 0.30),
        recycle_cluster_size=_parse_env_int(f"{prefix}RECYCLE_CLUSTER_SIZE", 10),
        min_recycle_tokens=_parse_env_int(f"{prefix}MIN_RECYCLE_TOKENS", 1),
        budget_projection=_parse_env_flag(os.environ.get(f"{prefix}BUDGET_PROJECTION", None), default=True),
        scaffold_only=_parse_env_flag(os.environ.get(f"{prefix}SCAFFOLD_ONLY", None), default=False),
        disable_token_prune=_parse_env_flag(os.environ.get(f"{prefix}DISABLE_TOKEN_PRUNE", None), default=False),
        disable_recycling=_parse_env_flag(os.environ.get(f"{prefix}DISABLE_RECYCLING", None), default=False),
        # Keep generation KV cache by default so SparseVLM latency is measured
        # under the same generate path as the no-pruning and AutoPrune baselines.
        # Set MANTIS_SPARSEVLM_KEEP_KV_CACHE=0 only for explicit no-cache ablations.
        disable_kv_cache=not _parse_env_flag(os.environ.get(f"{prefix}KEEP_KV_CACHE", None), default=True),
        debug=_parse_env_flag(os.environ.get(f"{prefix}DEBUG", None), default=False),
    )


def resolve_mantis_sparsevlm_attn_implementation(
    requested: Optional[str],
    sparsevlm_config: Optional[MantisSparseVLMConfig],
) -> Optional[str]:
    if requested is not None:
        return requested
    if sparsevlm_config is not None and sparsevlm_config.enabled:
        return "sdpa"
    return None


__all__ = [
    "MantisSparseVLMConfig",
    "build_mantis_sparsevlm_config_from_env",
    "resolve_mantis_sparsevlm_attn_implementation",
]
