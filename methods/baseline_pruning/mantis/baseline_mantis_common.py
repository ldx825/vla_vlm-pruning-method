import os
from dataclasses import dataclass
from typing import Optional, Tuple


def _parse_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _parse_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _parse_int_tuple(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = []
    for chunk in raw.split(","):
        try:
            values.append(int(chunk.strip()))
        except ValueError:
            pass
    return tuple(values) if values else default


@dataclass
class MantisBaselinePruneConfig:
    enabled: bool = False
    method: str = "fastervlm"
    target_token_num: int = 128
    debug: bool = False
    disable_kv_cache: bool = False
    visionzip_dominant_ratio: float = 0.7
    pyramiddrop_start_layer: int = 2
    pyramiddrop_max_pruning_layers: int = 6
    pyramiddrop_layer_indices: Tuple[int, ...] = ()


def build_mantis_baseline_config_from_env(prefix: str = "MANTIS_BASELINE_PRUNE_") -> MantisBaselinePruneConfig:
    method = os.environ.get(f"{prefix}METHOD", "fastervlm").strip().lower()
    return MantisBaselinePruneConfig(
        enabled=_parse_bool(f"{prefix}ENABLED", False),
        method=method,
        target_token_num=_parse_int(f"{prefix}TARGET_TOKEN_NUM", 128),
        debug=_parse_bool(f"{prefix}DEBUG", False),
        disable_kv_cache=_parse_bool(f"{prefix}DISABLE_KV_CACHE", False),
        visionzip_dominant_ratio=_parse_float(f"{prefix}VISIONZIP_DOMINANT_RATIO", 0.7),
        pyramiddrop_start_layer=_parse_int(f"{prefix}PYRAMIDDROP_START_LAYER", 2),
        pyramiddrop_max_pruning_layers=_parse_int(f"{prefix}PYRAMIDDROP_MAX_PRUNING_LAYERS", 6),
        pyramiddrop_layer_indices=_parse_int_tuple(f"{prefix}PYRAMIDDROP_LAYER_INDICES", ()),
    )


def normalize_method_name(method: Optional[str]) -> str:
    value = (method or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "fastvlm": "fastervlm",
        "fastervlm": "fastervlm",
        "visionzip": "visionzip",
        "pyramiddrop": "pyramiddrop",
        "pdrop": "pyramiddrop",
    }
    if value not in aliases:
        raise ValueError(f"Unknown Mantis baseline pruning method: {method!r}")
    return aliases[value]


__all__ = [
    "MantisBaselinePruneConfig",
    "build_mantis_baseline_config_from_env",
    "normalize_method_name",
]
