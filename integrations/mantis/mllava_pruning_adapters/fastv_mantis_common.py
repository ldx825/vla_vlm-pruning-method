import os
from dataclasses import dataclass
from typing import Optional


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


@dataclass
class MantisFastVConfig:
    enabled: bool = False
    start_layer: int = 2
    target_token_num: int = 128
    scaffold_only: bool = False
    disable_token_prune: bool = False
    disable_kv_cache: bool = False
    debug: bool = False


def build_mantis_fastv_config_from_env(prefix: str = "MANTIS_FASTV_") -> MantisFastVConfig:
    return MantisFastVConfig(
        enabled=_parse_env_flag(os.environ.get(f"{prefix}ENABLED", None), default=False),
        start_layer=_parse_env_int(f"{prefix}START_LAYER", 2),
        target_token_num=_parse_env_int(f"{prefix}TARGET_TOKEN_NUM", 128),
        scaffold_only=_parse_env_flag(os.environ.get(f"{prefix}SCAFFOLD_ONLY", None), default=False),
        disable_token_prune=_parse_env_flag(os.environ.get(f"{prefix}DISABLE_TOKEN_PRUNE", None), default=False),
        # Keep generation KV cache by default so FastV latency uses the same
        # generate path as no-pruning baselines. Set MANTIS_FASTV_KEEP_KV_CACHE=0
        # only for explicit no-cache ablations.
        disable_kv_cache=not _parse_env_flag(os.environ.get(f"{prefix}KEEP_KV_CACHE", None), default=True),
        debug=_parse_env_flag(os.environ.get(f"{prefix}DEBUG", None), default=False),
    )


def resolve_mantis_fastv_attn_implementation(
    requested: Optional[str],
    fastv_config: Optional[MantisFastVConfig],
) -> Optional[str]:
    if requested is not None:
        return requested
    if fastv_config is not None and fastv_config.enabled:
        return "sdpa"
    return None


__all__ = [
    "MantisFastVConfig",
    "build_mantis_fastv_config_from_env",
    "resolve_mantis_fastv_attn_implementation",
]
