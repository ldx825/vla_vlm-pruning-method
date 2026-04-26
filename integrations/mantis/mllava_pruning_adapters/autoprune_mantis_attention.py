import inspect
import math
import time
import types
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half


def _read_past_key_states(past_key_value: Any, layer_idx: Optional[int]) -> Optional[torch.Tensor]:
    if past_key_value is None or layer_idx is None:
        return None
    try:
        if hasattr(past_key_value, "key_cache"):
            key_cache = getattr(past_key_value, "key_cache", None)
            if key_cache is not None and len(key_cache) > int(layer_idx):
                cached = key_cache[int(layer_idx)]
                if isinstance(cached, torch.Tensor) and cached.numel() > 0:
                    return cached
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) > int(layer_idx):
            cached = past_key_value[int(layer_idx)][0]
            if isinstance(cached, torch.Tensor) and cached.numel() > 0:
                return cached
    except Exception:
        return None
    return None


def _project_query_states(self_attn, hidden_states: torch.Tensor) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.size()
    if getattr(self_attn.config, "pretraining_tp", 1) > 1:
        query_slices = self_attn.q_proj.weight.split(
            (self_attn.num_heads * self_attn.head_dim) // self_attn.config.pretraining_tp,
            dim=0,
        )
        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self_attn.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)
    else:
        query_states = self_attn.q_proj(hidden_states)
    return query_states.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)


def _project_key_states(self_attn, hidden_states: torch.Tensor) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.size()
    if getattr(self_attn.config, "pretraining_tp", 1) > 1:
        key_value_slicing = (self_attn.num_key_value_heads * self_attn.head_dim) // self_attn.config.pretraining_tp
        key_slices = self_attn.k_proj.weight.split(key_value_slicing, dim=0)
        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self_attn.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)
    else:
        key_states = self_attn.k_proj(hidden_states)
    return key_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)


def _apply_rotary_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def _apply_probe_mask(
    attn_scores: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _, _, q_len, key_len = attn_scores.shape
    min_value = torch.finfo(attn_scores.dtype).min

    if attention_mask is not None:
        if attention_mask.dim() == 4:
            causal_mask = attention_mask[:, :, :, :key_len]
            if query_positions is not None and query_positions.numel() > 0 and causal_mask.shape[2] != q_len:
                causal_mask = causal_mask.index_select(2, query_positions)
            return attn_scores + causal_mask.to(dtype=attn_scores.dtype, device=attn_scores.device)
        if attention_mask.dim() == 2:
            valid = attention_mask[:, None, None, :key_len].to(device=attn_scores.device)
            attn_scores = attn_scores.masked_fill(valid == 0, min_value)

    if q_len > 0:
        if query_positions is None or query_positions.numel() == 0:
            query_positions = torch.arange(q_len, device=attn_scores.device)
        else:
            query_positions = query_positions.to(device=attn_scores.device, dtype=torch.long)
        query_positions = query_positions.unsqueeze(-1)
        key_positions = torch.arange(key_len, device=attn_scores.device).unsqueeze(0)
        causal_invalid = key_positions > query_positions
        attn_scores = attn_scores.masked_fill(causal_invalid.view(1, 1, q_len, key_len), min_value)

    return attn_scores


def _compute_internal_attention(
    self_attn,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_value: Any,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    query_positions: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    try:
        if query_positions is not None and query_positions.numel() > 0:
            query_positions = query_positions.to(device=hidden_states.device, dtype=torch.long)
            query_hidden_states = hidden_states.index_select(1, query_positions)
        else:
            query_positions = None
            query_hidden_states = hidden_states

        query_states = _project_query_states(self_attn, query_hidden_states)
        key_states = _project_key_states(self_attn, hidden_states)

        if position_embeddings is None:
            if position_ids is None:
                return None
            key_position_ids = position_ids
            query_position_ids = position_ids if query_positions is None else position_ids.index_select(1, query_positions)
            cos_k, sin_k = self_attn.rotary_emb(key_states, key_position_ids)
            cos_q, sin_q = self_attn.rotary_emb(query_states, query_position_ids)
        else:
            cos_full, sin_full = position_embeddings
            if query_positions is None:
                cos_q, sin_q = cos_full, sin_full
            else:
                cos_q = cos_full.index_select(1, query_positions)
                sin_q = sin_full.index_select(1, query_positions)
            cos_k, sin_k = cos_full, sin_full

        query_states = _apply_rotary_single(query_states, cos_q, sin_q)
        key_states = _apply_rotary_single(key_states, cos_k, sin_k)

        cached_key_states = _read_past_key_states(past_key_value, getattr(self_attn, "layer_idx", None))
        if cached_key_states is not None:
            key_states = torch.cat([cached_key_states.to(dtype=key_states.dtype), key_states], dim=-2)

        key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(float(self_attn.head_dim))
        attn_scores = _apply_probe_mask(attn_scores, attention_mask, query_positions=query_positions)
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None


def patch_attention_module(attn_module) -> Any:
    if getattr(attn_module, "_autoprune_probe_installed", False):
        return getattr(attn_module, "_autoprune_original_forward", attn_module.forward)

    original_forward = attn_module.forward
    forward_signature = inspect.signature(original_forward)
    forward_parameter_names = set(forward_signature.parameters.keys())
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in forward_signature.parameters.values()
    )

    def new_forward(
        self_attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        collect_internal_attn = bool(kwargs.pop("autoprune_collect_attn", False))
        query_positions = kwargs.pop("autoprune_query_positions", None)
        self_attn._autoprune_last_attn = None
        self_attn._autoprune_last_probe_stats = None

        if collect_internal_attn:
            probe_t0 = time.perf_counter()
            internal_attn = _compute_internal_attention(
                self_attn,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                position_embeddings,
                query_positions=query_positions,
            )
            probe_elapsed = time.perf_counter() - probe_t0
            probe_query_len = int(internal_attn.shape[2]) if internal_attn is not None else (
                int(query_positions.numel()) if isinstance(query_positions, torch.Tensor) else int(hidden_states.shape[1])
            )
            probe_key_len = int(internal_attn.shape[3]) if internal_attn is not None else int(hidden_states.shape[1])
            self_attn._autoprune_last_probe_stats = {
                "probe_seconds": float(probe_elapsed),
                "probe_query_len": int(probe_query_len),
                "probe_key_len": int(probe_key_len),
            }
            if internal_attn is not None:
                self_attn._autoprune_last_attn = internal_attn.detach()

        call_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_value,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }
        if accepts_var_kwargs:
            call_kwargs.update(kwargs)
        else:
            for key, value in kwargs.items():
                if key in forward_parameter_names:
                    call_kwargs[key] = value
        filtered_kwargs = {key: value for key, value in call_kwargs.items() if key in forward_parameter_names}
        return original_forward(**filtered_kwargs)

    attn_module._autoprune_original_forward = original_forward
    attn_module._autoprune_probe_installed = True
    attn_module._autoprune_last_attn = None
    attn_module._autoprune_last_probe_stats = None
    attn_module.forward = types.MethodType(new_forward, attn_module)
    return original_forward


def consume_internal_attention(attn_module) -> Optional[torch.Tensor]:
    attn = getattr(attn_module, "_autoprune_last_attn", None)
    attn_module._autoprune_last_attn = None
    return attn


def consume_internal_probe_stats(attn_module) -> Optional[dict]:
    stats = getattr(attn_module, "_autoprune_last_probe_stats", None)
    attn_module._autoprune_last_probe_stats = None
    return stats


__all__ = [
    "consume_internal_attention",
    "consume_internal_probe_stats",
    "patch_attention_module",
]
