import time
import types
from typing import Any, Dict, Optional


def consume_internal_attention(attn_module):
    value = getattr(attn_module, "_autoprune_last_attention", None)
    attn_module._autoprune_last_attention = None
    return value


def consume_internal_probe_stats(attn_module) -> Optional[Dict[str, Any]]:
    value = getattr(attn_module, "_autoprune_probe_stats", None)
    attn_module._autoprune_probe_stats = None
    return value


def patch_attention_module(attn_module):
    original_forward = attn_module.forward

    def new_forward(self_attn, *args, **kwargs):
        collect = bool(kwargs.pop("autoprune_collect_attn", False))
        kwargs.pop("autoprune_query_positions", None)
        old_output_attentions = kwargs.get("output_attentions", False)
        if collect:
            kwargs["output_attentions"] = True
        t0 = time.perf_counter()
        outputs = original_forward(*args, **kwargs)
        self_attn._autoprune_probe_stats = {"probe_seconds": time.perf_counter() - t0}
        if collect and isinstance(outputs, tuple) and len(outputs) > 1:
            self_attn._autoprune_last_attention = outputs[1]
            if not old_output_attentions:
                outputs = (outputs[0],) + outputs[2:]
        return outputs

    attn_module.forward = types.MethodType(new_forward, attn_module)
    return original_forward


__all__ = [
    "consume_internal_attention",
    "consume_internal_probe_stats",
    "patch_attention_module",
]
