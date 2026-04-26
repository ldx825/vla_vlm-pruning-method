import time
import types
from typing import Any, Dict, Optional

import torch

from .autoprune_mantis_attention import (
    consume_internal_attention,
    consume_internal_probe_stats,
    patch_attention_module,
)
from .autoprune_mantis_layout import (
    build_runtime_stats,
    prepare_input_layout_from_merge,
    resolve_layout_state,
    summarize_layout,
)
from .autoprune_mantis_pruning import (
    align_attention_mask,
    align_position_embeddings,
    align_position_ids,
    build_full_indices,
)
from .fastv_mantis_common import MantisFastVConfig
from .fastv_mantis_pruning import (
    estimate_fastv_flops_reduction,
    record_fastv_schedule_stats,
    select_fastv_unified_visual_tokens,
)


class MantisFastVPatch:
    def __init__(self, model, config: MantisFastVConfig):
        self.model = model
        self.config = config
        self.layers = self._resolve_layers(model)
        self.original_layer_forwards: Dict[int, Any] = {}
        self.original_attn_forwards: Dict[int, Any] = {}
        self.original_merge = None
        self.num_layers = len(self.layers)
        self.start_layer = max(0, min(int(config.start_layer), max(0, self.num_layers - 1)))
        self.reset_runtime_state()

    @staticmethod
    def _resolve_layers(model):
        candidates = [
            ["language_model", "model", "layers"],
            ["language_model", "layers"],
            ["model", "layers"],
        ]
        for path in candidates:
            obj = model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
                return obj
        raise RuntimeError("Unable to locate Mantis decoder layers for FastV.")

    def reset_runtime_state(self) -> None:
        self.input_layout: Dict[str, Any] = {}
        self.global_indices: Optional[torch.Tensor] = None
        self.layout_cache: Dict[str, Any] = {}
        self.has_pruned = False
        self.runtime_stats = build_runtime_stats()
        self.runtime_stats["pruning_layer_indices"] = [int(self.start_layer)]
        self.runtime_stats["candidate_pruning_layer_indices"] = [int(self.start_layer)]
        self.runtime_stats["active_pruning_layer_indices"] = [int(self.start_layer)]
        self.runtime_stats["selected_pruning_mode"] = "fastv_oneshot_unified_visual"

    def _invalidate_layout_cache(self) -> None:
        self.layout_cache = {}

    def _resolve_cached_layout_state(self, seq_len: int, device: torch.device) -> Dict[str, Any]:
        cache_key = (id(self.global_indices), int(seq_len), str(device))
        cached = self.layout_cache.get("entry")
        if cached is not None and cached.get("key") == cache_key:
            self.runtime_stats["layout_cache_hits"] += 1
            return cached["layout"]
        layout = resolve_layout_state(self.input_layout, self.global_indices, seq_len, device)
        self.layout_cache["entry"] = {"key": cache_key, "layout": layout}
        self.runtime_stats["layout_recompute_count"] += 1
        return layout

    def apply(self) -> None:
        if not self.config.enabled:
            return
        for layer_idx, layer in enumerate(self.layers):
            self.original_layer_forwards[layer_idx] = layer.forward
            layer.forward = types.MethodType(self._make_new_forward(layer_idx), layer)
            if not self.config.scaffold_only:
                self.original_attn_forwards[layer_idx] = patch_attention_module(layer.self_attn)

        if not self.config.scaffold_only and hasattr(self.model, "_merge_input_ids_with_image_features"):
            self.original_merge = self.model._merge_input_ids_with_image_features
            self.model._merge_input_ids_with_image_features = types.MethodType(self._make_new_merge(), self.model)

        self.model._fastv_patch = self

    def _make_new_merge(self):
        def new_merge(model_self, image_features, inputs_embeds, input_ids, attention_mask, labels):
            self.reset_runtime_state()
            outputs = self.original_merge(image_features, inputs_embeds, input_ids, attention_mask, labels)
            try:
                self.input_layout = prepare_input_layout_from_merge(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_token_index=int(model_self.config.image_token_index),
                    visual_tokens_per_view=int(image_features.shape[1]),
                )
                self.runtime_stats.update(summarize_layout(self.input_layout))
            except Exception as exc:
                self.input_layout = {}
                if self.config.debug:
                    print(f"[MantisFastV] layout preparation failed: {exc}")
            return outputs

        return new_merge

    def _clean_outputs(self, outputs, original_output_attentions: bool, requested_output_attentions: bool):
        if original_output_attentions:
            return outputs
        if requested_output_attentions:
            return (outputs[0],) + outputs[2:]
        return outputs

    def _record_layer_timing(
        self,
        *,
        seq_len: int,
        use_cache: bool,
        patch_elapsed: float,
        forward_elapsed: float,
        decode_fastpath: bool = False,
    ) -> None:
        stats = self.runtime_stats
        is_decode_step = bool(use_cache and seq_len <= 1)
        overhead_elapsed = max(0.0, float(patch_elapsed) - float(forward_elapsed))

        stats["layer_call_count"] += 1
        stats["patch_total_seconds"] += float(patch_elapsed)
        stats["patch_overhead_seconds"] += overhead_elapsed
        stats["layer_forward_seconds"] += float(forward_elapsed)
        stats["max_seq_len_seen"] = max(int(stats.get("max_seq_len_seen", 0) or 0), int(seq_len))

        if use_cache:
            stats["cache_enabled_layer_calls"] += 1
        if seq_len <= 1:
            stats["seq_len_one_layer_calls"] += 1
        else:
            stats["seq_len_gt_one_layer_calls"] += 1

        if is_decode_step:
            stats["decode_layer_calls"] += 1
            stats["decode_patch_total_seconds"] += float(patch_elapsed)
            stats["decode_patch_overhead_seconds"] += overhead_elapsed
            stats["decode_layer_forward_seconds"] += float(forward_elapsed)
            if decode_fastpath:
                stats["decode_fastpath_layer_calls"] += 1
        else:
            stats["prefill_layer_calls"] += 1
            stats["prefill_patch_total_seconds"] += float(patch_elapsed)
            stats["prefill_patch_overhead_seconds"] += overhead_elapsed
            stats["prefill_layer_forward_seconds"] += float(forward_elapsed)

    def _make_new_forward(self, layer_idx: int):
        def new_forward(self_layer, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            patch_t0 = time.perf_counter()
            batch_size, seq_len, _ = hidden_states.shape
            use_cache = bool(kwargs.get("use_cache", False))
            is_decode_step = bool(use_cache and seq_len <= 1)

            def finish(result, *, forward_elapsed: float = 0.0, decode_fastpath: bool = False):
                patch_elapsed = time.perf_counter() - patch_t0
                self._record_layer_timing(
                    seq_len=seq_len,
                    use_cache=use_cache,
                    patch_elapsed=patch_elapsed,
                    forward_elapsed=forward_elapsed,
                    decode_fastpath=decode_fastpath,
                )
                return result

            if layer_idx == 0:
                if self.global_indices is None or not is_decode_step:
                    self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
                    self._invalidate_layout_cache()
                else:
                    self.runtime_stats["generated_token_steps"] += 1
                if self.runtime_stats.get("prefill_seq_len") is None and seq_len > 1:
                    self.runtime_stats["prefill_seq_len"] = int(seq_len)

            if self.config.scaffold_only or is_decode_step:
                forward_t0 = time.perf_counter()
                outputs = self.original_layer_forwards[layer_idx](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs,
                )
                forward_elapsed = time.perf_counter() - forward_t0
                requested_output_attentions = kwargs.get("output_attentions", False)
                cleaned_outputs = self._clean_outputs(outputs, requested_output_attentions, requested_output_attentions)
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed, decode_fastpath=is_decode_step)

            align_t0 = time.perf_counter()
            attention_mask = align_attention_mask(self.global_indices, attention_mask, batch_size, seq_len)
            position_ids = align_position_ids(self.global_indices, position_ids, batch_size, seq_len)
            align_position_embeddings(self.global_indices, kwargs, batch_size, seq_len)
            self.runtime_stats["alignment_seconds"] += time.perf_counter() - align_t0

            should_prune_here = (
                layer_idx == self.start_layer
                and not self.has_pruned
                and not self.config.disable_token_prune
                and seq_len > 1
            )
            layout = None
            visual_positions = torch.empty(0, dtype=torch.long, device=hidden_states.device)
            if should_prune_here:
                layout_t0 = time.perf_counter()
                layout = self._resolve_cached_layout_state(seq_len, hidden_states.device)
                visual_positions = layout["visual_positions"]
                self.runtime_stats["layout_seconds"] += time.perf_counter() - layout_t0
                if visual_positions.numel() > 0:
                    # FastV paper Sec. 4.1: compute the ranking signal at layer K.
                    # We probe every query row in the current sequence and let
                    # select_fastv_unified_visual_tokens exclude each visual
                    # token's self row. This keeps action/output tokens in the
                    # score and avoids the multi-view per-view heuristic.
                    kwargs["autoprune_collect_attn"] = True
                    kwargs["autoprune_query_positions"] = torch.arange(
                        seq_len,
                        dtype=torch.long,
                        device=hidden_states.device,
                    )

            original_output_attentions = kwargs.get("output_attentions", False)
            forward_t0 = time.perf_counter()
            outputs = self.original_layer_forwards[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            forward_elapsed = time.perf_counter() - forward_t0

            requested_output_attentions = kwargs.get("output_attentions", False)
            cleaned_outputs = self._clean_outputs(outputs, original_output_attentions, requested_output_attentions)

            if not should_prune_here or visual_positions.numel() <= 0 or layout is None:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            attn_weights = outputs[1] if len(outputs) > 1 and original_output_attentions else consume_internal_attention(self_layer.self_attn)
            probe_stats = consume_internal_probe_stats(self_layer.self_attn)
            if probe_stats is not None:
                self.runtime_stats["probe_seconds"] += float(probe_stats.get("probe_seconds", 0.0))
            if attn_weights is None:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            budget_t0 = time.perf_counter()
            keep_visual_tokens = int(self.config.target_token_num)
            keep_visual_tokens = max(1, min(keep_visual_tokens, int(visual_positions.numel())))
            record_fastv_schedule_stats(
                self.runtime_stats,
                start_layer=self.start_layer,
                visual_tokens_before=int(visual_positions.numel()),
                keep_visual_tokens=keep_visual_tokens,
                budget_elapsed=budget_t0,
            )

            score_t0 = time.perf_counter()
            selected_visual_positions = select_fastv_unified_visual_tokens(
                attn_weights,
                visual_positions,
                keep_visual_tokens,
                batch_size,
                hidden_states.device,
                query_positions=torch.arange(seq_len, dtype=torch.long, device=hidden_states.device),
            )
            self.runtime_stats["topk_seconds"] += time.perf_counter() - score_t0

            full_indices = build_full_indices(
                batch_size=batch_size,
                seq_len=seq_len,
                always_keep_positions=layout["always_keep_positions"],
                selected_view_positions=[selected_visual_positions],
                device=hidden_states.device,
            )

            if full_indices.shape[1] >= seq_len:
                self.has_pruned = True
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            gather_t0 = time.perf_counter()
            new_hidden = outputs[0].gather(
                1,
                full_indices.unsqueeze(-1).expand(-1, -1, outputs[0].size(-1)),
            ).contiguous()
            current_globals = self.global_indices
            if current_globals is not None:
                if current_globals.shape[0] == 1 and batch_size > 1:
                    current_globals = current_globals.expand(batch_size, -1)
                self.global_indices = current_globals.gather(1, full_indices)
                self._invalidate_layout_cache()
            self.runtime_stats["gather_seconds"] += time.perf_counter() - gather_t0

            self.has_pruned = True
            self.runtime_stats["pruned_layers"] += 1
            self.runtime_stats["last_tokens_before"] = int(seq_len)
            self.runtime_stats["last_tokens_after"] = int(full_indices.shape[1])
            self.runtime_stats["fastv_keep_visual_tokens"] = int(keep_visual_tokens)
            self.runtime_stats["fastv_prune_ratio"] = float(
                1.0 - keep_visual_tokens / float(max(1, int(visual_positions.numel())))
            )
            flops_reduction = estimate_fastv_flops_reduction(
                config=getattr(self.model, "config", None),
                num_layers=self.num_layers,
                start_layer=self.start_layer,
                tokens_before=int(seq_len),
                tokens_after=int(full_indices.shape[1]),
            )
            self.runtime_stats["flops_reduction"] = flops_reduction
            self.runtime_stats["flops_reduction_ratio"] = flops_reduction

            if self.config.debug:
                print(
                    "[MantisFastV] "
                    f"layer={layer_idx}, visual_before={visual_positions.numel()}, "
                    f"visual_after={keep_visual_tokens}, before={seq_len}, after={full_indices.shape[1]}"
                )

            if len(outputs) == 3:
                return finish(
                    self._clean_outputs((new_hidden, outputs[1], outputs[2]), original_output_attentions, requested_output_attentions),
                    forward_elapsed=forward_elapsed,
                )
            if len(outputs) == 2:
                return finish(
                    self._clean_outputs((new_hidden, outputs[1]), original_output_attentions, requested_output_attentions),
                    forward_elapsed=forward_elapsed,
                )
            return finish(
                self._clean_outputs((new_hidden,), original_output_attentions, requested_output_attentions),
                forward_elapsed=forward_elapsed,
            )

        return new_forward


def maybe_apply_mantis_fastv(model, config: Optional[MantisFastVConfig]) -> Optional[MantisFastVPatch]:
    if config is None or not config.enabled:
        return None
    patch = MantisFastVPatch(model, config)
    patch.apply()
    if config.disable_kv_cache:
        model.config.use_cache = False
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            model.language_model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
    model._fastv_patch = patch
    model._autoprune_patch = patch
    return patch


__all__ = [
    "MantisFastVPatch",
    "maybe_apply_mantis_fastv",
]
