import time
import types
from typing import Any, Dict, Optional

import torch

from .autoprune_mantis_attention import (
    consume_internal_attention,
    consume_internal_probe_stats,
    patch_attention_module,
)
from .autoprune_mantis_common import AdaptiveBudgetStrategy, MantisAutoPruneConfig
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
    compute_schedule_from_attn,
    select_view_token_indices,
)
from .sparsevlm_mantis_pruning import estimate_sparsevlm_flops_reduction


class MantisAutoPrunePatch:
    def __init__(self, model, config: MantisAutoPruneConfig):
        self.model = model
        self.config = config
        self.strategy = AdaptiveBudgetStrategy(config.b_min, config.b_max, config.mu, config.tau)

        self.layers = self._resolve_layers(model)
        self.original_layer_forwards: Dict[int, Any] = {}
        self.original_attn_forwards: Dict[int, Any] = {}
        self.original_merge = None

        self.input_layout: Dict[str, Any] = {}
        self.global_indices: Optional[torch.Tensor] = None
        self.keep_count_schedule = []
        self.layout_cache: Dict[str, Any] = {}

        self.num_layers = len(self.layers)
        self.pruning_loc = self._build_pruning_layers()
        self.active_pruning_loc = list(self.pruning_loc)
        self.pruning_layer_set = set(self.active_pruning_loc)
        self.pruning_schedule_index = {layer_idx: idx for idx, layer_idx in enumerate(self.active_pruning_loc)}
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
        raise RuntimeError("Unable to locate Mantis decoder layers for AutoPrune.")

    def _build_pruning_layers(self):
        all_layers = list(range(int(self.config.start_layer), self.num_layers))
        if not all_layers:
            return []
        if self.config.monitor_only or self.config.disable_progressive_prune:
            return [all_layers[0]]

        max_pruning_layers = int(getattr(self.config, "max_pruning_layers", 0) or 0)
        if max_pruning_layers <= 0 or max_pruning_layers >= len(all_layers):
            return all_layers
        if max_pruning_layers == 1:
            return [all_layers[0]]

        selected = []
        seen = set()
        last_index = len(all_layers) - 1
        for offset in range(max_pruning_layers):
            layer_offset = int(round(offset * last_index / max(1, max_pruning_layers - 1)))
            layer_idx = int(all_layers[layer_offset])
            if layer_idx in seen:
                continue
            selected.append(layer_idx)
            seen.add(layer_idx)

        if len(selected) < max_pruning_layers:
            for layer_idx in all_layers:
                if layer_idx in seen:
                    continue
                selected.append(int(layer_idx))
                seen.add(int(layer_idx))
                if len(selected) >= max_pruning_layers:
                    break
        return sorted(selected)

    def reset_runtime_state(self) -> None:
        self.input_layout = {}
        self.global_indices = None
        self.keep_count_schedule = []
        self.layout_cache = {}
        self.runtime_stats = build_runtime_stats()
        self.active_pruning_loc = list(self.pruning_loc)
        self.pruning_layer_set = set(self.active_pruning_loc)
        self.pruning_schedule_index = {layer_idx: idx for idx, layer_idx in enumerate(self.active_pruning_loc)}
        self.runtime_stats["configured_max_pruning_layers"] = int(getattr(self.config, "max_pruning_layers", 0) or 0)
        self.runtime_stats["pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.pruning_loc]
        self.runtime_stats["candidate_pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.pruning_loc]
        self.runtime_stats["active_pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.active_pruning_loc]
        self.runtime_stats["adaptive_layer_policy_enabled"] = int(bool(getattr(self.config, "enable_adaptive_layer_policy", False)))
        self.runtime_stats["adaptive_layer_choices"] = [
            int(value) for value in getattr(self.config, "adaptive_layer_choices", ())
        ]
        self.runtime_stats["autoprune_layer_records"] = []

    def _invalidate_layout_cache(self) -> None:
        self.layout_cache = {}

    def _set_active_pruning_layers(self, pruning_loc) -> None:
        if pruning_loc:
            self.active_pruning_loc = [int(layer_idx) for layer_idx in pruning_loc]
        else:
            self.active_pruning_loc = [int(self.config.start_layer)]
        self.pruning_layer_set = set(self.active_pruning_loc)
        self.pruning_schedule_index = {layer_idx: idx for idx, layer_idx in enumerate(self.active_pruning_loc)}
        self.runtime_stats["active_pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.active_pruning_loc]
        self.runtime_stats["adaptive_layer_count"] = int(len(self.active_pruning_loc))

    def _resolve_cached_layout_state(self, seq_len: int, device: torch.device) -> Dict[str, Any]:
        cache_key = (
            id(self.global_indices),
            int(seq_len),
            str(device),
        )
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

        self.model._autoprune_patch = self

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
                    print(f"[MantisAutoPrune] layout preparation failed: {exc}")
            return outputs

        return new_merge

    def _clean_outputs(self, outputs, original_output_attentions: bool, requested_output_attentions: bool):
        if original_output_attentions:
            return outputs
        if requested_output_attentions:
            return (outputs[0],) + outputs[2:]
        return outputs

    def _should_collect_attn(self, layer_idx: int, kwargs: Dict[str, Any], seq_len: int) -> bool:
        if seq_len <= 1:
            return False
        if layer_idx < self.config.start_layer or self.config.disable_token_prune:
            return False
        if layer_idx not in self.pruning_layer_set:
            return False
        if self.config.monitor_only:
            return layer_idx == self.config.start_layer
        return True

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
                if not is_decode_step or self.global_indices is None:
                    self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
                    self._invalidate_layout_cache()
                else:
                    self.runtime_stats["generated_token_steps"] += 1
                if self.runtime_stats.get("prefill_seq_len") is None and seq_len > 1:
                    self.runtime_stats["prefill_seq_len"] = int(seq_len)

            if self.config.scaffold_only:
                forward_t0 = time.perf_counter()
                outputs = self.original_layer_forwards[layer_idx](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs,
                )
                forward_elapsed = time.perf_counter() - forward_t0
                requested_output_attentions = kwargs.get("output_attentions", False)
                original_output_attentions = requested_output_attentions
                cleaned_outputs = self._clean_outputs(outputs, original_output_attentions, requested_output_attentions)
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed, decode_fastpath=is_decode_step)

            if is_decode_step:
                forward_t0 = time.perf_counter()
                outputs = self.original_layer_forwards[layer_idx](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs,
                )
                forward_elapsed = time.perf_counter() - forward_t0
                requested_output_attentions = kwargs.get("output_attentions", False)
                original_output_attentions = requested_output_attentions
                cleaned_outputs = self._clean_outputs(outputs, original_output_attentions, requested_output_attentions)
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed, decode_fastpath=True)

            align_t0 = time.perf_counter()
            attention_mask = align_attention_mask(self.global_indices, attention_mask, batch_size, seq_len)
            position_ids = align_position_ids(self.global_indices, position_ids, batch_size, seq_len)
            align_position_embeddings(self.global_indices, kwargs, batch_size, seq_len)
            self.runtime_stats["alignment_seconds"] += time.perf_counter() - align_t0

            pruning_layer = layer_idx in self.pruning_layer_set
            query_positions = torch.empty(0, dtype=torch.long, device=hidden_states.device)
            view_positions = []
            layout = None
            if pruning_layer:
                layout_t0 = time.perf_counter()
                layout = self._resolve_cached_layout_state(seq_len, hidden_states.device)
                self.runtime_stats["layout_seconds"] += time.perf_counter() - layout_t0
                query_positions = layout["query_positions"]
                view_positions = layout["view_positions"]

            need_attentions = self._should_collect_attn(layer_idx, kwargs, seq_len)
            original_output_attentions = kwargs.get("output_attentions", False)
            if need_attentions and query_positions.numel() > 0 and view_positions:
                kwargs["autoprune_collect_attn"] = True
                kwargs["autoprune_query_positions"] = query_positions

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

            if layer_idx < self.config.start_layer:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            if layer_idx not in self.pruning_layer_set:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            if query_positions.numel() <= 0 or not view_positions:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            attn_weights = outputs[1] if len(outputs) > 1 and original_output_attentions else consume_internal_attention(self_layer.self_attn)
            probe_stats = consume_internal_probe_stats(self_layer.self_attn)
            if probe_stats is not None:
                self.runtime_stats["probe_seconds"] += float(probe_stats.get("probe_seconds", 0.0))
            if attn_weights is None:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            if layer_idx == self.config.start_layer and not self.keep_count_schedule:
                schedule_result = compute_schedule_from_attn(
                    self.config,
                    self.strategy,
                    self.pruning_loc,
                    hidden_states,
                    attn_weights,
                    layout,
                    self.runtime_stats,
                )
                self.keep_count_schedule = schedule_result.get("keep_count_schedule", [])
                self._set_active_pruning_layers(schedule_result.get("active_pruning_loc", self.pruning_loc))

            if not self.keep_count_schedule or self.config.monitor_only or self.config.disable_token_prune:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            schedule_idx = self.pruning_schedule_index.get(layer_idx, 0)
            schedule_idx = min(max(0, schedule_idx), len(self.keep_count_schedule) - 1)
            target_keep_per_view = self.keep_count_schedule[schedule_idx]

            score_t0 = time.perf_counter()
            selected_view_positions = select_view_token_indices(
                attn_weights,
                query_positions,
                view_positions,
                target_keep_per_view,
                batch_size,
                hidden_states.device,
            )
            self.runtime_stats["topk_seconds"] += time.perf_counter() - score_t0

            full_indices = build_full_indices(
                batch_size=batch_size,
                seq_len=seq_len,
                always_keep_positions=layout["always_keep_positions"],
                selected_view_positions=selected_view_positions,
                device=hidden_states.device,
            )

            if full_indices.shape[1] >= seq_len:
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
            self.runtime_stats["pruned_layers"] += 1
            self.runtime_stats["last_tokens_before"] = int(seq_len)
            self.runtime_stats["last_tokens_after"] = int(full_indices.shape[1])
            self.runtime_stats.setdefault("autoprune_layer_records", [])
            self.runtime_stats["autoprune_layer_records"].append(
                {
                    "layer_idx": int(layer_idx),
                    "tokens_before": int(seq_len),
                    "tokens_after": int(full_indices.shape[1]),
                    "visual_after": int(full_indices.shape[1] - layout["always_keep_positions"].numel()),
                    "target_keep_per_view": [int(value) for value in target_keep_per_view],
                }
            )
            flops_reduction = estimate_sparsevlm_flops_reduction(
                config=getattr(self.model, "config", None),
                num_layers=self.num_layers,
                layer_records=self.runtime_stats.get("autoprune_layer_records", []),
                prefill_seq_len=self.runtime_stats.get("prefill_seq_len"),
            )
            self.runtime_stats["flops_reduction"] = flops_reduction
            self.runtime_stats["flops_reduction_ratio"] = flops_reduction
            if self.config.debug:
                print(
                    "[MantisAutoPrune] "
                    f"layer={layer_idx}, before={seq_len}, after={full_indices.shape[1]}, "
                    f"views={[int(v.shape[1]) for v in selected_view_positions]}"
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


class MantisInputControlPatch:
    def __init__(self, model, config: MantisAutoPruneConfig):
        self.model = model
        self.config = config
        self.original_merge = None
        self.input_layout: Dict[str, Any] = {}
        self.runtime_stats = build_runtime_stats()

    def reset_runtime_state(self) -> None:
        self.input_layout = {}
        self.runtime_stats = build_runtime_stats()

    def apply(self) -> None:
        if hasattr(self.model, "_merge_input_ids_with_image_features"):
            self.original_merge = self.model._merge_input_ids_with_image_features
            self.model._merge_input_ids_with_image_features = types.MethodType(self._make_new_merge(), self.model)
        self.model._autoprune_patch = self

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
                if isinstance(outputs, tuple) and len(outputs) >= 1 and isinstance(outputs[0], torch.Tensor):
                    self.runtime_stats["prefill_seq_len"] = int(outputs[0].shape[1])
            except Exception as exc:
                self.input_layout = {}
                if self.config.debug:
                    print(f"[MantisInputControl] layout preparation failed: {exc}")
            return outputs

        return new_merge


def maybe_apply_mantis_autoprune(model, config: Optional[MantisAutoPruneConfig]) -> Optional[MantisAutoPrunePatch]:
    if config is None or not config.enabled:
        return None
    if getattr(config, "input_control_only", False):
        patch = MantisInputControlPatch(model, config)
        patch.apply()
        model._autoprune_patch = patch
        return patch
    patch = MantisAutoPrunePatch(model, config)
    patch.apply()
    if config.disable_kv_cache:
        model.config.use_cache = False
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            model.language_model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
    model._autoprune_patch = patch
    return patch


__all__ = [
    "MantisAutoPrunePatch",
    "maybe_apply_mantis_autoprune",
]
