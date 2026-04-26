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
)
from .sparsevlm_mantis_common import MantisSparseVLMConfig
from .sparsevlm_mantis_pruning import (
    build_sparsevlm_sequence,
    decide_sparsevlm_keep_count,
    estimate_sparsevlm_flops_reduction,
    record_sparsevlm_stats,
    score_visual_tokens_by_text_raters,
    select_and_recycle_visual_tokens,
    select_visual_relevant_text_tokens,
)


class MantisSparseVLMPatch:
    def __init__(self, model, config: MantisSparseVLMConfig):
        self.model = model
        self.config = config
        self.layers = self._resolve_layers(model)
        self.original_layer_forwards: Dict[int, Any] = {}
        self.original_attn_forwards: Dict[int, Any] = {}
        self.original_merge = None
        self.num_layers = len(self.layers)
        self.pruning_loc = self._build_pruning_layers()
        self.pruning_layer_set = set(self.pruning_loc)
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
        raise RuntimeError("Unable to locate Mantis decoder layers for SparseVLM.")

    def _build_pruning_layers(self):
        configured = [int(layer_idx) for layer_idx in getattr(self.config, "pruning_layers", ()) if int(layer_idx) >= 0]
        if configured:
            return sorted({min(layer_idx, self.num_layers - 1) for layer_idx in configured})
        return [max(0, min(int(self.config.start_layer), max(0, self.num_layers - 1)))]

    def reset_runtime_state(self) -> None:
        self.input_layout: Dict[str, Any] = {}
        self.global_indices: Optional[torch.Tensor] = None
        self.layout_cache: Dict[str, Any] = {}
        self.scorer_global_positions: Optional[torch.Tensor] = None
        self.runtime_stats = build_runtime_stats()
        self.runtime_stats["configured_max_pruning_layers"] = int(len(self.pruning_loc))
        self.runtime_stats["pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.pruning_loc]
        self.runtime_stats["candidate_pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.pruning_loc]
        self.runtime_stats["active_pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.pruning_loc]
        self.runtime_stats["adaptive_layer_count"] = int(len(self.pruning_loc))
        self.runtime_stats["selected_pruning_mode"] = "sparsevlm_progressive_unified_visual"
        self.runtime_stats["sparsevlm_layer_records"] = []

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

        self.model._sparsevlm_patch = self
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
                    print(f"[MantisSparseVLM] layout preparation failed: {exc}")
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

    def _current_scorer_positions(self, layout: Dict[str, Any], hidden_states: torch.Tensor) -> torch.Tensor:
        visual_positions = layout["visual_positions"]
        query_positions = layout["query_positions"]
        if self.scorer_global_positions is None:
            scorer_positions = select_visual_relevant_text_tokens(
                hidden_states,
                visual_positions,
                query_positions,
                threshold_scale=float(getattr(self.config, "scorer_threshold_scale", 1.0)),
            )
            if scorer_positions.numel() > 0 and self.global_indices is not None:
                globals_for_row = self.global_indices[0].to(device=scorer_positions.device)
                self.scorer_global_positions = globals_for_row.index_select(0, scorer_positions).detach().clone()
                self.runtime_stats["sparsevlm_scorer_tokens"] = int(scorer_positions.numel())
            return scorer_positions

        globals_for_row = self.global_indices[0].to(device=hidden_states.device)
        scorer_globals = self.scorer_global_positions.to(device=hidden_states.device, dtype=globals_for_row.dtype)
        mask = torch.isin(globals_for_row, scorer_globals)
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _update_global_indices_with_recycling(
        self,
        full_indices: torch.Tensor,
        selected_visual_positions: torch.Tensor,
        recycled_tokens: torch.Tensor,
        visual_insert_position: int,
    ) -> None:
        if self.global_indices is None:
            return
        current_globals = self.global_indices
        batch_size = full_indices.shape[0]
        if current_globals.shape[0] == 1 and batch_size > 1:
            current_globals = current_globals.expand(batch_size, -1)
        gathered_globals = current_globals.gather(1, full_indices)
        if recycled_tokens.numel() <= 0:
            self.global_indices = gathered_globals
            self._invalidate_layout_cache()
            return
        if selected_visual_positions.numel() > 0:
            recycle_anchor = current_globals.gather(1, selected_visual_positions[:, :1])
        else:
            anchor_idx = torch.full((batch_size, 1), int(visual_insert_position), device=full_indices.device, dtype=torch.long)
            recycle_anchor = current_globals.gather(1, anchor_idx.clamp(max=current_globals.shape[1] - 1))
        recycled_globals = recycle_anchor.expand(batch_size, recycled_tokens.shape[1])
        insert_at = int((full_indices[0] < int(visual_insert_position)).sum().item())
        self.global_indices = torch.cat(
            [gathered_globals[:, :insert_at], recycled_globals, gathered_globals[:, insert_at:]],
            dim=1,
        ).contiguous()
        self._invalidate_layout_cache()

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
                layer_idx in self.pruning_layer_set
                and not self.config.disable_token_prune
                and seq_len > 1
            )
            layout = None
            visual_positions = torch.empty(0, dtype=torch.long, device=hidden_states.device)
            scorer_positions = torch.empty(0, dtype=torch.long, device=hidden_states.device)
            if should_prune_here:
                layout_t0 = time.perf_counter()
                layout = self._resolve_cached_layout_state(seq_len, hidden_states.device)
                visual_positions = layout["visual_positions"]
                scorer_positions = self._current_scorer_positions(layout, hidden_states)
                self.runtime_stats["layout_seconds"] += time.perf_counter() - layout_t0
                if visual_positions.numel() > 0 and scorer_positions.numel() > 0:
                    kwargs["autoprune_collect_attn"] = True
                    kwargs["autoprune_query_positions"] = scorer_positions

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

            if (
                not should_prune_here
                or layout is None
                or visual_positions.numel() <= 1
                or scorer_positions.numel() <= 0
            ):
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            attn_weights = outputs[1] if len(outputs) > 1 and original_output_attentions else consume_internal_attention(self_layer.self_attn)
            probe_stats = consume_internal_probe_stats(self_layer.self_attn)
            if probe_stats is not None:
                self.runtime_stats["probe_seconds"] += float(probe_stats.get("probe_seconds", 0.0))
            if attn_weights is None:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            score_t0 = time.perf_counter()
            scores = score_visual_tokens_by_text_raters(attn_weights, scorer_positions, visual_positions)
            self.runtime_stats["topk_seconds"] += time.perf_counter() - score_t0
            if scores.numel() <= 0:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            budget_t0 = time.perf_counter()
            remaining_layers = sum(1 for value in self.pruning_loc if int(value) >= int(layer_idx))
            visual_hidden = outputs[0].index_select(1, visual_positions)
            keep_count, rank_stats = decide_sparsevlm_keep_count(
                self.config,
                visual_hidden,
                scores,
                current_visual_tokens=int(visual_positions.numel()),
                layer_idx=layer_idx,
                remaining_layers=remaining_layers,
            )

            selected_visual_positions, recycled_tokens, recycle_stats = select_and_recycle_visual_tokens(
                outputs[0],
                visual_positions,
                scores,
                keep_count,
                self.config,
            )
            if self.config.budget_projection and int(self.config.target_token_num) > 0:
                final_budget = max(1, int(self.config.target_token_num))
                if remaining_layers <= 1 and keep_count + int(recycled_tokens.shape[1]) > final_budget:
                    allowed_recycled = max(0, final_budget - int(keep_count))
                    recycled_tokens = recycled_tokens[:, :allowed_recycled].contiguous()
                    recycle_stats["recycled_tokens"] = int(recycled_tokens.shape[1])
            visual_after = int(keep_count + recycled_tokens.shape[1])
            record_sparsevlm_stats(
                self.runtime_stats,
                config=self.config,
                layer_idx=layer_idx,
                visual_before=int(visual_positions.numel()),
                visual_after=visual_after,
                keep_count=keep_count,
                recycle_stats=recycle_stats,
                rank_stats=rank_stats,
                budget_elapsed_t0=budget_t0,
            )

            new_hidden, full_indices = build_sparsevlm_sequence(
                outputs[0],
                layout["always_keep_positions"],
                selected_visual_positions,
                recycled_tokens,
                seq_len=seq_len,
                visual_insert_position=int(visual_positions.min().item()),
            )
            if new_hidden.shape[1] >= seq_len:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)
            if self.runtime_stats.get("sparsevlm_layer_records"):
                self.runtime_stats["sparsevlm_layer_records"][-1]["tokens_before"] = int(seq_len)
                self.runtime_stats["sparsevlm_layer_records"][-1]["tokens_after"] = int(new_hidden.shape[1])

            gather_t0 = time.perf_counter()
            self._update_global_indices_with_recycling(
                full_indices,
                selected_visual_positions,
                recycled_tokens,
                int(visual_positions.min().item()),
            )
            self.runtime_stats["gather_seconds"] += time.perf_counter() - gather_t0
            self.runtime_stats["pruned_layers"] += 1
            self.runtime_stats["last_tokens_before"] = int(seq_len)
            self.runtime_stats["last_tokens_after"] = int(new_hidden.shape[1])
            flops_reduction = estimate_sparsevlm_flops_reduction(
                config=getattr(self.model, "config", None),
                num_layers=self.num_layers,
                layer_records=self.runtime_stats.get("sparsevlm_layer_records", []),
                prefill_seq_len=self.runtime_stats.get("prefill_seq_len"),
            )
            self.runtime_stats["flops_reduction"] = flops_reduction
            self.runtime_stats["flops_reduction_ratio"] = flops_reduction
            self.runtime_stats["keep_count_schedule"] = [
                [int(record.get("visual_after", 0))]
                for record in self.runtime_stats.get("sparsevlm_layer_records", [])
            ]
            self.runtime_stats["dense_keep_count_schedule"] = list(self.runtime_stats["keep_count_schedule"])
            self.runtime_stats["dense_schedule_length"] = int(len(self.runtime_stats["keep_count_schedule"]))
            self.runtime_stats["compressed_schedule_length"] = int(len(self.runtime_stats["keep_count_schedule"]))
            self.runtime_stats["dense_pruning_layer_indices"] = [
                int(record.get("layer_idx", 0))
                for record in self.runtime_stats.get("sparsevlm_layer_records", [])
            ]

            if self.config.debug:
                print(
                    "[MantisSparseVLM] "
                    f"layer={layer_idx}, visual_before={visual_positions.numel()}, "
                    f"kept={keep_count}, recycled={recycled_tokens.shape[1]}, "
                    f"before={seq_len}, after={new_hidden.shape[1]}"
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


def maybe_apply_mantis_sparsevlm(
    model,
    config: Optional[MantisSparseVLMConfig],
) -> Optional[MantisSparseVLMPatch]:
    if config is None or not config.enabled:
        return None
    patch = MantisSparseVLMPatch(model, config)
    patch.apply()
    if config.disable_kv_cache:
        model.config.use_cache = False
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            model.language_model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
    model._sparsevlm_patch = patch
    model._autoprune_patch = patch
    return patch


__all__ = [
    "MantisSparseVLMPatch",
    "maybe_apply_mantis_sparsevlm",
]
