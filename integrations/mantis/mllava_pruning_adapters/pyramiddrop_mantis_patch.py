import time
import types
from typing import Any, Dict, List, Optional

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
from .pyramiddrop_mantis_common import MantisPyramidDropConfig
from .sparsevlm_mantis_pruning import estimate_sparsevlm_flops_reduction


def _select_query_rows(attn_weights: torch.Tensor, query_positions: torch.Tensor) -> torch.Tensor:
    if query_positions.numel() <= 0:
        return attn_weights[:, :, :0, :]
    if attn_weights.shape[2] == int(query_positions.numel()):
        return attn_weights
    return attn_weights.index_select(2, query_positions)


def _build_runtime_stage_lengths(num_layers: int, pruning_loc: List[int]) -> List[int]:
    if num_layers <= 0:
        return []
    if not pruning_loc:
        return [int(num_layers)]

    boundaries = sorted(int(layer_idx) for layer_idx in pruning_loc)
    lengths: List[int] = []
    previous = -1
    for boundary in boundaries:
        boundary = max(previous, min(boundary, int(num_layers) - 1))
        # This patch drops after the boundary layer has run, so layers
        # previous+1..boundary still belong to the pre-drop stage.
        lengths.append(max(0, int(boundary - previous)))
        previous = int(boundary)
    lengths.append(max(0, int(num_layers) - 1 - previous))
    return lengths


def _weighted_average_from_lambda(initial_visual_tokens: int, lam: float, stage_lengths: List[int]) -> float:
    total_layers = int(sum(stage_lengths))
    if total_layers <= 0:
        return float(initial_visual_tokens)
    total = 0.0
    for stage_idx, layer_count in enumerate(stage_lengths):
        total += float(layer_count) * float(initial_visual_tokens) * (float(lam) ** float(stage_idx))
    return total / float(total_layers)


def _build_exponential_keep_schedule(
    *,
    initial_visual_tokens: int,
    target_final_tokens: int,
    stage_lengths: List[int],
    min_keep_tokens: int,
) -> Dict[str, Any]:
    initial_visual_tokens = max(0, int(initial_visual_tokens))
    if initial_visual_tokens <= 0:
        return {"lambda": 1.0, "keep_schedule": [], "stage_lengths": []}

    stage_lengths = [max(0, int(value)) for value in stage_lengths if int(value) >= 0]
    if not stage_lengths:
        stage_lengths = [1]
    drop_count = max(0, len(stage_lengths) - 1)
    if drop_count <= 0:
        return {
            "lambda": 1.0,
            "keep_schedule": [],
            "stage_lengths": stage_lengths,
            "weighted_average_visual_tokens": float(initial_visual_tokens),
            "target_final_clamped": int(initial_visual_tokens),
            "target_clamped_by_bounds": False,
        }

    floor = max(1, min(int(min_keep_tokens), initial_visual_tokens))
    clamped_target = max(floor, min(int(target_final_tokens), initial_visual_tokens))
    target_clamped = int(clamped_target) != int(target_final_tokens)
    if drop_count <= 1:
        lam = float(clamped_target) / float(initial_visual_tokens)
    else:
        # The Mantis sweeps compare methods at an explicit final visual-token
        # budget. Build an exponential PyramidDrop path whose last pruning
        # stage lands on that budget, rather than treating the budget as a
        # layer-weighted average; the latter over-prunes BLINK to one token.
        lam = (float(clamped_target) / float(initial_visual_tokens)) ** (1.0 / float(drop_count))

    counts: List[int] = []
    previous = initial_visual_tokens
    for drop_idx in range(drop_count):
        # Paper Eq. 3: stage visual tokens follow V_s = V_0 * lambda^(s-1).
        # Since stage 0 runs all tokens, each drop prepares the next stage.
        raw = initial_visual_tokens * (lam ** float(drop_idx + 1))
        keep = int(round(raw))
        keep = max(floor, min(previous, keep, initial_visual_tokens))
        counts.append(int(keep))
        previous = int(keep)

    if counts:
        counts[-1] = int(clamped_target)
        for idx in range(len(counts) - 2, -1, -1):
            counts[idx] = max(int(counts[idx]), int(counts[idx + 1]))
            counts[idx] = min(int(counts[idx]), initial_visual_tokens)

    def weighted_sum(values: List[int]) -> int:
        total = int(initial_visual_tokens) * int(stage_lengths[0])
        for idx, value in enumerate(values):
            total += int(value) * int(stage_lengths[idx + 1])
        return int(total)

    total_layers = max(1, int(sum(stage_lengths)))
    realized_average = float(weighted_sum(counts)) / float(total_layers)
    return {
        "lambda": float(lam),
        "keep_schedule": [int(value) for value in counts],
        "stage_lengths": [int(value) for value in stage_lengths],
        "weighted_average_visual_tokens": float(realized_average),
        "target_final_clamped": int(clamped_target),
        "target_clamped_by_bounds": bool(target_clamped),
    }


def _select_unified_visual_indices(
    attn_weights: torch.Tensor,
    query_positions: torch.Tensor,
    visual_positions: torch.Tensor,
    keep_count: int,
    batch_size: int,
    device: torch.device,
) -> List[torch.Tensor]:
    if visual_positions.numel() <= 0:
        return [torch.empty((batch_size, 0), dtype=torch.long, device=device)]
    keep_count = max(1, min(int(keep_count), int(visual_positions.numel())))
    attn_queries = _select_query_rows(attn_weights, query_positions)
    # Paper-aligned Rank & Drop: score each image token by text-to-image
    # attention, but treat all Mantis views as one unified visual sequence.
    score = attn_queries.index_select(3, visual_positions).mean(dim=1).mean(dim=1)
    score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
    _, topk = torch.topk(score, keep_count, dim=1)
    base = visual_positions.unsqueeze(0).expand(batch_size, -1)
    selected = base.gather(1, topk)
    return [torch.sort(selected, dim=1).values]


class MantisPyramidDropPatch:
    def __init__(self, model, config: MantisPyramidDropConfig):
        self.model = model
        self.config = config
        self.layers = self._resolve_layers(model)
        self.original_layer_forwards: Dict[int, Any] = {}
        self.original_attn_forwards: Dict[int, Any] = {}
        self.original_merge = None
        self.input_layout: Dict[str, Any] = {}
        self.global_indices: Optional[torch.Tensor] = None
        self.layout_cache: Dict[str, Any] = {}
        self.keep_count_schedule: List[int] = []
        self.pyramid_lambda: float = 1.0
        self.num_layers = len(self.layers)
        self.pruning_loc = self._build_pruning_layers()
        self.pruning_layer_set = set(self.pruning_loc)
        self.pruning_schedule_index = {layer_idx: idx for idx, layer_idx in enumerate(self.pruning_loc)}
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
        raise RuntimeError("Unable to locate Mantis decoder layers for PyramidDrop.")

    def _build_pruning_layers(self) -> List[int]:
        all_layers = list(range(max(0, int(self.config.start_layer)), self.num_layers))
        if not all_layers:
            return []
        max_pruning_layers = int(getattr(self.config, "max_pruning_layers", 0) or 0)
        if max_pruning_layers <= 0 or max_pruning_layers >= len(all_layers):
            return all_layers
        if max_pruning_layers == 1:
            return [all_layers[0]]
        start_layer = max(0, int(self.config.start_layer))
        if start_layer > 0:
            # PyramidDrop specifies stage boundaries such as [8, 16, 24]:
            # each boundary ends a stage, and the final post-drop stage runs
            # until the decoder output. Prefer that arithmetic progression
            # when START_LAYER gives a natural first stage length.
            staged = []
            for stage_idx in range(max_pruning_layers):
                layer_idx = int(start_layer * (stage_idx + 1))
                if layer_idx >= self.num_layers:
                    break
                staged.append(layer_idx)
            if staged:
                return staged
        candidate_layers = all_layers[:-1] if len(all_layers) > 1 else all_layers
        selected = []
        seen = set()
        last_index = len(candidate_layers) - 1
        for offset in range(max_pruning_layers):
            layer_offset = int(round(offset * last_index / max(1, max_pruning_layers - 1)))
            layer_idx = int(candidate_layers[layer_offset])
            if layer_idx in seen:
                continue
            selected.append(layer_idx)
            seen.add(layer_idx)
        return sorted(selected)

    def reset_runtime_state(self) -> None:
        self.input_layout = {}
        self.global_indices = None
        self.layout_cache = {}
        self.keep_count_schedule = []
        self.pyramid_lambda = 1.0
        self.runtime_stats = build_runtime_stats()
        self.runtime_stats["configured_max_pruning_layers"] = int(getattr(self.config, "max_pruning_layers", 0) or 0)
        self.runtime_stats["pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.pruning_loc]
        self.runtime_stats["candidate_pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.pruning_loc]
        self.runtime_stats["active_pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.pruning_loc]
        self.runtime_stats["selected_pruning_mode"] = "pyramiddrop_exponential_unified"
        self.runtime_stats["pyramiddrop_layer_records"] = []

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
        self.model._autoprune_patch = self
        self.model._pyramiddrop_patch = self

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
                    print(f"[MantisPyramidDrop] layout preparation failed: {exc}")
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
                return finish(
                    self._clean_outputs(outputs, requested_output_attentions, requested_output_attentions),
                    forward_elapsed=forward_elapsed,
                    decode_fastpath=is_decode_step,
                )

            align_t0 = time.perf_counter()
            attention_mask = align_attention_mask(self.global_indices, attention_mask, batch_size, seq_len)
            position_ids = align_position_ids(self.global_indices, position_ids, batch_size, seq_len)
            align_position_embeddings(self.global_indices, kwargs, batch_size, seq_len)
            self.runtime_stats["alignment_seconds"] += time.perf_counter() - align_t0

            pruning_layer = layer_idx in self.pruning_layer_set
            layout = None
            query_positions = torch.empty(0, dtype=torch.long, device=hidden_states.device)
            visual_positions = torch.empty(0, dtype=torch.long, device=hidden_states.device)
            if pruning_layer:
                layout_t0 = time.perf_counter()
                layout = self._resolve_cached_layout_state(seq_len, hidden_states.device)
                self.runtime_stats["layout_seconds"] += time.perf_counter() - layout_t0
                query_positions = layout["query_positions"]
                visual_positions = layout["visual_positions"]

            need_attentions = (
                pruning_layer
                and not self.config.disable_token_prune
                and not self.config.monitor_only
                and query_positions.numel() > 0
                and visual_positions.numel() > 0
            )
            original_output_attentions = kwargs.get("output_attentions", False)
            if need_attentions:
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

            if (
                not pruning_layer
                or self.config.disable_token_prune
                or self.config.monitor_only
                or layout is None
                or query_positions.numel() <= 0
                or visual_positions.numel() <= 0
            ):
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            attn_weights = outputs[1] if len(outputs) > 1 and original_output_attentions else consume_internal_attention(self_layer.self_attn)
            probe_stats = consume_internal_probe_stats(self_layer.self_attn)
            if probe_stats is not None:
                self.runtime_stats["probe_seconds"] += float(probe_stats.get("probe_seconds", 0.0))
            if attn_weights is None:
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            if not self.keep_count_schedule:
                schedule_t0 = time.perf_counter()
                stage_lengths = _build_runtime_stage_lengths(self.num_layers, self.pruning_loc)
                schedule = _build_exponential_keep_schedule(
                    initial_visual_tokens=int(visual_positions.numel()),
                    target_final_tokens=int(self.config.target_token_num),
                    stage_lengths=stage_lengths,
                    min_keep_tokens=int(self.config.min_keep_tokens),
                )
                self.pyramid_lambda = float(schedule["lambda"])
                self.keep_count_schedule = list(schedule["keep_schedule"])
                self.runtime_stats["budget_schedule_seconds"] += time.perf_counter() - schedule_t0
                self.runtime_stats["pyramiddrop_lambda"] = float(self.pyramid_lambda)
                self.runtime_stats["pyramiddrop_initial_visual_tokens"] = int(visual_positions.numel())
                self.runtime_stats["pyramiddrop_stage_lengths"] = [int(value) for value in schedule.get("stage_lengths", [])]
                self.runtime_stats["pyramiddrop_weighted_average_visual_tokens"] = float(
                    schedule.get("weighted_average_visual_tokens", 0.0)
                )
                self.runtime_stats["pyramiddrop_target_average_visual_tokens"] = float(self.config.target_token_num)
                self.runtime_stats["pyramiddrop_target_final_visual_tokens"] = int(self.config.target_token_num)
                self.runtime_stats["pyramiddrop_target_final_clamped"] = int(
                    schedule.get("target_final_clamped", self.config.target_token_num)
                )
                self.runtime_stats["pyramiddrop_target_clamped_by_bounds"] = int(
                    bool(schedule.get("target_clamped_by_bounds", False))
                )
                self.runtime_stats["configured_visual_budget_cap"] = int(self.config.target_token_num)
                self.runtime_stats["total_budget"] = int(self.config.target_token_num)
                self.runtime_stats["target_keep_per_view"] = [int(self.config.target_token_num)]
                self.runtime_stats["keep_count_schedule"] = [int(value) for value in self.keep_count_schedule]
                self.runtime_stats["dense_keep_count_schedule"] = [int(value) for value in self.keep_count_schedule]
                self.runtime_stats["dense_schedule_length"] = int(len(self.keep_count_schedule))
                self.runtime_stats["compressed_schedule_length"] = int(len(self.keep_count_schedule))

            schedule_idx = self.pruning_schedule_index.get(layer_idx, 0)
            schedule_idx = min(max(0, schedule_idx), len(self.keep_count_schedule) - 1)
            target_keep = int(self.keep_count_schedule[schedule_idx])
            target_keep = min(target_keep, int(visual_positions.numel()))
            if target_keep >= int(visual_positions.numel()):
                return finish(cleaned_outputs, forward_elapsed=forward_elapsed)

            score_t0 = time.perf_counter()
            selected_visual_positions = _select_unified_visual_indices(
                attn_weights,
                query_positions,
                visual_positions,
                target_keep,
                batch_size,
                hidden_states.device,
            )
            self.runtime_stats["topk_seconds"] += time.perf_counter() - score_t0

            full_indices = build_full_indices(
                batch_size=batch_size,
                seq_len=seq_len,
                always_keep_positions=layout["always_keep_positions"],
                selected_view_positions=selected_visual_positions,
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
            self.runtime_stats["pyramiddrop_last_visual_after"] = int(target_keep)
            self.runtime_stats.setdefault("pyramiddrop_layer_records", [])
            self.runtime_stats["pyramiddrop_layer_records"].append(
                {
                    "layer_idx": int(layer_idx),
                    "visual_before": int(visual_positions.numel()),
                    "visual_after": int(target_keep),
                    "tokens_before": int(seq_len),
                    "tokens_after": int(full_indices.shape[1]),
                }
            )
            flops_reduction = estimate_sparsevlm_flops_reduction(
                config=getattr(self.model, "config", None),
                num_layers=self.num_layers,
                layer_records=self.runtime_stats.get("pyramiddrop_layer_records", []),
                prefill_seq_len=self.runtime_stats.get("prefill_seq_len"),
            )
            self.runtime_stats["flops_reduction"] = flops_reduction
            self.runtime_stats["flops_reduction_ratio"] = flops_reduction

            if self.config.debug:
                print(
                    "[MantisPyramidDrop] "
                    f"layer={layer_idx}, lambda={self.pyramid_lambda:.6f}, "
                    f"visual_before={int(visual_positions.numel())}, visual_after={target_keep}, "
                    f"seq_before={seq_len}, seq_after={int(full_indices.shape[1])}"
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


def maybe_apply_mantis_pyramiddrop(
    model,
    config: Optional[MantisPyramidDropConfig],
) -> Optional[MantisPyramidDropPatch]:
    if config is None or not config.enabled:
        return None
    patch = MantisPyramidDropPatch(model, config)
    patch.apply()
    if config.disable_kv_cache:
        model.config.use_cache = False
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            model.language_model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
    model._autoprune_patch = patch
    model._pyramiddrop_patch = patch
    return patch


__all__ = [
    "MantisPyramidDropPatch",
    "maybe_apply_mantis_pyramiddrop",
]
