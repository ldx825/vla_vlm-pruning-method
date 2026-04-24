import math
import time
import types
from typing import Any, Dict, Optional

import torch

from .baseline_mantis_algorithms import (
    estimate_flops_reduction,
    fastervlm_prune_features,
    pyramiddrop_keep_schedule,
    visionzip_prune_features,
)
from .baseline_mantis_common import MantisBaselinePruneConfig, normalize_method_name
from .baseline_mantis_fastv_sparsevlm import (
    fastv_visual_scores,
    select_top_visual_positions,
    sparsevlm_adaptive_counts,
    sparsevlm_merge_rescued_tokens,
    sparsevlm_select_text_raters,
    sparsevlm_visual_scores,
)

try:
    from methods.auto_prune.mantis.autoprune_mantis_layout import (
        build_runtime_stats,
        prepare_input_layout_from_merge,
        resolve_layout_state,
        summarize_layout,
    )
    from methods.auto_prune.mantis.autoprune_mantis_pruning import (
        align_attention_mask,
        align_position_embeddings,
        align_position_ids,
        build_full_indices,
    )
except ImportError:  # pragma: no cover - supports copying these files into Mantis directly.
    from autoprune_mantis_layout import build_runtime_stats, prepare_input_layout_from_merge, resolve_layout_state, summarize_layout
    from autoprune_mantis_pruning import align_attention_mask, align_position_embeddings, align_position_ids, build_full_indices


def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compact_to_first_image_token(model, input_ids, inputs_embeds, attention_mask, labels):
    if input_ids.shape[0] != 1:
        return input_ids, inputs_embeds, attention_mask, labels
    image_token_index = int(model.config.image_token_index)
    image_positions = torch.where(input_ids[0] == image_token_index)[0]
    if image_positions.numel() <= 1:
        return input_ids, inputs_embeds, attention_mask, labels
    keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    keep[image_positions[1:]] = False
    new_input_ids = input_ids[:, keep]
    new_inputs_embeds = inputs_embeds[:, keep]
    new_attention_mask = attention_mask[:, keep] if attention_mask is not None else attention_mask
    new_labels = labels[:, keep] if labels is not None else labels
    return new_input_ids, new_inputs_embeds, new_attention_mask, new_labels


class MantisBaselinePrunePatch:
    def __init__(self, model, config: MantisBaselinePruneConfig):
        self.model = model
        self.config = config
        self.method = normalize_method_name(config.method)
        self.layers = self._resolve_layers(model)
        self.original_merge = None
        self.original_vision_forward = None
        self.original_projector_forward = None
        self.original_layer_forwards: Dict[int, Any] = {}
        self.input_layout: Dict[str, Any] = {}
        self.global_indices: Optional[torch.Tensor] = None
        self.layout_cache: Dict[str, Any] = {}
        self.pruning_layers = self._build_pruning_layers()
        self.keep_schedule = []
        self.reset_runtime_state()

    @staticmethod
    def _resolve_layers(model):
        for path in (("language_model", "model", "layers"), ("language_model", "layers"), ("model", "layers")):
            obj = model
            for attr in path:
                if not hasattr(obj, attr):
                    obj = None
                    break
                obj = getattr(obj, attr)
            if isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
                return obj
        return []

    def _build_progressive_layers(self, explicit_indices, start_layer, max_pruning_layers):
        if explicit_indices:
            return [idx for idx in explicit_indices if 0 <= idx < len(self.layers)]
        candidates = list(range(max(0, int(start_layer)), len(self.layers)))
        max_layers = int(max_pruning_layers)
        if max_layers <= 0 or max_layers >= len(candidates):
            return candidates
        if max_layers == 1:
            return candidates[:1]
        selected = []
        last = len(candidates) - 1
        for offset in range(max_layers):
            selected.append(candidates[int(round(offset * last / max(1, max_layers - 1)))])
        return sorted(set(selected))

    def _build_pruning_layers(self):
        if self.method == "fastv":
            # FastV paper step: run the first K layers densely, then prune once
            # after layer K. The public knob is 1-based for easier table logs.
            idx = max(0, int(self.config.fastv_pruning_layer) - 1)
            return [idx] if idx < len(self.layers) else []
        if self.method == "sparsevlm":
            return self._build_progressive_layers(
                self.config.sparsevlm_layer_indices,
                self.config.sparsevlm_start_layer,
                self.config.sparsevlm_max_pruning_layers,
            )
        if self.config.pyramiddrop_layer_indices:
            return [idx for idx in self.config.pyramiddrop_layer_indices if 0 <= idx < len(self.layers)]
        return self._build_progressive_layers(
            self.config.pyramiddrop_layer_indices,
            self.config.pyramiddrop_start_layer,
            self.config.pyramiddrop_max_pruning_layers,
        )

    def reset_runtime_state(self) -> None:
        self.input_layout = {}
        self.global_indices = None
        self.layout_cache = {}
        self.keep_schedule = []
        self.runtime_stats = build_runtime_stats()
        self.runtime_stats.update(
            {
                "method": self.method,
                "configured_target_token_num": int(self.config.target_token_num),
                "method_overhead_seconds": 0.0,
                "pre_projector": int(self.method in {"fastervlm", "visionzip"}),
                "pre_llm_prefill": 1,
                "unified_visual_pruning": 1,
                "fastv_pruning_layer": int(self.config.fastv_pruning_layer),
                "pyramiddrop_lambda": None,
                "pyramiddrop_layer_indices": [int(idx) for idx in self.pruning_layers] if self.method == "pyramiddrop" else [],
                "sparsevlm_layer_indices": [int(idx) for idx in self.pruning_layers] if self.method == "sparsevlm" else [],
                "active_pruning_layer_indices": [int(idx) for idx in self.pruning_layers],
                "sparsevlm_recycle_top_ratio": float(self.config.sparsevlm_recycle_top_ratio),
                "sparsevlm_cluster_ratio": float(self.config.sparsevlm_cluster_ratio),
                "flops_reduction": 0.0,
            }
        )

    def apply(self) -> None:
        if not self.config.enabled:
            return
        if hasattr(self.model, "vision_tower"):
            self.original_vision_forward = self.model.vision_tower.forward
            self.model.vision_tower.forward = types.MethodType(self._make_vision_forward(), self.model.vision_tower)
        if self.method in {"fastervlm", "visionzip"} and hasattr(self.model, "multi_modal_projector"):
            self.original_projector_forward = self.model.multi_modal_projector.forward
            self.model.multi_modal_projector.forward = types.MethodType(
                self._make_projector_forward(),
                self.model.multi_modal_projector,
            )
        if hasattr(self.model, "_merge_input_ids_with_image_features"):
            self.original_merge = self.model._merge_input_ids_with_image_features
            self.model._merge_input_ids_with_image_features = types.MethodType(self._make_merge(), self.model)
        if self.method in {"pyramiddrop", "fastv", "sparsevlm"}:
            for layer_idx, layer in enumerate(self.layers):
                self.original_layer_forwards[layer_idx] = layer.forward
                layer.forward = types.MethodType(self._make_layer_forward(layer_idx), layer)
        self.model._baseline_prune_patch = self
        # Reuse the AutoPrune stats hook name so existing BLINK/Mantis eval code
        # can collect latency, FLOPs proxy, and last-token counts uniformly.
        self.model._autoprune_patch = self

    def _make_vision_forward(self):
        def new_forward(vision_self, *args, **kwargs):
            kwargs["output_hidden_states"] = True
            kwargs["output_attentions"] = True
            try:
                outputs = self.original_vision_forward(*args, **kwargs)
            except TypeError:
                kwargs.pop("output_attentions", None)
                outputs = self.original_vision_forward(*args, **kwargs)
            self.model._baseline_last_vision_outputs = outputs
            return outputs

        return new_forward

    def _flatten_cls_attention(self, image_features: torch.Tensor) -> Optional[torch.Tensor]:
        outputs = getattr(self.model, "_baseline_last_vision_outputs", None)
        if outputs is None or getattr(outputs, "attentions", None) is None or not outputs.attentions:
            return None
        attn = outputs.attentions[-1]
        if attn is None:
            return None
        # Every image is treated as one slice of a single visual pool. This is
        # intentionally not the AutoPrune per-view policy.
        patch_count = int(image_features.shape[1])
        scores = attn[:, :, 0, 1 : 1 + patch_count].mean(dim=1)
        return scores.reshape(1, -1).to(device=image_features.device, dtype=image_features.dtype)

    def _apply_pre_projector_pruning(self, selected_features: torch.Tensor) -> torch.Tensor:
        before = int(selected_features.shape[0] * selected_features.shape[1])
        flat = selected_features.reshape(1, before, selected_features.shape[-1])
        scores = self._flatten_cls_attention(selected_features)
        t0 = time.perf_counter()
        if self.method == "fastervlm":
            pruned, extra = fastervlm_prune_features(flat, scores, self.config.target_token_num)
        elif self.method == "visionzip":
            pruned, extra = visionzip_prune_features(
                flat,
                scores,
                self.config.target_token_num,
                dominant_ratio=self.config.visionzip_dominant_ratio,
            )
        else:
            return selected_features
        elapsed = time.perf_counter() - t0
        after = int(pruned.shape[1])
        self.runtime_stats["method_overhead_seconds"] += elapsed
        self.runtime_stats["gather_seconds"] += elapsed
        self.runtime_stats["pruned_layers"] = 1
        self.runtime_stats["last_tokens_before"] = before
        self.runtime_stats["last_tokens_after"] = after
        self.runtime_stats["visual_tokens_before"] = before
        self.runtime_stats["visual_tokens_after"] = after
        self.runtime_stats["flops_reduction"] = estimate_flops_reduction(before, before, after)
        self.runtime_stats.update(extra)
        if self.config.debug:
            print(f"[MantisBaseline:{self.method}] visual tokens {before} -> {after}")
        return pruned

    def _make_projector_forward(self):
        def new_projector_forward(projector_self, selected_features):
            self.reset_runtime_state()
            if selected_features.dim() == 3:
                _cuda_sync_if_needed(selected_features.device)
                selected_features = self._apply_pre_projector_pruning(selected_features)
                _cuda_sync_if_needed(selected_features.device)
            return self.original_projector_forward(selected_features)

        return new_projector_forward

    def _make_merge(self):
        def new_merge(model_self, image_features, inputs_embeds, input_ids, attention_mask, labels):
            if self.method in {"pyramiddrop", "fastv", "sparsevlm"}:
                self.reset_runtime_state()
            if self.method in {"fastervlm", "visionzip"}:
                input_ids, inputs_embeds, attention_mask, labels = _compact_to_first_image_token(
                    model_self, input_ids, inputs_embeds, attention_mask, labels
                )
            outputs = self.original_merge(image_features, inputs_embeds, input_ids, attention_mask, labels)
            try:
                visual_tokens = int(image_features.shape[1])
                self.input_layout = prepare_input_layout_from_merge(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_token_index=int(model_self.config.image_token_index),
                    visual_tokens_per_view=visual_tokens,
                )
                self.runtime_stats.update(summarize_layout(self.input_layout))
                if isinstance(outputs, tuple) and outputs and isinstance(outputs[0], torch.Tensor):
                    self.runtime_stats["prefill_seq_len"] = int(outputs[0].shape[1])
                    if self.method in {"fastervlm", "visionzip"}:
                        self.runtime_stats["flops_reduction"] = estimate_flops_reduction(
                            int(outputs[0].shape[1]) + max(0, int(self.runtime_stats["last_tokens_before"]) - int(self.runtime_stats["last_tokens_after"])),
                            int(self.runtime_stats["last_tokens_before"]),
                            int(self.runtime_stats["last_tokens_after"]),
                        )
            except Exception as exc:
                if self.config.debug:
                    print(f"[MantisBaseline:{self.method}] layout failed: {exc}")
            return outputs

        return new_merge

    def _resolve_layout_state(self, seq_len: int, device: torch.device) -> Dict[str, Any]:
        key = (id(self.global_indices), seq_len, str(device))
        cached = self.layout_cache.get("entry")
        if cached is not None and cached.get("key") == key:
            self.runtime_stats["layout_cache_hits"] += 1
            return cached["layout"]
        layout = resolve_layout_state(self.input_layout, self.global_indices, seq_len, device)
        self.layout_cache["entry"] = {"key": key, "layout": layout}
        self.runtime_stats["layout_recompute_count"] += 1
        return layout

    def _select_pyramiddrop_indices(self, hidden_states, layer, attention_mask, position_ids, visual_positions, keep_count):
        if keep_count >= int(visual_positions.numel()):
            return visual_positions.unsqueeze(0).expand(hidden_states.shape[0], -1)
        # PyramidDrop scores visual tokens at stage boundaries with a text query.
        # We use the final valid text token as the query, matching the paper's
        # instruction-token attention idea while keeping all images as one pool.
        with torch.no_grad():
            normed = layer.input_layernorm(hidden_states)
            q = layer.self_attn.q_proj(normed)
            k = layer.self_attn.k_proj(normed)
            bsz, seq_len, _ = hidden_states.shape
            num_heads = layer.self_attn.num_heads
            head_dim = layer.self_attn.head_dim
            num_kv_heads = layer.self_attn.num_key_value_heads
            q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            if num_kv_heads != num_heads:
                repeat = max(1, num_heads // max(1, num_kv_heads))
                k = k.repeat_interleave(repeat, dim=1)
            if attention_mask is not None and attention_mask.dim() == 2:
                valid = attention_mask.bool()
                query_pos = valid.long().sum(dim=1).sub(1).clamp(min=0)
            else:
                query_pos = torch.full((bsz,), seq_len - 1, dtype=torch.long, device=hidden_states.device)
            batch = torch.arange(bsz, device=hidden_states.device)
            q_text = q[batch, :, query_pos, :].unsqueeze(2)
            k_visual = k.index_select(2, visual_positions)
            score = torch.matmul(q_text, k_visual.transpose(-1, -2)).squeeze(2).mean(dim=1) / math.sqrt(head_dim)
            score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
            _, topk = torch.topk(score, int(keep_count), dim=1)
            base = visual_positions.unsqueeze(0).expand(bsz, -1)
            return torch.sort(base.gather(1, topk), dim=1).values

    def _record_pruning_stats(self, *, seq_len: int, new_seq_len: int, visual_before: int, visual_after: int) -> None:
        self.runtime_stats["method_overhead_seconds"] = (
            float(self.runtime_stats.get("topk_seconds", 0.0))
            + float(self.runtime_stats.get("gather_seconds", 0.0))
            + float(self.runtime_stats.get("rank_seconds", 0.0))
        )
        self.runtime_stats["pruned_layers"] += 1
        self.runtime_stats["last_tokens_before"] = int(seq_len)
        self.runtime_stats["last_tokens_after"] = int(new_seq_len)
        self.runtime_stats["visual_tokens_before"] = int(visual_before)
        self.runtime_stats["visual_tokens_after"] = int(visual_after)
        self.runtime_stats["flops_reduction"] = estimate_flops_reduction(seq_len, int(visual_before), int(visual_after))

    def _current_global_indices(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.global_indices is None:
            self.global_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        current = self.global_indices.to(device=device, dtype=torch.long)
        if current.shape[0] == 1 and batch_size > 1:
            current = current.expand(batch_size, -1)
        return current

    def _rebuild_sparsevlm_sequence(self, hidden_states, visual_positions, selected_visual, dropped_visual, dropped_scores):
        bsz, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        current_globals = self._current_global_indices(bsz, seq_len, device)

        visual_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        visual_mask[visual_positions.to(device=device, dtype=torch.long)] = True
        non_visual_pos = torch.where(~visual_mask)[0]
        non_visual_hidden = hidden_states.index_select(1, non_visual_pos)
        non_visual_global = current_globals.index_select(1, non_visual_pos)

        selected_visual = selected_visual.to(device=device, dtype=torch.long)
        kept_hidden = hidden_states.gather(1, selected_visual.unsqueeze(-1).expand(-1, -1, hidden_size))
        kept_global = current_globals.gather(1, selected_visual)

        merged_hidden, merged_global = sparsevlm_merge_rescued_tokens(
            hidden_states,
            current_globals,
            dropped_visual.to(device=device, dtype=torch.long),
            dropped_scores,
            recycle_top_ratio=self.config.sparsevlm_recycle_top_ratio,
            cluster_ratio=self.config.sparsevlm_cluster_ratio,
        )

        # Real sequence shortening: concatenate surviving non-visual tokens,
        # selected visual tokens, and SparseVLM recycled contextual tokens, then
        # sort by original global index so later attention masks/position ids can
        # be aligned with the same AutoPrune machinery.
        candidate_hidden = torch.cat([non_visual_hidden, kept_hidden, merged_hidden], dim=1)
        candidate_global = torch.cat([non_visual_global, kept_global, merged_global], dim=1)
        order = torch.argsort(candidate_global, dim=1)
        new_hidden = candidate_hidden.gather(1, order.unsqueeze(-1).expand(-1, -1, hidden_size)).contiguous()
        new_global = candidate_global.gather(1, order).contiguous()
        return new_hidden, new_global, int(kept_hidden.shape[1] + merged_hidden.shape[1])

    @staticmethod
    def _outputs_with_hidden(outputs, new_hidden, *, forced_attentions: bool, original_output_attentions: bool, original_use_cache: bool):
        if not forced_attentions or original_output_attentions:
            return (new_hidden,) + tuple(outputs[1:])
        if original_use_cache and len(outputs) >= 3:
            return (new_hidden, outputs[2])
        return (new_hidden,)

    def _make_layer_forward(self, layer_idx: int):
        def new_forward(layer_self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            patch_t0 = time.perf_counter()
            bsz, seq_len, _ = hidden_states.shape
            use_cache = bool(kwargs.get("use_cache", False))
            is_decode = bool(use_cache and seq_len <= 1)
            original_output_attentions = bool(kwargs.get("output_attentions", False))
            need_attentions = (not is_decode) and self.method in {"fastv", "sparsevlm"} and layer_idx in self.pruning_layers
            if need_attentions and not original_output_attentions:
                kwargs["output_attentions"] = True

            if layer_idx == 0:
                if not is_decode or self.global_indices is None:
                    self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
                    self.layout_cache = {}
                if self.runtime_stats.get("prefill_seq_len") is None and seq_len > 1:
                    self.runtime_stats["prefill_seq_len"] = int(seq_len)

            if not is_decode:
                attention_mask = align_attention_mask(self.global_indices, attention_mask, bsz, seq_len)
                position_ids = align_position_ids(self.global_indices, position_ids, bsz, seq_len)
                align_position_embeddings(self.global_indices, kwargs, bsz, seq_len)

            forward_t0 = time.perf_counter()
            outputs = self.original_layer_forwards[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            forward_elapsed = time.perf_counter() - forward_t0
            patch_elapsed = time.perf_counter() - patch_t0
            self.runtime_stats["layer_call_count"] += 1
            self.runtime_stats["patch_total_seconds"] += patch_elapsed
            self.runtime_stats["layer_forward_seconds"] += forward_elapsed
            self.runtime_stats["patch_overhead_seconds"] += max(0.0, patch_elapsed - forward_elapsed)
            if use_cache:
                self.runtime_stats["cache_enabled_layer_calls"] += 1
            if seq_len <= 1:
                self.runtime_stats["seq_len_one_layer_calls"] += 1
            else:
                self.runtime_stats["seq_len_gt_one_layer_calls"] += 1
            if is_decode:
                self.runtime_stats["decode_layer_calls"] += 1
                self.runtime_stats["decode_patch_total_seconds"] += patch_elapsed
                self.runtime_stats["decode_layer_forward_seconds"] += forward_elapsed
                self.runtime_stats["decode_patch_overhead_seconds"] += max(0.0, patch_elapsed - forward_elapsed)
            else:
                self.runtime_stats["prefill_layer_calls"] += 1
                self.runtime_stats["prefill_patch_total_seconds"] += patch_elapsed
                self.runtime_stats["prefill_layer_forward_seconds"] += forward_elapsed
                self.runtime_stats["prefill_patch_overhead_seconds"] += max(0.0, patch_elapsed - forward_elapsed)

            def clean_outputs(hidden_override=None):
                return self._outputs_with_hidden(
                    outputs,
                    outputs[0] if hidden_override is None else hidden_override,
                    forced_attentions=need_attentions,
                    original_output_attentions=original_output_attentions,
                    original_use_cache=use_cache,
                )

            if is_decode or layer_idx not in self.pruning_layers:
                return outputs

            layout = self._resolve_layout_state(seq_len, hidden_states.device)
            visual_positions = layout.get("all_visual_positions")
            if visual_positions is None or visual_positions.numel() <= 0:
                return clean_outputs()

            if self.method == "fastv":
                if int(visual_positions.numel()) <= int(self.config.target_token_num):
                    return clean_outputs()
                score_t0 = time.perf_counter()
                attentions = outputs[1] if len(outputs) > 1 else None
                if attentions is None:
                    return clean_outputs()
                scores = fastv_visual_scores(attentions, visual_positions, seq_len)
                selected, keep_count = select_top_visual_positions(scores, visual_positions, self.config.target_token_num)
                self.runtime_stats["topk_seconds"] += time.perf_counter() - score_t0
                full_indices = build_full_indices(
                    batch_size=bsz,
                    seq_len=seq_len,
                    always_keep_positions=layout["always_keep_positions"],
                    selected_view_positions=[selected],
                    device=hidden_states.device,
                )
                if full_indices.shape[1] >= seq_len:
                    return clean_outputs()
                gather_t0 = time.perf_counter()
                new_hidden = outputs[0].gather(1, full_indices.unsqueeze(-1).expand(-1, -1, outputs[0].size(-1))).contiguous()
                self.global_indices = self._current_global_indices(bsz, seq_len, hidden_states.device).gather(1, full_indices)
                self.layout_cache = {}
                self.runtime_stats["gather_seconds"] += time.perf_counter() - gather_t0
                self._record_pruning_stats(
                    seq_len=seq_len,
                    new_seq_len=int(full_indices.shape[1]),
                    visual_before=int(visual_positions.numel()),
                    visual_after=int(keep_count),
                )
                self.runtime_stats["fastv_attention_includes_all_nonvisual_tokens"] = 1
                if self.config.debug:
                    print(f"[MantisBaseline:fastv] layer={layer_idx} seq {seq_len}->{full_indices.shape[1]}, visual={keep_count}")
                return clean_outputs(new_hidden)

            if self.method == "sparsevlm":
                if int(visual_positions.numel()) <= int(self.config.target_token_num):
                    return clean_outputs()
                score_t0 = time.perf_counter()
                attentions = outputs[1] if len(outputs) > 1 else None
                if attentions is None:
                    return clean_outputs()
                text_positions = layout.get("query_positions", torch.empty(0, dtype=torch.long, device=hidden_states.device))
                rater_positions = sparsevlm_select_text_raters(outputs[0], visual_positions, text_positions)
                scores = sparsevlm_visual_scores(attentions, visual_positions, rater_positions)
                self.runtime_stats["topk_seconds"] += time.perf_counter() - score_t0

                rank_t0 = time.perf_counter()
                stage = self.pruning_layers.index(layer_idx)
                remaining_layers = max(1, len(self.pruning_layers) - stage)
                keep_count, drop_count, extra = sparsevlm_adaptive_counts(
                    outputs[0],
                    visual_positions,
                    int(self.config.target_token_num),
                    remaining_layers,
                    rank_tolerance=float(self.config.sparsevlm_rank_tolerance),
                    min_prune_tokens=int(self.config.sparsevlm_min_prune_tokens),
                )
                self.runtime_stats["rank_seconds"] += time.perf_counter() - rank_t0
                self.runtime_stats.update(extra)
                if drop_count <= 0 or keep_count >= int(visual_positions.numel()):
                    return clean_outputs()

                _, order = torch.sort(scores, dim=1, descending=True)
                base = visual_positions.to(device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
                selected = torch.sort(base.gather(1, order[:, :keep_count]), dim=1).values
                dropped = base.gather(1, order[:, keep_count:])
                dropped_score = scores.gather(1, order[:, keep_count:])

                gather_t0 = time.perf_counter()
                new_hidden, new_global, visual_after = self._rebuild_sparsevlm_sequence(
                    outputs[0],
                    visual_positions,
                    selected,
                    dropped[0],
                    dropped_score,
                )
                if new_hidden.shape[1] >= seq_len:
                    return clean_outputs()
                self.global_indices = new_global
                self.layout_cache = {}
                self.runtime_stats["gather_seconds"] += time.perf_counter() - gather_t0
                self._record_pruning_stats(
                    seq_len=seq_len,
                    new_seq_len=int(new_hidden.shape[1]),
                    visual_before=int(visual_positions.numel()),
                    visual_after=int(visual_after),
                )
                self.runtime_stats["sparsevlm_text_raters"] = int(rater_positions.numel())
                self.runtime_stats["sparsevlm_recycled_tokens"] = max(0, int(visual_after) - int(keep_count))
                self.runtime_stats["sparsevlm_text_only_raters"] = 1
                if self.config.debug:
                    print(f"[MantisBaseline:sparsevlm] layer={layer_idx} seq {seq_len}->{new_hidden.shape[1]}, visual={visual_after}")
                return clean_outputs(new_hidden)

            if not self.keep_schedule:
                self.keep_schedule, lam = pyramiddrop_keep_schedule(
                    int(visual_positions.numel()),
                    int(self.config.target_token_num),
                    len(self.pruning_layers),
                )
                self.runtime_stats["pyramiddrop_lambda"] = float(lam)
                self.runtime_stats["keep_count_schedule"] = [int(v) for v in self.keep_schedule]

            stage = self.pruning_layers.index(layer_idx)
            keep_count = int(self.keep_schedule[min(stage, len(self.keep_schedule) - 1)])
            if keep_count >= int(visual_positions.numel()):
                return outputs

            score_t0 = time.perf_counter()
            selected = self._select_pyramiddrop_indices(
                outputs[0],
                layer_self,
                attention_mask,
                position_ids,
                visual_positions,
                keep_count,
            )
            self.runtime_stats["topk_seconds"] += time.perf_counter() - score_t0
            full_indices = build_full_indices(
                batch_size=bsz,
                seq_len=seq_len,
                always_keep_positions=layout["always_keep_positions"],
                selected_view_positions=[selected],
                device=hidden_states.device,
            )
            if full_indices.shape[1] >= seq_len:
                return outputs

            gather_t0 = time.perf_counter()
            new_hidden = outputs[0].gather(1, full_indices.unsqueeze(-1).expand(-1, -1, outputs[0].size(-1))).contiguous()
            self.global_indices = self._current_global_indices(bsz, seq_len, hidden_states.device).gather(1, full_indices)
            self.layout_cache = {}
            self.runtime_stats["gather_seconds"] += time.perf_counter() - gather_t0
            self._record_pruning_stats(
                seq_len=seq_len,
                new_seq_len=int(full_indices.shape[1]),
                visual_before=int(visual_positions.numel()),
                visual_after=int(keep_count),
            )
            if self.config.debug:
                print(f"[MantisBaseline:pyramiddrop] layer={layer_idx} seq {seq_len}->{full_indices.shape[1]}, visual={keep_count}")
            return (new_hidden,) + tuple(outputs[1:])

        return new_forward

    def get_runtime_stats(self) -> Dict[str, Any]:
        return dict(self.runtime_stats)

    def get_flops_reduction(self) -> float:
        return float(self.runtime_stats.get("flops_reduction", 0.0) or 0.0)


def maybe_apply_mantis_baseline_pruning(
    model,
    config: Optional[MantisBaselinePruneConfig],
) -> Optional[MantisBaselinePrunePatch]:
    if config is None or not config.enabled:
        return None
    patch = MantisBaselinePrunePatch(model, config)
    patch.apply()
    if config.disable_kv_cache:
        model.config.use_cache = False
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            model.language_model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
    return patch


__all__ = ["MantisBaselinePrunePatch", "maybe_apply_mantis_baseline_pruning"]
