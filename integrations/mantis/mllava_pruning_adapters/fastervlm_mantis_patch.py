import time
import types
from typing import Any, Dict, Optional

import torch

from .autoprune_mantis_layout import (
    build_runtime_stats,
    prepare_input_layout_from_merge,
    summarize_layout,
)
from .fastervlm_mantis_common import MantisFasterVLMConfig
from .fastervlm_mantis_pruning import (
    estimate_fastv_flops_reduction,
    select_fastervlm_visual_tokens,
    select_fastervlm_unified_visual_tokens,
)


class MantisFasterVLMPatch:
    def __init__(self, model, config: MantisFasterVLMConfig):
        self.model = model
        self.config = config
        self.layers = self._resolve_layers(model)
        self.original_layer_forwards: Dict[int, Any] = {}
        self.original_model_forward = None
        self.original_vision_forward = None
        self.original_projector_forward = None
        self.original_merge = None
        self.num_layers = len(self.layers)
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
        raise RuntimeError("Unable to locate Mantis decoder layers for FasterVLM.")

    def reset_runtime_state(self) -> None:
        self.input_layout: Dict[str, Any] = {}
        self.global_indices: Optional[torch.Tensor] = None
        self.runtime_stats = build_runtime_stats()
        self.runtime_stats["pruning_layer_indices"] = [0]
        self.runtime_stats["candidate_pruning_layer_indices"] = [0]
        self.runtime_stats["active_pruning_layer_indices"] = [0]
        self.runtime_stats["selected_pruning_mode"] = "fastervlm_cls_visual_prefilter_unified"
        self.runtime_stats["configured_visual_budget_cap"] = int(self.config.target_token_num)
        self._pending_vision_attn: Optional[torch.Tensor] = None
        self._prefill_open = False
        self._visual_tokens_before_per_image: Optional[int] = None
        self._visual_tokens_after_per_image: Optional[int] = None
        self._total_visual_tokens_before = 0
        self._total_visual_tokens_after = 0
        self._projector_calls = 0
        self._pending_image_token_counts = None
        self._accumulated_image_features = []
        self._accumulated_vision_attn = []
        self.has_pruned = False

    def apply(self) -> None:
        if not self.config.enabled:
            return
        self.original_model_forward = self.model.forward
        self.model.forward = types.MethodType(self._make_new_model_forward(), self.model)
        if not self.config.scaffold_only and hasattr(self.model, "vision_tower"):
            self.original_vision_forward = self.model.vision_tower.forward
            self.model.vision_tower.forward = types.MethodType(self._make_new_vision_forward(), self.model.vision_tower)
        if not self.config.scaffold_only and hasattr(self.model, "multi_modal_projector"):
            self.original_projector_forward = self.model.multi_modal_projector.forward
            self.model.multi_modal_projector.forward = types.MethodType(
                self._make_new_projector_forward(),
                self.model.multi_modal_projector,
            )
        if not self.config.scaffold_only and hasattr(self.model, "_merge_input_ids_with_image_features"):
            self.original_merge = self.model._merge_input_ids_with_image_features
            self.model._merge_input_ids_with_image_features = types.MethodType(self._make_new_merge(), self.model)

        for layer_idx, layer in enumerate(self.layers):
            self.original_layer_forwards[layer_idx] = layer.forward
            layer.forward = types.MethodType(self._make_new_forward(layer_idx), layer)

        self.model._fastervlm_patch = self

    def _make_new_model_forward(self):
        def new_model_forward(model_self, *args, **kwargs):
            input_ids = kwargs.get("input_ids", None)
            pixel_values = kwargs.get("pixel_values", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            if pixel_values is None and len(args) > 1:
                pixel_values = args[1]

            self._pending_image_token_counts = None
            self._accumulated_image_features = []
            self._accumulated_vision_attn = []
            try:
                if (
                    input_ids is not None
                    and pixel_values is not None
                    and getattr(input_ids, "ndim", 0) == 2
                    and int(input_ids.shape[1]) != 1
                ):
                    counts = (input_ids == int(model_self.config.image_token_index)).sum(dim=-1)
                    self._pending_image_token_counts = [int(value) for value in counts.detach().cpu().tolist()]
            except Exception as exc:
                self._pending_image_token_counts = None
                if self.config.debug:
                    print(f"[MantisFasterVLM] failed to capture image-token counts: {exc}")
            return self.original_model_forward(*args, **kwargs)

        return new_model_forward

    def _make_new_vision_forward(self):
        def new_vision_forward(vision_self, *args, **kwargs):
            if not self._prefill_open:
                pending_counts = self._pending_image_token_counts
                self.reset_runtime_state()
                self._pending_image_token_counts = pending_counts
                self._prefill_open = True

            # FasterVLM Step 2: request visual encoder attentions so the patch
            # budget can be decided from final-layer [CLS] -> patch attention.
            kwargs["output_attentions"] = True
            kwargs["output_hidden_states"] = True
            output = self.original_vision_forward(*args, **kwargs)
            try:
                attentions = getattr(output, "attentions", None)
                self._pending_vision_attn = attentions[-1].detach() if attentions is not None and len(attentions) > 0 else None
            except Exception as exc:
                self._pending_vision_attn = None
                if self.config.debug:
                    print(f"[MantisFasterVLM] failed to capture vision attention: {exc}")
            return output

        return new_vision_forward

    def _make_new_projector_forward(self):
        def new_projector_forward(projector_self, image_features):
            if self.config.disable_token_prune or self._pending_vision_attn is None or image_features.ndim != 3:
                return self.original_projector_forward(image_features)

            prune_t0 = time.perf_counter()
            visual_tokens_before = int(image_features.shape[1])
            counts = self._pending_image_token_counts
            if counts is None or sum(counts) <= 0:
                selected = select_fastervlm_visual_tokens(
                    self._pending_vision_attn,
                    visual_tokens_before=visual_tokens_before,
                    keep_visual_tokens=max(1, min(int(self.config.target_token_num), visual_tokens_before)),
                    device=image_features.device,
                )
                if selected is None:
                    return self.original_projector_forward(image_features)
                batch_images = int(image_features.shape[0])
                if selected.shape[0] == 1 and batch_images > 1:
                    selected = selected.expand(batch_images, -1)
                if selected.shape[0] != batch_images:
                    return self.original_projector_forward(image_features)
                pruned_features = image_features.gather(
                    1,
                    selected.unsqueeze(-1).expand(-1, -1, image_features.shape[-1]),
                ).contiguous()
                projected = self.original_projector_forward(pruned_features)
                self._record_projector_prune_stats(
                    elapsed=time.perf_counter() - prune_t0,
                    visual_tokens_before=visual_tokens_before,
                    total_before=visual_tokens_before * batch_images,
                    visual_tokens_after=int(pruned_features.shape[1]),
                    total_after=int(pruned_features.shape[1]) * batch_images,
                    keep_visual_tokens=int(pruned_features.shape[1]),
                )
                return projected

            self._accumulated_image_features.append(image_features)
            if self._pending_vision_attn is not None:
                self._accumulated_vision_attn.append(self._pending_vision_attn.to(device=image_features.device))

            collected_images = sum(int(chunk.shape[0]) for chunk in self._accumulated_image_features)
            total_images = int(sum(counts))
            if collected_images < total_images:
                min_sample_visual_tokens = min(max(1, int(image_count)) * visual_tokens_before for image_count in counts)
                target_tokens = max(1, min(int(self.config.target_token_num), int(min_sample_visual_tokens)))
                empty_features = image_features.new_empty((0, target_tokens, image_features.shape[-1]))
                return self.original_projector_forward(empty_features)
            if collected_images != total_images:
                self._accumulated_image_features = []
                self._accumulated_vision_attn = []
                return self.original_projector_forward(image_features)

            all_image_features = torch.cat(self._accumulated_image_features, dim=0)
            all_vision_attn = torch.cat(self._accumulated_vision_attn, dim=0) if self._accumulated_vision_attn else None
            self._accumulated_image_features = []
            self._accumulated_vision_attn = []

            min_sample_visual_tokens = min(max(1, int(image_count)) * visual_tokens_before for image_count in counts)
            keep_visual_tokens = max(1, min(int(self.config.target_token_num), int(min_sample_visual_tokens)))
            pooled_blocks = []
            offset = 0
            for image_count in counts:
                image_count = int(image_count)
                sample_features = all_image_features[offset : offset + image_count]
                sample_attn = (
                    all_vision_attn[offset : offset + image_count]
                    if all_vision_attn is not None and all_vision_attn.shape[0] >= offset + image_count
                    else None
                )
                flat_selected = select_fastervlm_unified_visual_tokens(
                    sample_attn,
                    visual_tokens_before=visual_tokens_before,
                    keep_visual_tokens=keep_visual_tokens,
                    device=all_image_features.device,
                )
                if flat_selected is None:
                    flat_selected = torch.arange(
                        keep_visual_tokens,
                        dtype=torch.long,
                        device=all_image_features.device,
                    ).unsqueeze(0)
                # FasterVLM Steps 3-5, adapted to BLINK multi-view: every image
                # in the sample forms one flattened visual pool, so top-k is run
                # once over image_count * patch_count before the projector.
                flat_features = sample_features.reshape(1, image_count * visual_tokens_before, image_features.shape[-1])
                pooled_blocks.append(
                    flat_features.gather(
                        1,
                        flat_selected.unsqueeze(-1).expand(-1, -1, image_features.shape[-1]),
                    ).contiguous()
                )
                offset += image_count

            pruned_features = torch.cat(pooled_blocks, dim=0)
            projected = self.original_projector_forward(pruned_features)
            self._record_projector_prune_stats(
                elapsed=time.perf_counter() - prune_t0,
                visual_tokens_before=visual_tokens_before,
                total_before=visual_tokens_before * total_images,
                visual_tokens_after=int(pruned_features.shape[1]),
                total_after=int(pruned_features.shape[1]) * len(counts),
                keep_visual_tokens=int(pruned_features.shape[1]),
            )
            return projected

        return new_projector_forward

    def _record_projector_prune_stats(
        self,
        *,
        elapsed: float,
        visual_tokens_before: int,
        total_before: int,
        visual_tokens_after: int,
        total_after: int,
        keep_visual_tokens: int,
    ) -> None:
        self._projector_calls += 1
        self._visual_tokens_before_per_image = int(visual_tokens_before)
        self._visual_tokens_after_per_image = int(visual_tokens_after)
        self._total_visual_tokens_before += int(total_before)
        self._total_visual_tokens_after += int(total_after)
        self.has_pruned = self.has_pruned or int(total_after) < int(total_before)
        reduction_ratio = 1.0 - float(keep_visual_tokens) / float(max(1, int(total_before)))

        stats = self.runtime_stats
        stats["topk_seconds"] += float(elapsed)
        stats["gather_seconds"] += float(elapsed)
        stats["patch_overhead_seconds"] += float(elapsed)
        stats["prefill_patch_overhead_seconds"] += float(elapsed)
        stats["raw_total_budget"] = float(keep_visual_tokens)
        stats["total_budget"] = int(keep_visual_tokens)
        stats["target_keep_per_view"] = [int(keep_visual_tokens)]
        stats["keep_count_schedule"] = [[int(keep_visual_tokens)]]
        stats["dense_keep_count_schedule"] = [[int(keep_visual_tokens)]]
        stats["dense_schedule_length"] = 1
        stats["compressed_schedule_length"] = 1
        stats["dense_pruning_layer_indices"] = [0]
        stats["fastervlm_visual_tokens_before"] = int(total_before)
        stats["fastervlm_visual_tokens_after"] = int(visual_tokens_after)
        stats["fastervlm_total_visual_tokens_before"] = int(self._total_visual_tokens_before)
        stats["fastervlm_total_visual_tokens_after"] = int(self._total_visual_tokens_after)
        stats["fastervlm_keep_visual_tokens"] = int(keep_visual_tokens)
        stats["fastervlm_reduction_ratio_R"] = float(reduction_ratio)
        stats["pruned_layers"] = int(self.has_pruned)

    def _make_new_merge(self):
        def new_merge(model_self, image_features, inputs_embeds, input_ids, attention_mask, labels):
            if self._pending_image_token_counts is not None and image_features.ndim == 3:
                outputs = self._merge_pooled_visual_features(
                    image_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels,
                    int(model_self.config.image_token_index),
                    int(model_self.config.ignore_index),
                )
            else:
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
                    tokens_after = int(outputs[0].shape[1])
                    removed = max(0, int(self._total_visual_tokens_before) - int(self._total_visual_tokens_after))
                    tokens_before = int(tokens_after + removed)
                    self.runtime_stats["prefill_seq_len"] = tokens_after
                    self.runtime_stats["last_tokens_before"] = tokens_before
                    self.runtime_stats["last_tokens_after"] = tokens_after
                    flops_reduction = estimate_fastv_flops_reduction(
                        config=getattr(self.model, "config", None),
                        num_layers=self.num_layers,
                        start_layer=0,
                        tokens_before=tokens_before,
                        tokens_after=tokens_after,
                    )
                    self.runtime_stats["flops_reduction"] = flops_reduction
                    self.runtime_stats["flops_reduction_ratio"] = flops_reduction
            except Exception as exc:
                self.input_layout = {}
                if self.config.debug:
                    print(f"[MantisFasterVLM] layout preparation failed: {exc}")
            finally:
                self._prefill_open = False
            return outputs

        return new_merge

    def _merge_pooled_visual_features(
        self,
        image_features,
        inputs_embeds,
        input_ids,
        attention_mask,
        labels,
        image_token_index: int,
        ignore_index: int,
    ):
        batch_size, sequence_length = input_ids.shape
        _, visual_tokens, embed_dim = image_features.shape
        device = inputs_embeds.device
        attention_mask = attention_mask.to(device)
        image_features = image_features.to(device)

        rows = []
        masks = []
        label_rows = [] if labels is not None else None
        max_len = 0
        for batch_idx in range(batch_size):
            row_parts = []
            mask_parts = []
            label_parts = [] if labels is not None else None
            inserted_visual = False
            for token_idx in range(sequence_length):
                if int(input_ids[batch_idx, token_idx].item()) == int(image_token_index):
                    if not inserted_visual:
                        row_parts.append(image_features[batch_idx])
                        mask_parts.append(torch.ones(visual_tokens, dtype=attention_mask.dtype, device=device))
                        if labels is not None:
                            label_parts.append(
                                torch.full((visual_tokens,), ignore_index, dtype=labels.dtype, device=labels.device)
                            )
                        inserted_visual = True
                    continue
                row_parts.append(inputs_embeds[batch_idx, token_idx : token_idx + 1])
                mask_parts.append(attention_mask[batch_idx, token_idx : token_idx + 1])
                if labels is not None:
                    label_parts.append(labels[batch_idx, token_idx : token_idx + 1])
            row = torch.cat(row_parts, dim=0)
            mask = torch.cat(mask_parts, dim=0)
            rows.append(row)
            masks.append(mask)
            if labels is not None:
                label_rows.append(torch.cat(label_parts, dim=0))
            max_len = max(max_len, int(row.shape[0]))

        final_embedding = inputs_embeds.new_zeros((batch_size, max_len, embed_dim))
        final_attention_mask = attention_mask.new_zeros((batch_size, max_len))
        final_labels = (
            torch.full((batch_size, max_len), ignore_index, dtype=labels.dtype, device=labels.device)
            if labels is not None
            else None
        )
        for batch_idx, row in enumerate(rows):
            length = int(row.shape[0])
            final_embedding[batch_idx, :length] = row
            final_attention_mask[batch_idx, :length] = masks[batch_idx]
            if labels is not None:
                final_labels[batch_idx, :length] = label_rows[batch_idx]

        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        return final_embedding, final_attention_mask, final_labels, position_ids

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
        overhead_elapsed = max(0.0, float(patch_elapsed) - float(forward_elapsed))
        is_decode_step = bool(use_cache and seq_len <= 1)
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
            if layer_idx == 0 and seq_len > 1:
                self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
                if self.runtime_stats.get("prefill_seq_len") is None:
                    self.runtime_stats["prefill_seq_len"] = int(seq_len)
            elif layer_idx == 0 and is_decode_step:
                self.runtime_stats["generated_token_steps"] += 1

            forward_t0 = time.perf_counter()
            outputs = self.original_layer_forwards[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            forward_elapsed = time.perf_counter() - forward_t0
            self._record_layer_timing(
                seq_len=seq_len,
                use_cache=use_cache,
                patch_elapsed=time.perf_counter() - patch_t0,
                forward_elapsed=forward_elapsed,
                decode_fastpath=is_decode_step,
            )
            return outputs

        return new_forward


def maybe_apply_mantis_fastervlm(
    model,
    config: Optional[MantisFasterVLMConfig],
) -> Optional[MantisFasterVLMPatch]:
    if config is None or not config.enabled:
        return None
    patch = MantisFasterVLMPatch(model, config)
    patch.apply()
    if config.disable_kv_cache:
        model.config.use_cache = False
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            model.language_model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
    model._fastervlm_patch = patch
    model._autoprune_patch = patch
    return patch


__all__ = [
    "MantisFasterVLMPatch",
    "maybe_apply_mantis_fastervlm",
]
