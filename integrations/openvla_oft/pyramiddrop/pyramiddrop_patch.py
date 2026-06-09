import math
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class PyramidDropConfig:
    method: str = "pyramiddrop"
    visual_token_num: int = 336
    num_images_in_input: int = 2
    layer_list: Tuple[int, ...] = (8, 16, 24)


class PyramidDropPatch:
    """Paper-aligned PyramidDrop baseline for OpenVLA-OFT.

    All camera-view tokens form one unified visual block. Decoder layers are
    divided into stages, and visual tokens are pruned only after each stage
    boundary. The query is the final prompt/instruction token immediately
    before OpenVLA's action placeholders, never a generated action token.

    ``visual_token_num`` is interpreted as the requested average number of
    visual tokens processed per decoder layer. Lambda is solved against the
    actual stage lengths so the FLOPs comparison matches AutoPrune's layer-wise
    token-count proxy.
    """

    def __init__(self, model: torch.nn.Module, config: PyramidDropConfig):
        self.model = model
        self.config = config
        if str(config.method).lower() != "pyramiddrop":
            raise ValueError(f"Expected pyramiddrop, got {config.method!r}")

        self.layers = self._resolve_layers(model)
        self.num_layers = len(self.layers)
        self.boundaries = self._resolve_boundaries(config.layer_list)
        self.original_forwards: Dict[int, Any] = {}
        self.original_regression_or_discrete_prediction = None
        self.original_run_diffusion_prediction = None

        self.global_indices: Optional[torch.Tensor] = None
        self.visual_global_start = 1
        self.original_visual_tokens = 0
        self.query_global_position = 0
        self.lambda_value = 1.0
        self.keep_schedule: Dict[int, int] = {}
        self.final_visual_tokens = 0
        self.latest_stats: Dict[str, Any] = {}
        self._reported_real_pruning = False

    def apply(self):
        for idx, layer in enumerate(self.layers):
            self.original_forwards[idx] = layer.forward
            layer.forward = types.MethodType(self._make_new_forward(idx), layer)

        if hasattr(self.model, "_regression_or_discrete_prediction"):
            self.original_regression_or_discrete_prediction = self.model._regression_or_discrete_prediction
            self.model._regression_or_discrete_prediction = types.MethodType(
                self._patched_regression_or_discrete_prediction, self.model
            )
        if hasattr(self.model, "_run_diffusion_prediction"):
            self.original_run_diffusion_prediction = self.model._run_diffusion_prediction
            self.model._run_diffusion_prediction = types.MethodType(
                self._patched_run_diffusion_prediction, self.model
            )

        print("PyramidDrop patch applied.")
        print(f"   Stage-end pruning layers (1-based): {[x + 1 for x in self.boundaries]}")
        print(f"   Requested average visual tokens/layer: {int(self.config.visual_token_num)}")
        print("   Query: final instruction token before OpenVLA action placeholders")

    def get_flops_reduction(self) -> float:
        return float(self.latest_stats.get("flops_reduction", 0.0) or 0.0)

    def _prepare_prefill_layout(self, visual_tokens: int, prefix_tokens: int, prompt_tokens: int):
        self.original_visual_tokens = int(visual_tokens)
        # Multimodal layout is [BOS, visual block, prompt tail, action tokens].
        # The final instruction token is immediately before the first action
        # placeholder at 1 + visual_tokens + prompt_tokens.
        self.query_global_position = int(prefix_tokens) + int(prompt_tokens)
        self.lambda_value = self._solve_lambda(
            initial_tokens=int(visual_tokens),
            target_average=int(self.config.visual_token_num),
        )
        self.keep_schedule = self._build_keep_schedule(int(visual_tokens), self.lambda_value)
        self.final_visual_tokens = (
            min(self.keep_schedule.values()) if self.keep_schedule else int(visual_tokens)
        )

    def _dense_image_token_count(self, model_self, projected_prefix_tokens: int) -> int:
        expected = int(model_self.vision_backbone.get_num_patches()) * int(
            model_self.vision_backbone.get_num_images_in_input()
        )
        return max(1, min(expected, int(projected_prefix_tokens)))

    def _patched_regression_or_discrete_prediction(
        self,
        model_self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):
        visual_tokens = self._dense_image_token_count(model_self, projected_patch_embeddings.shape[1])
        prefix_extras = max(0, int(NUM_PATCHES) - visual_tokens)
        self._prepare_prefill_layout(visual_tokens, int(NUM_PATCHES), NUM_PROMPT_TOKENS)
        return self.original_regression_or_discrete_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            self.final_visual_tokens + prefix_extras,
            NUM_PROMPT_TOKENS,
            action_head,
        )

    def _patched_run_diffusion_prediction(
        self,
        model_self,
        input_embeddings,
        all_actions_mask,
        noise,
        action_head,
        projected_patch_embeddings,
        labels,
        attention_mask,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        noisy_action_projector,
    ):
        visual_tokens = self._dense_image_token_count(model_self, projected_patch_embeddings.shape[1])
        prefix_extras = max(0, int(NUM_PATCHES) - visual_tokens)
        self._prepare_prefill_layout(visual_tokens, int(NUM_PATCHES), NUM_PROMPT_TOKENS)
        # Proprio and the diffusion timestep token are prefix extras, not image
        # tokens. Keep both while replacing only the image-token count.
        return self.original_run_diffusion_prediction(
            input_embeddings,
            all_actions_mask,
            noise,
            action_head,
            projected_patch_embeddings,
            labels,
            attention_mask,
            self.final_visual_tokens + prefix_extras,
            NUM_PROMPT_TOKENS,
            noisy_action_projector,
        )

    def _make_new_forward(self, layer_idx: int):
        def new_forward(self_layer, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            batch_size, seq_len, _ = hidden_states.shape

            if layer_idx == 0:
                self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
                self.latest_stats = {
                    "original": seq_len,
                    "final": seq_len,
                    "original_visual_tokens": int(self.original_visual_tokens),
                    "final_visual_tokens": int(self.original_visual_tokens),
                    "raw_total_budget": float(self.config.visual_token_num),
                    "active_pruning_layers": [],
                    "keep_count_schedule": [],
                    "view_keep_counts": [int(self.original_visual_tokens)],
                    "layer_tokens": {},
                    "vis_tokens": {},
                    "pyramiddrop_lambda": float(self.lambda_value),
                    "pyramiddrop_score_latency_ms": 0.0,
                    "pyramiddrop_gather_latency_ms": 0.0,
                    "compare_pruning_overhead_latency_ms": 0.0,
                }

            if self.global_indices is None:
                self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            if self.global_indices.shape[0] == 1 and batch_size > 1:
                self.global_indices = self.global_indices.expand(batch_size, -1)

            self.latest_stats["layer_tokens"][layer_idx] = int(seq_len)
            self.latest_stats["final"] = int(seq_len)
            current_visual_positions = self._current_visual_positions(hidden_states.device)
            self.latest_stats["vis_tokens"][layer_idx] = int(current_visual_positions.numel())

            if position_ids is None:
                position_ids = self.global_indices
            elif position_ids.dim() == 2 and position_ids.shape[1] > seq_len:
                position_ids = position_ids.gather(1, self.global_indices)

            if attention_mask is not None and attention_mask.shape[-1] != seq_len:
                # The compacted sequence is still a standard causal sequence.
                # Rebuilding the causal mask avoids assigning stale dense-mask
                # rows to newly adjacent kept tokens.
                attention_mask = None

            if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
                try:
                    cos, sin = kwargs["position_embeddings"]
                    if cos.dim() == 3 and cos.shape[1] > seq_len:
                        gather = self.global_indices.unsqueeze(-1).expand(-1, -1, cos.size(-1))
                        kwargs["position_embeddings"] = (cos.gather(1, gather), sin.gather(1, gather))
                except Exception:
                    kwargs["position_embeddings"] = None

            original_output_attentions = bool(kwargs.get("output_attentions", False))
            original_use_cache = bool(kwargs.get("use_cache", False))
            need_pruning_attn = layer_idx in self.keep_schedule
            if need_pruning_attn and not original_output_attentions:
                kwargs["output_attentions"] = True

            outputs = self.original_forwards[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

            def clean_outputs(hidden_override=None):
                result = outputs
                if hidden_override is not None:
                    result = (hidden_override,) + tuple(outputs[1:])
                if original_output_attentions or not need_pruning_attn:
                    return result
                if original_use_cache and len(result) >= 3:
                    return (result[0], result[2])
                return (result[0],) + tuple(result[2:])

            if not need_pruning_attn or current_visual_positions.numel() <= 1:
                self._finalize_stats()
                return clean_outputs()

            target_keep = max(
                1, min(int(self.keep_schedule[layer_idx]), int(current_visual_positions.numel()))
            )
            attn_weights = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
            query_positions = self._current_query_positions(hidden_states.device)
            if target_keep >= current_visual_positions.numel():
                self._finalize_stats()
                return clean_outputs()
            if attn_weights is None:
                raise RuntimeError(
                    f"PyramidDrop could not obtain attention weights at stage-end layer {layer_idx + 1}; "
                    "refusing to continue with fake pruning"
                )
            if query_positions.numel() == 0:
                raise RuntimeError(
                    "PyramidDrop lost the final instruction-token query position; "
                    "refusing to use an action token or sequence tail as fallback"
                )

            score_t0 = time.perf_counter()
            # PyramidDrop paper: only the last instruction token is the query.
            rel_attn = attn_weights.index_select(2, query_positions).index_select(3, current_visual_positions)
            visual_scores = rel_attn.mean(dim=1).mean(dim=1)
            keep_rel = torch.topk(visual_scores, k=target_keep, dim=1, largest=True, sorted=False).indices
            keep_rel, _ = torch.sort(keep_rel, dim=1)
            visual_base = current_visual_positions.unsqueeze(0).expand(batch_size, -1)
            selected_visual_positions = visual_base.gather(1, keep_rel)
            self.latest_stats["pyramiddrop_score_latency_ms"] += (time.perf_counter() - score_t0) * 1000.0

            gather_t0 = time.perf_counter()
            visual_mask = torch.zeros(seq_len, dtype=torch.bool, device=hidden_states.device)
            visual_mask[current_visual_positions] = True
            nonvisual_positions = torch.nonzero(~visual_mask, as_tuple=False).flatten()
            nonvisual_positions = nonvisual_positions.unsqueeze(0).expand(batch_size, -1)
            full_indices = torch.cat([selected_visual_positions, nonvisual_positions], dim=1)
            full_indices, _ = torch.sort(full_indices, dim=1)
            pruned_hidden = outputs[0].gather(
                1, full_indices.unsqueeze(-1).expand(-1, -1, outputs[0].shape[-1])
            ).contiguous()
            self.global_indices = self.global_indices.gather(1, full_indices)
            self.latest_stats["pyramiddrop_gather_latency_ms"] += (time.perf_counter() - gather_t0) * 1000.0

            self.latest_stats["active_pruning_layers"].append(int(layer_idx))
            self.latest_stats["keep_count_schedule"].append([int(target_keep)])
            self.latest_stats["final_visual_tokens"] = int(target_keep)
            self.latest_stats["view_keep_counts"] = [int(target_keep)]
            if not self._reported_real_pruning:
                print(
                    "PyramidDrop real pruning verified: "
                    f"visual tokens {self.original_visual_tokens} -> {target_keep} "
                    f"after decoder layer {layer_idx + 1}; lambda={self.lambda_value:.6f}; "
                    f"query_global_position={self.query_global_position}"
                )
                self._reported_real_pruning = True

            self._finalize_stats()
            return clean_outputs(pruned_hidden)

        return new_forward

    def _current_visual_positions(self, device):
        if self.global_indices is None or self.original_visual_tokens <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        globals_row = self.global_indices[0]
        mask = (globals_row >= self.visual_global_start) & (
            globals_row < self.visual_global_start + self.original_visual_tokens
        )
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _current_query_positions(self, device):
        if self.global_indices is None:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.nonzero(
            self.global_indices[0] == int(self.query_global_position), as_tuple=False
        ).flatten()[-1:]

    def _finalize_stats(self):
        self.latest_stats["compare_pruning_overhead_latency_ms"] = float(
            self.latest_stats.get("pyramiddrop_score_latency_ms", 0.0)
            + self.latest_stats.get("pyramiddrop_gather_latency_ms", 0.0)
        )
        layer_tokens = self.latest_stats.get("layer_tokens", {})
        original = int(self.latest_stats.get("original", 0) or 0)
        if original > 0 and len(layer_tokens) == self.num_layers:
            dense_total = original * self.num_layers
            actual_total = sum(int(x) for x in layer_tokens.values())
            self.latest_stats["original_decoder_token_sum"] = int(dense_total)
            self.latest_stats["actual_decoder_token_sum"] = int(actual_total)
            self.latest_stats["saved_decoder_token_sum"] = int(dense_total - actual_total)
            self.latest_stats["decoder_layers_recorded"] = int(self.num_layers)
            self.latest_stats["decoder_layers_missing"] = 0
            self.latest_stats["flops_reduction"] = max(0.0, 1.0 - float(actual_total) / float(dense_total))

    def _build_keep_schedule(self, initial_tokens: int, lambda_value: float) -> Dict[int, int]:
        schedule: Dict[int, int] = {}
        current = int(initial_tokens)
        for stage_idx, boundary in enumerate(self.boundaries):
            target = max(1, min(current, int(round(initial_tokens * (lambda_value ** (stage_idx + 1))))))
            schedule[int(boundary)] = int(target)
            current = target
        return schedule

    def _solve_lambda(self, initial_tokens: int, target_average: int) -> float:
        initial_tokens = max(1, int(initial_tokens))
        target_average = max(1, min(int(target_average), initial_tokens))
        if target_average >= initial_tokens or not self.boundaries:
            return 1.0

        def average_for(value: float) -> float:
            schedule = self._build_keep_schedule(initial_tokens, value)
            current = initial_tokens
            total = 0
            for layer_idx in range(self.num_layers):
                total += current
                if layer_idx in schedule:
                    current = schedule[layer_idx]
            return float(total) / float(self.num_layers)

        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) * 0.5
            if average_for(mid) < target_average:
                lo = mid
            else:
                hi = mid
        return float(hi)

    def _resolve_boundaries(self, layer_list: Tuple[int, ...]):
        # Paper stages prune at the end of every stage except the final stage.
        return sorted(
            {
                max(0, min(self.num_layers - 2, int(layer_number) - 1))
                for layer_number in layer_list
                if int(layer_number) > 0 and self.num_layers > 1
            }
        )

    @staticmethod
    def _resolve_layers(model):
        candidates = [
            ("language_model", "model", "layers"),
            ("language_model", "layers"),
            ("model", "layers"),
            ("base_model", "model", "model", "layers"),
        ]
        for path in candidates:
            obj = model
            for attr in path:
                if not hasattr(obj, attr):
                    obj = None
                    break
                obj = getattr(obj, attr)
            if isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
                return obj
        raise RuntimeError("Could not locate OpenVLA decoder layers for PyramidDrop")
