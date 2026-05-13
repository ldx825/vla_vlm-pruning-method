"""
VLA-Pruner patch for OpenVLA-OFT LIBERO evaluation.

This implementation follows the paper pipeline, but adapts action-to-vision
attention collection to OpenVLA-OFT's action-chunk decoder.  All visual tokens
from all input views are treated as one unified VLM-style visual pool.
"""

from __future__ import annotations

import time
import types
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@dataclass
class VLAPrunerOFTConfig:
    enabled: bool = False
    keep_visual_tokens: int = 224
    visual_tokens: int = 512
    pruning_layer: int = 3
    temporal_window: int = 3
    temporal_gamma: float = 0.8
    warmup_steps: int = 3
    action_attention_layer_start: str = "half"
    debug: bool = False


def _find_decoder_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
        if hasattr(lm, "layers"):
            return list(lm.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []


def _estimate_flops_reduction(original_tokens: int, actual_by_layer: Dict[int, int], num_layers: int) -> float:
    if original_tokens <= 0 or num_layers <= 0:
        return 0.0
    dense = float(original_tokens * num_layers)
    actual = float(sum(int(actual_by_layer.get(i, original_tokens)) for i in range(num_layers)))
    return max(0.0, min(1.0, 1.0 - actual / max(dense, 1.0)))


def _max_min_redundancy_filter(features: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Paper Step 7: greedy min-redundancy filtering over the candidate pool."""
    num_tokens = int(features.shape[0])
    keep_count = max(1, min(int(keep_count), num_tokens))
    if num_tokens <= keep_count:
        return torch.arange(num_tokens, device=features.device, dtype=torch.long)

    x = features.float()
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    distance = 1.0 - torch.mm(x, x.t())

    selected = torch.empty(keep_count, device=features.device, dtype=torch.long)
    for i in range(keep_count):
        if i == 0:
            # First point: choose a token far from its nearest neighbor.
            scores = torch.topk(distance, k=2, dim=0, largest=False).values[1, :]
        else:
            prev = selected[:i]
            scores = distance.index_select(0, prev).min(dim=0).values
            scores[prev] = -1.0
        selected[i] = torch.argmax(scores)
    return selected


class VLAPrunerOFTPatch:
    def __init__(self, model, config: VLAPrunerOFTConfig):
        self.model = model
        self.config = config
        self.layers = _find_decoder_layers(model)
        self.num_layers = len(self.layers)
        self.original_forwards: Dict[int, Any] = {}
        self.original_regression_or_discrete_prediction = None
        self.original_run_diffusion_prediction = None

        self.pruning_layer_idx = max(0, int(config.pruning_layer) - 1)
        self.action_layer_start_idx = self._resolve_action_layer_start(config.action_attention_layer_start)
        self.action_attention_history: Deque[torch.Tensor] = deque(maxlen=max(1, int(config.temporal_window)))

        self.global_indices: Optional[torch.Tensor] = None
        self.visual_positions: Optional[torch.Tensor] = None
        self.current_projected_tokens: int = int(config.visual_tokens)
        self.current_action_len: int = ACTION_DIM * NUM_ACTIONS_CHUNK
        self._has_pruned_this_forward = False
        self._pending_multimodal_build_latency_ms = 0.0
        self._current_action_vectors: List[torch.Tensor] = []
        self._current_flow_vectors: List[torch.Tensor] = []
        self._avg_budget_token_sum = 0.0
        self._avg_budget_step_count = 0
        self._avg_budget_horizon_steps: Optional[int] = None
        self.latest_stats: Dict[str, Any] = {}

    def _resolve_action_layer_start(self, value: str) -> int:
        if str(value).lower() == "half":
            return max(0, self.num_layers // 2)
        try:
            return max(0, min(self.num_layers - 1, int(value)))
        except Exception:
            return max(0, self.num_layers // 2)

    def apply(self):
        if not self.layers:
            raise RuntimeError("VLA-Pruner-OFT could not find decoder layers to patch.")

        for idx, layer in enumerate(self.layers):
            self.original_forwards[idx] = layer.forward
            layer.forward = types.MethodType(self._make_layer_forward(idx), layer)

        if hasattr(self.model, "_regression_or_discrete_prediction"):
            self.original_regression_or_discrete_prediction = self.model._regression_or_discrete_prediction
            self.model._regression_or_discrete_prediction = types.MethodType(
                self._patched_regression_or_discrete_prediction,
                self.model,
            )

        if hasattr(self.model, "_run_diffusion_prediction"):
            self.original_run_diffusion_prediction = self.model._run_diffusion_prediction
            self.model._run_diffusion_prediction = types.MethodType(self._patched_run_diffusion_prediction, self.model)

        setattr(self.model, "_vlapruner_patch_ref", self)
        print(
            "✅ VLA-Pruner-OFT applied: "
            f"K={int(self.config.pruning_layer)}, keep_visual_tokens={int(self.config.keep_visual_tokens)}, "
            f"visual_tokens={int(self.config.visual_tokens)}, temporal_window={int(self.config.temporal_window)}, "
            f"gamma={float(self.config.temporal_gamma)}"
        )

    def reset_episode_history(self, average_budget_horizon_steps: Optional[int] = None):
        self.action_attention_history.clear()
        self._avg_budget_token_sum = 0.0
        self._avg_budget_step_count = 0
        if average_budget_horizon_steps is not None:
            self._avg_budget_horizon_steps = max(1, int(average_budget_horizon_steps))

    def reset_runtime_state(self, seq_len: int, device: torch.device):
        visual_tokens = max(0, min(int(self.config.visual_tokens), max(0, int(seq_len) - 1)))
        self.global_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        self.visual_positions = torch.arange(1, 1 + visual_tokens, device=device, dtype=torch.long)
        self._has_pruned_this_forward = False
        self._current_action_vectors = []
        self.latest_stats = {
            "method": "vlapruner_oft",
            "original": int(seq_len),
            "final": int(seq_len),
            "layer_tokens": {},
            "vis_tokens": {},
            "layer_times": {},
            "active_pruning_layers": [int(self.pruning_layer_idx)],
            "keep_count_schedule": [],
            "raw_total_budget": float(self.config.keep_visual_tokens),
            "target_average_visual_tokens": int(self.config.keep_visual_tokens),
            "final_visual_tokens": int(visual_tokens),
            "view_keep_counts": [int(visual_tokens)],
            "view_score_masses": [],
            "unified_visual_pool": True,
            "visual_pool_start": 1,
            "visual_pool_length": int(visual_tokens),
            "requested_keep_visual_tokens": int(self.config.keep_visual_tokens),
            "effective_layer_keep_visual_tokens": int(visual_tokens),
            "average_budget_horizon_steps": int(self._avg_budget_horizon_steps or 0),
            "average_budget_step_count": int(self._avg_budget_step_count),
            "average_budget_running_visual_tokens": 0.0,
            "vlapruner_pruning_layer_k": int(self.config.pruning_layer),
            "vlapruner_history_ready": int(self._history_ready()),
            "vlapruner_temporal_window": int(self.config.temporal_window),
            "vlapruner_temporal_gamma": float(self.config.temporal_gamma),
            "decoder_layers_recorded": 0,
            "decoder_layers_missing": 0,
            "multimodal_build_latency_ms": float(self._pending_multimodal_build_latency_ms),
            "prefill_latency_ms": 0.0,
            "decode_latency_ms": 0.0,
            "decoder_forward_latency_ms": 0.0,
            "pruned_decoder_compute_latency_ms": 0.0,
            "pruning_layer_latency_ms": 0.0,
            "fastv_score_latency_ms": 0.0,
            "fastv_select_latency_ms": 0.0,
            "fastv_gather_latency_ms": 0.0,
            "fastv_overhead_latency_ms": 0.0,
            "compare_pruning_overhead_latency_ms": 0.0,
            "original_decoder_token_sum": 0,
            "actual_decoder_token_sum": 0,
            "saved_decoder_token_sum": 0,
            "flops_reduction": 0.0,
        }

    def get_flops_reduction(self) -> float:
        # Keep the reported FLOPs proxy aligned with AutoPrune:
        # dense token work = original sequence length * decoder layers;
        # actual token work = sum of the sequence length observed at each layer.
        layer_tokens = self.latest_stats.get("layer_tokens", {}) if self.latest_stats else {}
        if not layer_tokens:
            return 0.0
        original = int(self.latest_stats.get("original", 0) or 0)
        total_original_tokens = original * max(1, int(self.num_layers))
        total_actual_tokens = sum(int(v) for v in layer_tokens.values())
        if total_original_tokens <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (total_actual_tokens / total_original_tokens)))

    def _history_ready(self) -> bool:
        needed = max(1, int(self.config.warmup_steps))
        return len(self.action_attention_history) >= needed

    def _target_average_visual_tokens(self, total_visual: int) -> int:
        return max(1, min(int(self.config.keep_visual_tokens), int(total_visual)))

    def _average_corrected_keep_count(self, total_visual: int) -> int:
        """Convert requested average visual tokens into a stable post-warmup budget.

        The public CLI argument is interpreted as the desired average number of
        final visual tokens. Warmup forwards keep the full visual pool, so we
        use the expected episode query horizon to compute one fixed lower keep
        count for every post-warmup pruning step. This avoids an unstable early
        sequence of extremely small keep counts followed by the raw target.
        """
        total_visual = int(total_visual)
        target = self._target_average_visual_tokens(total_visual)
        horizon = int(self._avg_budget_horizon_steps or 0)
        if horizon <= 0:
            return target

        warmup_steps = min(max(0, int(self.config.warmup_steps)), horizon)
        pruning_steps = max(1, horizon - warmup_steps)
        desired_pruned_token_sum = float(target * horizon - total_visual * warmup_steps)
        raw_keep = int(round(desired_pruned_token_sum / float(pruning_steps)))
        return max(1, min(raw_keep, total_visual))

    def _record_average_budget_observation(self, final_visual_tokens: int, total_visual: int, effective_keep: int):
        final_visual_tokens = int(final_visual_tokens)
        self._avg_budget_token_sum += float(final_visual_tokens)
        self._avg_budget_step_count += 1
        running_avg = self._avg_budget_token_sum / max(1, self._avg_budget_step_count)
        target = self._target_average_visual_tokens(int(total_visual))
        self.latest_stats["target_average_visual_tokens"] = int(target)
        self.latest_stats["effective_layer_keep_visual_tokens"] = int(effective_keep)
        self.latest_stats["average_budget_horizon_steps"] = int(self._avg_budget_horizon_steps or 0)
        self.latest_stats["average_budget_step_count"] = int(self._avg_budget_step_count)
        self.latest_stats["average_budget_token_sum"] = float(self._avg_budget_token_sum)
        self.latest_stats["average_budget_running_visual_tokens"] = float(running_avg)
        self.latest_stats["average_budget_error"] = float(running_avg - target)

    def _temporal_action_scores(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self._history_ready():
            return None
        gamma = float(self.config.temporal_gamma)
        hist = list(self.action_attention_history)
        weights = torch.tensor(
            [gamma ** i for i in range(len(hist) - 1, -1, -1)],
            device=device,
            dtype=torch.float32,
        )
        stacked = torch.stack([x.to(device=device, dtype=torch.float32) for x in hist], dim=0)
        scores = (stacked * weights[:, None]).sum(dim=0) / weights.sum().clamp(min=1e-6)
        return scores.to(dtype=dtype)

    def _align_position_inputs(self, hidden_states, attention_mask, position_ids, kwargs):
        bsz, seq_len, _ = hidden_states.shape
        if self.global_indices is None or self.global_indices.shape[-1] != seq_len:
            self.global_indices = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)

        gather_idx = self.global_indices
        if gather_idx.shape[0] == 1 and bsz > 1:
            gather_idx = gather_idx.expand(bsz, -1)

        if position_ids is None:
            position_ids = gather_idx
        elif position_ids.shape[-1] != seq_len and position_ids.shape[-1] >= int(gather_idx.max().item()) + 1:
            position_ids = position_ids.gather(1, gather_idx)

        if attention_mask is not None and attention_mask.shape[-1] != seq_len:
            attention_mask = None

        if kwargs.get("position_embeddings") is not None:
            try:
                cos, sin = kwargs["position_embeddings"]
                if hasattr(cos, "shape") and cos.shape[1] != seq_len:
                    cos = cos.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, cos.size(-1)))
                    sin = sin.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, sin.size(-1)))
                    kwargs["position_embeddings"] = (cos, sin)
            except Exception:
                kwargs["position_embeddings"] = None

        return attention_mask, position_ids, kwargs

    def _semantic_visual_scores(self, attn_weights: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Paper Step 3: current semantic-level visual token importance."""
        visual_count = int(self.visual_positions.numel()) if self.visual_positions is not None else 0
        if visual_count <= 0:
            return torch.empty((attn_weights.shape[0], 0), device=attn_weights.device, dtype=attn_weights.dtype)

        prompt_start = 1 + max(int(self.current_projected_tokens), int(self.config.visual_tokens))
        action_start = max(prompt_start, seq_len - int(self.current_action_len) - 1)
        text_positions = torch.arange(prompt_start, action_start, device=attn_weights.device, dtype=torch.long)
        if text_positions.numel() == 0:
            # Fallback: exclude BOS, visual keys, action chunk, and STOP.
            mask = torch.ones(seq_len, device=attn_weights.device, dtype=torch.bool)
            mask[0] = False
            mask[1 : 1 + visual_count] = False
            mask[max(0, seq_len - int(self.current_action_len) - 1) :] = False
            text_positions = torch.where(mask)[0]
        if text_positions.numel() == 0:
            text_positions = torch.tensor([seq_len - 1], device=attn_weights.device, dtype=torch.long)

        visual_positions = self.visual_positions[:visual_count]
        rel_attn = attn_weights[:, :, text_positions, :][:, :, :, visual_positions]
        return rel_attn.mean(dim=1).mean(dim=1)

    def _record_action_to_vision_attention(self, layer_idx: int, attn_weights: Optional[torch.Tensor], seq_len: int):
        """Paper Step 4 support: record action-chunk-to-vision attention for future timesteps."""
        if attn_weights is None or layer_idx < self.action_layer_start_idx:
            return
        if self.global_indices is None or self.visual_positions is None or self.visual_positions.numel() == 0:
            return

        action_len = int(self.current_action_len)
        action_start = seq_len - action_len - 1
        action_end = seq_len - 1
        if action_len <= 0 or action_start < 0 or action_start >= action_end:
            return

        visual_positions = self.visual_positions[self.visual_positions < seq_len]
        if visual_positions.numel() == 0:
            return

        action_positions = torch.arange(action_start, action_end, device=attn_weights.device, dtype=torch.long)
        rel_attn = attn_weights[:, :, action_positions, :][:, :, :, visual_positions]
        current_scores = rel_attn.float().mean(dim=(0, 1, 2))

        gather_globals = self.global_indices[0, visual_positions]
        visual_ids = gather_globals - 1
        valid = (visual_ids >= 0) & (visual_ids < int(self.config.visual_tokens))
        if not torch.any(valid):
            return

        full_scores = torch.zeros(int(self.config.visual_tokens), device=attn_weights.device, dtype=torch.float32)
        full_scores.scatter_add_(0, visual_ids[valid], current_scores[valid])
        self._current_action_vectors.append(full_scores.detach())

    def _update_action_history_from_current_forward(self):
        """Paper Step 4: average action-to-vision attention over latter-half layers."""
        if not self._current_action_vectors:
            return
        vec = torch.stack(self._current_action_vectors, dim=0).mean(dim=0)
        self.action_attention_history.append(vec.detach().cpu())
        self.latest_stats["vlapruner_history_size"] = int(len(self.action_attention_history))
        self._current_action_vectors = []

    def _select_vlapruner_indices(
        self,
        hidden_states: torch.Tensor,
        semantic_scores: torch.Tensor,
        temporal_scores: torch.Tensor,
        keep_count: int,
    ) -> torch.Tensor:
        """Paper Steps 5-7: dual top-k, union, then greedy redundancy filtering."""
        bsz = hidden_states.shape[0]
        if bsz != 1:
            raise RuntimeError("VLA-Pruner-OFT currently expects batch size 1 during LIBERO evaluation.")

        total_visual = int(semantic_scores.shape[-1])
        keep_count = max(1, min(int(keep_count), total_visual))

        # Step 5: select semantically important tokens and action-important tokens.
        sem_topk = torch.topk(semantic_scores[0], k=keep_count, dim=-1).indices
        act_topk = torch.topk(temporal_scores[:total_visual], k=keep_count, dim=-1).indices

        # Step 6: maximize relevance by combining both candidate sets.
        candidate_local = torch.unique(torch.cat([sem_topk, act_topk], dim=0))

        # Step 7: remove redundancy by greedy Max-Min filtering over candidate features.
        if candidate_local.numel() > keep_count:
            visual_hidden = hidden_states[0, self.visual_positions[:total_visual], :]
            candidate_features = visual_hidden.index_select(0, candidate_local)
            chosen_in_candidate = _max_min_redundancy_filter(candidate_features, keep_count)
            candidate_local = candidate_local.index_select(0, chosen_in_candidate)

        candidate_local, _ = candidate_local.sort()
        return candidate_local

    def _make_layer_forward(self, layer_idx: int):
        def new_forward(self_layer, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            layer_t0 = time.perf_counter()
            bsz, seq_len, _ = hidden_states.shape

            # Paper Step 2: enter early Transformer layers without pruning.
            # Runtime state starts from the full multimodal sequence at layer 0.
            if layer_idx == 0:
                self.reset_runtime_state(seq_len, hidden_states.device)
            if not self.latest_stats:
                self.reset_runtime_state(seq_len, hidden_states.device)

            self.latest_stats["layer_tokens"][int(layer_idx)] = int(seq_len)
            self.latest_stats["final"] = int(seq_len)
            self.latest_stats["decoder_layers_recorded"] = len(self.latest_stats["layer_tokens"])

            attention_mask, position_ids, kwargs = self._align_position_inputs(
                hidden_states, attention_mask, position_ids, kwargs
            )

            original_output_attentions = bool(kwargs.get("output_attentions", False))
            need_scores = layer_idx == self.pruning_layer_idx or layer_idx >= self.action_layer_start_idx
            kwargs["output_attentions"] = bool(original_output_attentions or need_scores)
            original_use_cache = bool(kwargs.get("use_cache", False))

            fwd_t0 = time.perf_counter()
            outputs = self.original_forwards[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            fwd_ms = (time.perf_counter() - fwd_t0) * 1000.0
            self.latest_stats["decoder_forward_latency_ms"] += fwd_ms
            if self._has_pruned_this_forward:
                self.latest_stats["pruned_decoder_compute_latency_ms"] += fwd_ms

            def clean_outputs(hidden_override=None):
                hidden = outputs[0] if hidden_override is None else hidden_override
                if original_output_attentions:
                    return (hidden,) + outputs[1:]
                if not need_scores:
                    return outputs if hidden_override is None else (hidden,) + outputs[1:]
                if original_use_cache and len(outputs) >= 3:
                    return (hidden, outputs[2])
                return (hidden,)

            attn_weights = outputs[1] if len(outputs) > 1 else None
            current_hidden = outputs[0]
            current_seq_len = int(current_hidden.shape[1])

            self._record_action_to_vision_attention(layer_idx, attn_weights, current_seq_len)

            if (
                layer_idx != self.pruning_layer_idx
                or self._has_pruned_this_forward
                or attn_weights is None
                or self.visual_positions is None
                or self.visual_positions.numel() == 0
            ):
                self._finish_layer_stats(layer_idx, layer_t0)
                return clean_outputs()

            total_visual = int(min(self.visual_positions.numel(), int(self.config.visual_tokens)))
            keep_count = self._average_corrected_keep_count(total_visual)
            temporal_scores = self._temporal_action_scores(current_hidden.device, attn_weights.dtype)

            # Paper warm start: do not prune until enough past action attention exists.
            if temporal_scores is None or keep_count >= total_visual:
                self.latest_stats["final_visual_tokens"] = int(total_visual)
                self.latest_stats["view_keep_counts"] = [int(total_visual)]
                self.latest_stats["vis_tokens"][int(layer_idx)] = int(total_visual)
                self._record_average_budget_observation(
                    final_visual_tokens=total_visual,
                    total_visual=total_visual,
                    effective_keep=total_visual,
                )
                self._finish_layer_stats(layer_idx, layer_t0)
                return clean_outputs()

            score_t0 = time.perf_counter()
            semantic_scores = self._semantic_visual_scores(attn_weights, current_seq_len)
            score_ms = (time.perf_counter() - score_t0) * 1000.0

            select_t0 = time.perf_counter()
            keep_local = self._select_vlapruner_indices(
                current_hidden,
                semantic_scores,
                temporal_scores,
                keep_count=keep_count,
            )
            selected_visual = self.visual_positions[:total_visual].unsqueeze(0).gather(1, keep_local.unsqueeze(0))
            select_ms = (time.perf_counter() - select_t0) * 1000.0

            # Paper Step 8: drop unselected visual tokens physically; later layers
            # only see [BOS] + kept unified visual tokens + the unchanged suffix.
            gather_t0 = time.perf_counter()
            non_visual = torch.arange(
                1 + total_visual,
                current_seq_len,
                device=current_hidden.device,
                dtype=torch.long,
            ).unsqueeze(0)
            bos = torch.zeros((1, 1), device=current_hidden.device, dtype=torch.long)
            full_indices = torch.cat([bos, selected_visual, non_visual], dim=1)
            full_indices, _ = full_indices.sort(dim=1)
            if bsz > 1:
                full_indices = full_indices.expand(bsz, -1)

            pruned_hidden = current_hidden.gather(
                1,
                full_indices.unsqueeze(-1).expand(-1, -1, current_hidden.size(-1)),
            )

            if self.global_indices is not None:
                current_globals = self.global_indices.expand(bsz, -1) if self.global_indices.shape[0] == 1 and bsz > 1 else self.global_indices
                self.global_indices = current_globals.gather(1, full_indices)

            self.visual_positions = torch.arange(1, 1 + keep_count, device=current_hidden.device, dtype=torch.long)
            self._has_pruned_this_forward = True
            gather_ms = (time.perf_counter() - gather_t0) * 1000.0

            overhead_ms = score_ms + select_ms + gather_ms
            self.latest_stats["fastv_score_latency_ms"] += score_ms
            self.latest_stats["fastv_select_latency_ms"] += select_ms
            self.latest_stats["fastv_gather_latency_ms"] += gather_ms
            self.latest_stats["fastv_overhead_latency_ms"] += overhead_ms
            self.latest_stats["compare_pruning_overhead_latency_ms"] += overhead_ms
            self.latest_stats["final_visual_tokens"] = int(keep_count)
            self.latest_stats["view_keep_counts"] = [int(keep_count)]
            self.latest_stats["keep_count_schedule"] = [[int(keep_count)]]
            self.latest_stats["vis_tokens"][int(layer_idx)] = int(keep_count)
            self._record_average_budget_observation(
                final_visual_tokens=keep_count,
                total_visual=total_visual,
                effective_keep=keep_count,
            )
            self.latest_stats["view_score_masses"] = [
                float(semantic_scores.sum().detach().float().cpu().item()),
                float(temporal_scores.sum().detach().float().cpu().item()),
            ]

            self._finish_layer_stats(layer_idx, layer_t0)
            return clean_outputs(hidden_override=pruned_hidden)

        return new_forward

    def _finish_layer_stats(self, layer_idx: int, layer_t0: float):
        layer_ms = (time.perf_counter() - layer_t0) * 1000.0
        self.latest_stats.setdefault("layer_times", {})[int(layer_idx)] = layer_ms / 1000.0
        if layer_idx == self.pruning_layer_idx:
            self.latest_stats["pruning_layer_latency_ms"] += layer_ms
        original = int(self.latest_stats.get("original", 0) or 0)
        actual_by_layer = {int(k): int(v) for k, v in self.latest_stats.get("layer_tokens", {}).items()}
        self.latest_stats["original_decoder_token_sum"] = int(original * max(1, self.num_layers))
        self.latest_stats["actual_decoder_token_sum"] = int(
            sum(actual_by_layer.get(i, original) for i in range(self.num_layers))
        )
        self.latest_stats["saved_decoder_token_sum"] = max(
            0,
            int(self.latest_stats["original_decoder_token_sum"] - self.latest_stats["actual_decoder_token_sum"]),
        )
        self.latest_stats["decoder_layers_recorded"] = int(len(actual_by_layer))
        self.latest_stats["decoder_layers_missing"] = max(0, int(self.num_layers - len(actual_by_layer)))
        self.latest_stats["flops_reduction"] = self.get_flops_reduction()

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
        # Paper Step 1: receive visual observations and language instruction.
        # OpenVLA-OFT has already projected images into `projected_patch_embeddings`
        # and prompt/action placeholders into `input_embeddings` at this call site.
        self.current_projected_tokens = int(projected_patch_embeddings.shape[1])
        self.current_action_len = int(ACTION_DIM * NUM_ACTIONS_CHUNK)
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask

        build_t0 = time.perf_counter()
        multimodal_embeddings, multimodal_attention_mask = model_self._build_multimodal_attention(
            input_embeddings,
            projected_patch_embeddings,
            attention_mask,
        )
        self._pending_multimodal_build_latency_ms = (time.perf_counter() - build_t0) * 1000.0

        prefill_t0 = time.perf_counter()
        language_model_output = model_self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        self.latest_stats["prefill_latency_ms"] = (time.perf_counter() - prefill_t0) * 1000.0
        self.latest_stats["multimodal_build_latency_ms"] = float(self._pending_multimodal_build_latency_ms)

        self._update_action_history_from_current_forward()

        last_hidden_states = language_model_output.hidden_states[-1]
        action_len = int(ACTION_DIM * NUM_ACTIONS_CHUNK)
        actions_hidden_states = last_hidden_states[:, -(action_len + 1) : -1, :]

        if action_head is not None:
            normalized_actions = action_head.predict_action(actions_hidden_states)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            predicted_action_token_ids = language_model_output.logits[:, -(action_len + 1) : -1, :].argmax(dim=2)
            predicted_action_token_ids = predicted_action_token_ids.cpu().numpy()
            discretized_actions = model_self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model_self.bin_centers.shape[0] - 1)
            normalized_actions = model_self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states

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
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        curr_noisy_actions = noise
        action_len = int(ACTION_DIM * NUM_ACTIONS_CHUNK)
        flow_step_vectors: List[torch.Tensor] = []
        actions_hidden_states = None

        for t in action_head.noise_scheduler.timesteps:
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            ).unsqueeze(1)
            projected_patch_embeddings = torch.cat((orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1)

            bsz = curr_noisy_actions.shape[0]
            orig_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(bsz, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_shape)

            replaced_embeddings = model_self._replace_input_embeddings(
                input_embeddings.clone(),
                all_actions_mask,
                noisy_action_features,
            )

            self.current_projected_tokens = int(projected_patch_embeddings.shape[1])
            self.current_action_len = action_len
            multimodal_embeddings, multimodal_attention_mask = model_self._build_multimodal_attention(
                replaced_embeddings,
                projected_patch_embeddings,
                attention_mask,
            )

            self._current_action_vectors = []
            language_model_output = model_self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if self._current_action_vectors:
                flow_step_vectors.append(torch.stack(self._current_action_vectors, dim=0).mean(dim=0).detach())
            self._current_action_vectors = []

            last_hidden_states = language_model_output.hidden_states[-1]
            actions_hidden_states = last_hidden_states[:, -(action_len + 1) : -1, :]
            noise_pred = action_head.predict_noise(actions_hidden_states)
            curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        # Diffusion / flow policy adaptation: average action attention over the latter half of flow steps.
        if flow_step_vectors:
            half = max(0, len(flow_step_vectors) // 2)
            vec = torch.stack(flow_step_vectors[half:], dim=0).mean(dim=0)
            self.action_attention_history.append(vec.detach().cpu())
            self.latest_stats["vlapruner_history_size"] = int(len(self.action_attention_history))

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states


def maybe_apply_vlapruner_oft_patch(model, config: VLAPrunerOFTConfig) -> Optional[VLAPrunerOFTPatch]:
    if not config.enabled:
        return None
    patch = VLAPrunerOFTPatch(model=model, config=config)
    patch.apply()
    return patch
