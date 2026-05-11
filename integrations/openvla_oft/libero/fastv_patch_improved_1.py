"""
FastV patch for OpenVLA-OFT LIBERO evaluation.

This is the corrected, compact FastV implementation for Chen et al. 2024,
"An Image is Worth 1/2 Tokens After Layer 2".  The integration intentionally
mirrors the working AutoPrune layer-wrapper style in
`autoprune_patch_improved_1.py`, but FastV itself is much simpler:

  Step 1. Run the first K decoder layers densely.
  Step 2. At layer K, compute each visual token's received attention score.
  Step 3. Keep the top `keep_visual_tokens` visual tokens from one unified pool.
  Step 4. Continue all later decoder layers on the physically shortened sequence.

OpenVLA-OFT does not use autoregressive KV-cache generation in this LIBERO path,
so no KV-cache pruning is needed.  The paper's pruning ratio R is exposed here
as a direct keep-count parameter because that is easier to sweep in VLA tables:
`R = 1 - keep_visual_tokens / visual_tokens`.
"""

from __future__ import annotations

import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@dataclass
class FastVConfig:
    enabled: bool = False
    keep_visual_tokens: int = 128
    visual_tokens: int = 512
    pruning_layer: int = 2
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


class FastVPatch:
    """Plug-and-play FastV for OpenVLA-OFT.

    The patch mirrors AutoPrune's layer-level gather style so latency and FLOPs
    estimates are recorded through the same `latest_stats` surface used by the
    existing LIBERO evaluator.
    """

    def __init__(self, model, config: FastVConfig):
        self.model = model
        self.config = config
        self.layers = _find_decoder_layers(model)
        self.num_layers = len(self.layers)
        self.original_forwards: Dict[int, Any] = {}
        self.original_regression_or_discrete_prediction = None
        self.original_run_diffusion_prediction = None

        self.global_indices: Optional[torch.Tensor] = None
        self.visual_positions: Optional[torch.Tensor] = None
        self.active_visual_tokens: int = int(config.visual_tokens)
        self.latest_stats: Dict[str, Any] = {}
        self._pending_multimodal_build_latency_ms: float = 0.0

        # The paper denotes the layer as K.  CLI values are 1-based, while layer
        # wrappers are 0-based; pruning happens immediately after this layer runs.
        self.pruning_layer_idx = max(0, int(config.pruning_layer) - 1)

    def apply(self):
        if not self.layers:
            raise RuntimeError("FastV could not find OpenVLA-OFT decoder layers to patch.")

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

        setattr(self.model, "_fastv_patch_ref", self)
        print(
            "✅ FastV applied: "
            f"K={int(self.config.pruning_layer)}, keep_visual_tokens={int(self.config.keep_visual_tokens)}, "
            f"visual_tokens={int(self.config.visual_tokens)}"
        )

    def reset_runtime_state(self, seq_len: int, device: torch.device):
        visual_tokens = max(0, min(int(self.active_visual_tokens), max(0, int(seq_len) - 1)))
        self.global_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

        # FastV Step 0 / OpenVLA layout:
        #   [BOS] + [all visual tokens from every camera view] + [prompt/action/stop tokens]
        #
        # Evidence for "unified multi-view pruning": the whole visual span below
        # is represented by one contiguous tensor.  This file never constructs
        # per-camera slices, per-view scores, or per-view budgets.
        self.visual_positions = torch.arange(1, 1 + visual_tokens, device=device, dtype=torch.long)

        keep_visual = max(1, min(int(self.config.keep_visual_tokens), max(1, visual_tokens)))
        reduction_ratio = 0.0 if visual_tokens <= 0 else 1.0 - float(keep_visual) / float(visual_tokens)
        self.latest_stats = {
            "method": "fastv",
            "original": int(seq_len),
            "final": int(seq_len),
            "layer_tokens": {},
            "vis_tokens": {},
            "active_pruning_layers": [int(self.pruning_layer_idx)],
            "keep_count_schedule": [[int(keep_visual)]],
            "raw_total_budget": float(keep_visual),
            "final_visual_tokens": int(visual_tokens),
            "view_keep_counts": [int(keep_visual)],
            "view_score_masses": [],
            "unified_visual_pool": True,
            "visual_pool_start": 1,
            "visual_pool_length": int(visual_tokens),
            "requested_keep_visual_tokens": int(self.config.keep_visual_tokens),
            "fastv_pruning_layer_k": int(self.config.pruning_layer),
            "fastv_reduction_ratio_r": float(reduction_ratio),
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
        return float(self.latest_stats.get("flops_reduction", 0.0) or 0.0)

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

        # FastV physically removes keys/queries.  A stale 4D mask still references
        # the old sequence, so we let Llama rebuild a fresh causal mask for the
        # compacted sequence, matching the working AutoPrune patch.
        if attention_mask is not None and attention_mask.shape[-1] != seq_len:
            attention_mask = None

        # Transformers >=4.44 may pass precomputed RoPE cos/sin.  If they are
        # still full length after pruning, gather them with the same global index
        # map used for hidden_states/position_ids.
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

    def _fastv_visual_scores(self, attn_weights: torch.Tensor, seq_len: int) -> torch.Tensor:
        """FastV Step 2: compute visual-token importance from received attention.

        The original FastV code ranks image tokens by attention received from the
        decoding/output side at the pruning layer.  For OpenVLA-OFT we do not have
        an autoregressive output-token loop, so the query side is all current
        sequence tokens except each token's own self-attention.  This explicitly
        includes OpenVLA action tokens and keeps all camera views in one score
        vector of shape [B, total_visual_tokens].
        """
        visual_positions = self.visual_positions
        if visual_positions is None or visual_positions.numel() == 0:
            return torch.empty((attn_weights.shape[0], 0), device=attn_weights.device, dtype=attn_weights.dtype)

        visual_positions = visual_positions[visual_positions < seq_len]
        if visual_positions.numel() == 0:
            return torch.empty((attn_weights.shape[0], 0), device=attn_weights.device, dtype=attn_weights.dtype)

        # attn_weights: [B, Heads, Query, Key].
        # Select visual keys as one unified pool.  There is no view split here.
        rel_attn = attn_weights[:, :, :, visual_positions]  # [B, H, Seq, V]

        # "Received by a visual token from all other tokens": exclude only the
        # visual token's own query row, while action/prompt/other-visual queries
        # remain in the average.  The mask is shared across batch and heads.
        receive_mask = torch.ones(seq_len, visual_positions.numel(), device=attn_weights.device, dtype=attn_weights.dtype)
        receive_mask[visual_positions, torch.arange(visual_positions.numel(), device=attn_weights.device)] = 0.0
        denom = receive_mask.sum(dim=0).clamp(min=1.0)
        scores = (rel_attn * receive_mask.view(1, 1, seq_len, -1)).sum(dim=2) / denom.view(1, 1, -1)
        return scores.mean(dim=1)

    def _make_layer_forward(self, layer_idx: int):
        def new_forward(self_layer, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            layer_t0 = time.perf_counter()
            bsz, seq_len, _ = hidden_states.shape

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
            need_fastv_attention = layer_idx == self.pruning_layer_idx
            kwargs["output_attentions"] = bool(original_output_attentions or need_fastv_attention)
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
            if layer_idx > self.pruning_layer_idx:
                self.latest_stats["pruned_decoder_compute_latency_ms"] += fwd_ms

            def clean_outputs(hidden_override=None):
                hidden = outputs[0] if hidden_override is None else hidden_override
                if original_output_attentions:
                    return (hidden,) + outputs[1:]
                if not need_fastv_attention:
                    return outputs if hidden_override is None else (hidden,) + outputs[1:]
                if original_use_cache and len(outputs) >= 3:
                    return (hidden, outputs[2])
                return (hidden,)

            # FastV Step 1: layers before K run exactly as the dense model.
            if layer_idx != self.pruning_layer_idx:
                self._finish_layer_stats(layer_idx, layer_t0)
                return clean_outputs()

            attn_weights = outputs[1] if len(outputs) > 1 else None
            visual_positions = self.visual_positions
            if attn_weights is None or visual_positions is None or visual_positions.numel() == 0:
                self._finish_layer_stats(layer_idx, layer_t0)
                return clean_outputs()

            score_t0 = time.perf_counter()
            visual_scores = self._fastv_visual_scores(attn_weights, outputs[0].shape[1])
            score_ms = (time.perf_counter() - score_t0) * 1000.0

            total_visual = int(visual_scores.shape[-1])
            keep_visual = max(1, min(int(self.config.keep_visual_tokens), total_visual))
            if keep_visual >= total_visual:
                self.latest_stats["final_visual_tokens"] = int(total_visual)
                self.latest_stats["vis_tokens"][int(layer_idx)] = int(total_visual)
                self.latest_stats["fastv_score_latency_ms"] += score_ms
                self._finish_layer_stats(layer_idx, layer_t0)
                return clean_outputs()

            select_t0 = time.perf_counter()
            # FastV Step 3: keep top-ranked visual tokens and prune the least
            # important R%.  R is derived from the requested keep count.
            keep_local = torch.topk(visual_scores, keep_visual, dim=-1).indices
            keep_local, _ = keep_local.sort(dim=-1)
            kept_visual_global = visual_positions[:total_visual].unsqueeze(0).expand(bsz, -1).gather(1, keep_local)
            select_ms = (time.perf_counter() - select_t0) * 1000.0

            gather_t0 = time.perf_counter()
            # FastV Step 4: physically shorten the sequence for all subsequent
            # decoder layers.  `full_indices` is sorted, so the compacted layout
            # stays [BOS] + [kept visual tokens as one block] + [all suffix tokens].
            keep_mask = torch.ones(outputs[0].shape[1], device=outputs[0].device, dtype=torch.bool)
            keep_mask[visual_positions[:total_visual]] = False
            non_visual_positions = torch.where(keep_mask)[0].unsqueeze(0).expand(bsz, -1)
            full_indices = torch.cat([kept_visual_global, non_visual_positions], dim=1)
            full_indices, _ = full_indices.sort(dim=1)

            current_hidden = outputs[0]
            pruned_hidden = current_hidden.gather(1, full_indices.unsqueeze(-1).expand(-1, -1, current_hidden.size(-1)))

            if self.global_indices is not None:
                current_globals = self.global_indices
                if current_globals.shape[0] == 1 and bsz > 1:
                    current_globals = current_globals.expand(bsz, -1)
                self.global_indices = current_globals.gather(1, full_indices)

            # The next layer sees the kept visual tokens as a single contiguous
            # pool.  Again: no per-view state is tracked.
            self.visual_positions = torch.arange(1, 1 + keep_visual, device=outputs[0].device, dtype=torch.long)
            gather_ms = (time.perf_counter() - gather_t0) * 1000.0

            overhead_ms = score_ms + select_ms + gather_ms
            self.latest_stats["fastv_score_latency_ms"] += score_ms
            self.latest_stats["fastv_select_latency_ms"] += select_ms
            self.latest_stats["fastv_gather_latency_ms"] += gather_ms
            self.latest_stats["fastv_overhead_latency_ms"] += overhead_ms
            self.latest_stats["compare_pruning_overhead_latency_ms"] += overhead_ms
            self.latest_stats["final_visual_tokens"] = int(keep_visual)
            self.latest_stats["view_keep_counts"] = [int(keep_visual)]
            self.latest_stats["keep_count_schedule"] = [[int(keep_visual)]]
            self.latest_stats["vis_tokens"][int(layer_idx)] = int(keep_visual)
            self.latest_stats["fastv_reduction_ratio_r"] = 1.0 - float(keep_visual) / float(max(total_visual, 1))
            self.latest_stats["view_score_masses"] = [float(visual_scores.sum().detach().float().cpu().item())]

            self._finish_layer_stats(layer_idx, layer_t0)
            return clean_outputs(hidden_override=pruned_hidden)

        return new_forward

    def _finish_layer_stats(self, layer_idx: int, layer_t0: float):
        layer_ms = (time.perf_counter() - layer_t0) * 1000.0
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
        self.latest_stats["decoder_layers_missing"] = max(0, int(self.num_layers - len(actual_by_layer)))
        self.latest_stats["flops_reduction"] = _estimate_flops_reduction(original, actual_by_layer, self.num_layers)

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
        """Action prediction with robust post-pruning action-token extraction."""
        self.active_visual_tokens = int(projected_patch_embeddings.shape[1])
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask

        build_t0 = time.perf_counter()
        multimodal_embeddings, multimodal_attention_mask = model_self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        if self.latest_stats is not None:
            self._pending_multimodal_build_latency_ms = (time.perf_counter() - build_t0) * 1000.0
            self.latest_stats["multimodal_build_latency_ms"] = self._pending_multimodal_build_latency_ms

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
        if self.latest_stats is not None:
            self.latest_stats["prefill_latency_ms"] = (time.perf_counter() - prefill_t0) * 1000.0

        last_hidden_states = language_model_output.hidden_states[-1]
        action_len = ACTION_DIM * NUM_ACTIONS_CHUNK
        # OpenVLA appends [action tokens] + [STOP].  Since FastV removes visual
        # tokens before actions, extracting relative to the end is safer than
        # using the original NUM_PATCHES offset.
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
        """Diffusion path equivalent: action hidden states are sliced from the end."""
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        self.active_visual_tokens = int(orig_projected_patch_embeddings.shape[1])
        curr_noisy_actions = noise
        action_len = ACTION_DIM * NUM_ACTIONS_CHUNK

        for t in action_head.noise_scheduler.timesteps:
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)

            projected_patch_embeddings = torch.cat((orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1)

            bsz = curr_noisy_actions.shape[0]
            orig_curr_noisy_actions_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(bsz, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

            input_embeddings = model_self._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, noisy_action_features
            )

            multimodal_embeddings, multimodal_attention_mask = model_self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

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

            last_hidden_states = language_model_output.hidden_states[-1]
            actions_hidden_states = last_hidden_states[:, -(action_len + 1) : -1, :]
            noise_pred = action_head.predict_noise(actions_hidden_states)
            curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states


def maybe_apply_fastv_patch(model, config: FastVConfig) -> Optional[FastVPatch]:
    if config is None or not bool(getattr(config, "enabled", False)):
        return None
    patch = FastVPatch(model=model, config=config)
    patch.apply()
    return patch
