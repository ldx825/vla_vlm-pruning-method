"""
VLA-ADP patch for OpenVLA-OFT LIBERO evaluation.

The adaptation follows Pei et al. (2025): visual tokens are selected before the
LLM body using lightweight text-to-vision QK scores, and the dynamic gate uses
the previous open-loop action window to decide whether the current window should
be pruned.
"""

from __future__ import annotations

import math
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@dataclass
class VLAADPOFTConfig:
    enabled: bool = False
    keep_visual_tokens: int = 224
    visual_tokens: int = 512
    num_images_in_input: int = 2
    qk_layer: int = 0
    main_view_weight: float = 0.4
    wrist_view_weight: float = 0.6
    use_dynamic_visual_strategy: bool = True
    decision_method: str = "adjacent"
    adjacent_variant: str = "extrema"
    adjacent_extrema_window: int = 3
    adjacent_lookback: int = 2
    adjacent_last_state: bool = True
    initial_state: int = 0
    delta_method: str = "net"
    min_delta_pos: float = 0.0
    min_delta_rot: float = 0.0
    hysteresis_up: float = 0.0
    hysteresis_down: float = 0.0
    tol_equal: float = 0.0
    cold_start_windows: int = 2
    limit_consecutive_pruned_enabled: bool = True
    limit_max_consecutive_pruned: int = 3
    debug: bool = False


def _find_decoder_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
        if hasattr(lm, "layers"):
            return list(lm.layers)
    return []


def _quat_to_axis_angle(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[0] != 4:
        return np.zeros(3, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    w = float(np.clip(w, -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return np.asarray([x / s, y / s, z / s], dtype=np.float64) * angle


class _ActionTrajectoryGate:
    """Paper Steps 7-9: online action-aware pruning gate."""

    def __init__(self, config: VLAADPOFTConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.window_index = 0
        self.curr_state = int(self.config.initial_state)
        self.frame_counter_global = 0
        self.curr_window_start_frame = None
        self.pos_list: List[np.ndarray] = []
        self.aa_list: List[np.ndarray] = []
        self.history_d: List[float] = []
        self.avg_d: Optional[float] = None
        self.last_delta = 0.0
        self.last_forced_full = False
        self.executed_pruned_streak = 0

    def current_should_prune(self) -> bool:
        if not bool(self.config.use_dynamic_visual_strategy):
            return True
        if self.window_index < max(0, int(self.config.cold_start_windows)):
            return False
        if int(self.curr_state) != 1:
            return False
        # Step 9: force one full-vision window after three actually pruned forwards.
        if bool(self.config.limit_consecutive_pruned_enabled):
            max_ones = max(1, int(self.config.limit_max_consecutive_pruned))
            if self.executed_pruned_streak >= max_ones:
                self.curr_state = 0
                self.executed_pruned_streak = 0
                self.last_forced_full = True
                return False
        return True

    def record_forward_decision(self, actually_pruned: bool):
        if bool(actually_pruned):
            self.executed_pruned_streak += 1
        else:
            self.executed_pruned_streak = 0

    def observe_step(self, eef_pos, eef_quat):
        if self.curr_window_start_frame is None:
            self.curr_window_start_frame = self.frame_counter_global
        self.pos_list.append(np.asarray(eef_pos, dtype=np.float64))
        self.aa_list.append(_quat_to_axis_angle(eef_quat))
        self.frame_counter_global += 1

    def _compute_window_delta(self) -> float:
        if len(self.pos_list) <= 1:
            return 0.0
        positions = np.stack(self.pos_list, axis=0)
        method = (self.config.delta_method or "net").lower()
        if method == "net":
            return float(np.linalg.norm(positions[-1] - positions[0]))
        step_pos = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        step_pos = np.where(step_pos < float(self.config.min_delta_pos), 0.0, step_pos)
        return float(np.sum(step_pos))

    def close_window_and_decide_next(self):
        if self.curr_window_start_frame is None:
            return

        # Step 7: use the just-finished action trajectory window to decide the next forward.
        d = self._compute_window_delta()
        self.last_delta = float(d)
        dm = (self.config.decision_method or "adjacent").lower()

        if dm == "avg":
            self.avg_d = float(d) if self.avg_d is None else 0.5 * float(self.avg_d) + 0.5 * float(d)
            next_state = 1 if d >= float(self.avg_d) else 0
        else:
            self.history_d.append(float(d))
            tau = max(1, int(self.config.adjacent_extrema_window))
            if len(self.history_d) < tau:
                next_state = int(self.config.initial_state)
            else:
                # Paper Eq. 17-18: compare the current window delta with the
                # adjacent extrema over the recent tau-window, and keep the
                # previous state in the middle band instead of pruning by default.
                recent_vals = self.history_d[-tau:]
                up_thr = max(recent_vals) + float(self.config.hysteresis_up)
                dn_thr = min(recent_vals) - float(self.config.hysteresis_down)
                tol = float(self.config.tol_equal)
                # Step 8: coarse motion windows prune; fine motion windows keep full visual context.
                if d + tol >= up_thr:
                    next_state = 1
                elif d < dn_thr - tol:
                    next_state = 0
                else:
                    next_state = int(self.curr_state) if bool(self.config.adjacent_last_state) else 1

        if int(next_state) == 0:
            self.last_forced_full = False
        self.curr_state = int(next_state)
        self.window_index += 1
        self.pos_list.clear()
        self.aa_list.clear()
        self.curr_window_start_frame = None


class VLAADPOFTPatch:
    def __init__(self, model, config: VLAADPOFTConfig):
        self.model = model
        self.config = config
        self.layers = _find_decoder_layers(model)
        self.num_layers = len(self.layers)
        self.gate = _ActionTrajectoryGate(config)
        self.original_regression_or_discrete_prediction = None
        self.original_run_diffusion_prediction = None
        self.latest_stats: Dict[str, Any] = {}

    def apply(self):
        if hasattr(self.model, "_regression_or_discrete_prediction"):
            self.original_regression_or_discrete_prediction = self.model._regression_or_discrete_prediction
            self.model._regression_or_discrete_prediction = types.MethodType(
                self._patched_regression_or_discrete_prediction,
                self.model,
            )
        else:
            raise RuntimeError("VLA-ADP-OFT could not find _regression_or_discrete_prediction to patch.")

        if hasattr(self.model, "_run_diffusion_prediction"):
            self.original_run_diffusion_prediction = self.model._run_diffusion_prediction
            self.model._run_diffusion_prediction = types.MethodType(self._patched_run_diffusion_prediction, self.model)

        setattr(self.model, "_vlaadp_patch_ref", self)
        print(
            "VLA-ADP-OFT applied: "
            f"keep_visual_tokens={int(self.config.keep_visual_tokens)}, "
            f"qk_layer={int(self.config.qk_layer)}, "
            f"view_split={float(self.config.main_view_weight):.2f}:{float(self.config.wrist_view_weight):.2f}, "
            f"dynamic_gate={bool(self.config.use_dynamic_visual_strategy)}"
        )

    def reset_episode_history(self, average_budget_horizon_steps: Optional[int] = None):
        del average_budget_horizon_steps
        self.gate.reset()
        self.latest_stats = {}

    def observe_trajectory_step(self, eef_pos, eef_quat):
        self.gate.observe_step(eef_pos, eef_quat)

    def close_action_window(self):
        self.gate.close_window_and_decide_next()

    def get_flops_reduction(self) -> float:
        return float((self.latest_stats or {}).get("flops_reduction", 0.0) or 0.0)

    def _init_stats(self, total_visual: int, should_prune: bool, keep_total: int):
        self.latest_stats = {
            "method": "vla_adp_oft",
            "original": 0,
            "final": 0,
            "layer_tokens": {},
            "active_pruning_layers": list(range(self.num_layers)) if should_prune and keep_total < total_visual else [],
            "keep_count_schedule": [],
            "raw_total_budget": float(keep_total if should_prune else total_visual),
            "target_visual_tokens": int(self.config.keep_visual_tokens),
            "final_visual_tokens": int(total_visual),
            "view_keep_counts": [],
            "view_score_masses": [],
            "per_view_pruning": True,
            "unified_visual_pool": False,
            "qk_layer": int(self.config.qk_layer),
            "vlaadp_should_prune": int(bool(should_prune)),
            "vlaadp_window_index": int(self.gate.window_index),
            "vlaadp_gate_state": int(self.gate.curr_state),
            "vlaadp_last_delta": float(self.gate.last_delta),
            "vlaadp_executed_pruned_streak": int(self.gate.executed_pruned_streak),
            "vlaadp_forced_full": int(bool(self.gate.last_forced_full)),
            "decoder_layers_recorded": int(self.num_layers),
            "decoder_layers_missing": 0,
            "multimodal_build_latency_ms": 0.0,
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
            "estimated_dense_decoder_latency_ms_linear_proxy": 0.0,
            "estimated_saved_decoder_compute_latency_ms_linear_proxy": 0.0,
            "estimated_net_decoder_latency_delta_ms_linear_proxy": 0.0,
            "original_decoder_token_sum": 0,
            "actual_decoder_token_sum": 0,
            "saved_decoder_token_sum": 0,
            "flops_reduction": 0.0,
        }

    def _compute_qk_importance_generic(
        self,
        language_model,
        text_embeddings: torch.Tensor,
        vision_embeddings: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Paper Step 2: lightweight text-to-vision QK score before the LLM forward."""
        if hasattr(language_model, "transformer") and hasattr(language_model.transformer, "h"):
            layer = language_model.transformer.h[layer_idx]
            text_h = layer.ln_1(text_embeddings) if hasattr(layer, "ln_1") else text_embeddings
            vision_h = layer.ln_1(vision_embeddings) if hasattr(layer, "ln_1") else vision_embeddings
            qkv_text = layer.attn.c_attn(text_h)
            qkv_vis = layer.attn.c_attn(vision_h)
            hidden = qkv_text.shape[-1] // 3
            q_text = qkv_text[..., :hidden]
            k_vis = qkv_vis[..., hidden : 2 * hidden]
            n_head = int(language_model.config.n_head)
            head_dim = int(language_model.config.n_embd) // n_head
        else:
            layers = language_model.model.layers
            layer_idx = max(0, min(int(layer_idx), len(layers) - 1))
            layer = layers[layer_idx]
            text_h = layer.input_layernorm(text_embeddings) if hasattr(layer, "input_layernorm") else text_embeddings
            vision_h = layer.input_layernorm(vision_embeddings) if hasattr(layer, "input_layernorm") else vision_embeddings
            attn = layer.self_attn
            q_text = F.linear(text_h, attn.q_proj.weight, getattr(attn.q_proj, "bias", None))
            k_vis = F.linear(vision_h, attn.k_proj.weight, getattr(attn.k_proj, "bias", None))
            n_head = int(language_model.config.num_attention_heads)
            head_dim = int(language_model.config.hidden_size) // n_head

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            bsz, seq, hidden = x.shape
            return x.view(bsz, seq, n_head, head_dim).permute(0, 2, 1, 3).contiguous()

        q = split_heads(q_text)
        k = split_heads(k_vis)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(float(head_dim)))
        # Step 3: aggregate over batch, heads, and text tokens into one importance per visual token.
        return scores.float().mean(dim=(0, 1, 2))

    def _per_view_keep_counts(self, total_visual: int) -> List[int]:
        keep_total = max(1, min(int(self.config.keep_visual_tokens), int(total_visual)))
        num_images = max(1, int(self.config.num_images_in_input))
        if num_images == 1:
            return [keep_total]

        if num_images == 2:
            weights = [float(self.config.main_view_weight), float(self.config.wrist_view_weight)]
        else:
            main_w = float(self.config.main_view_weight)
            wrist_total = float(self.config.wrist_view_weight)
            weights = [main_w] + [wrist_total / float(num_images - 1)] * (num_images - 1)
        weight_sum = sum(max(0.0, w) for w in weights) or float(num_images)
        weights = [max(0.0, w) / weight_sum for w in weights]

        per_view = [max(1, int(round(keep_total * w))) for w in weights]
        per_image = int(total_visual) // num_images
        per_view = [min(k, per_image) for k in per_view]
        diff = keep_total - sum(per_view)
        order = sorted(range(num_images), key=lambda i: weights[i], reverse=True)
        ptr = 0
        while diff != 0 and order:
            idx = order[ptr % len(order)]
            if diff > 0 and per_view[idx] < per_image:
                per_view[idx] += 1
                diff -= 1
            elif diff < 0 and per_view[idx] > 1:
                per_view[idx] -= 1
                diff += 1
            ptr += 1
            if ptr > 10000:
                break
        return per_view

    def _select_and_prune_visual_tokens(
        self,
        model_self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        projected_total = int(projected_patch_embeddings.shape[1])
        total_visual = min(int(self.config.visual_tokens), projected_total)
        visual_embeddings = projected_patch_embeddings[:, :total_visual, :]
        projected_suffix = projected_patch_embeddings[:, total_visual:, :]
        should_prune = bool(self.gate.current_should_prune())
        keep_total = max(1, min(int(self.config.keep_visual_tokens), total_visual))
        self._init_stats(total_visual, should_prune, keep_total)
        if (not should_prune) or keep_total >= total_visual:
            self.gate.record_forward_decision(False)
            self.latest_stats["vlaadp_executed_pruned_streak"] = int(self.gate.executed_pruned_streak)
            view_tokens = self._dense_view_counts(total_visual)
            self.latest_stats["view_keep_counts"] = view_tokens
            return projected_patch_embeddings, {"score_ms": 0.0, "select_ms": 0.0, "gather_ms": 0.0}

        # Step 2: use text and vision embeddings with existing Q/K projections, before the main LLM forward.
        score_t0 = time.perf_counter()
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0],
            -1,
            input_embeddings.shape[-1],
        )
        qk_layer = int(self.config.qk_layer)
        if qk_layer > 0:
            with torch.no_grad():
                zero_action_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)
                probe_embeddings, probe_mask = model_self._build_multimodal_attention(
                    zero_action_embeddings,
                    projected_patch_embeddings,
                    attention_mask,
                )
                probe_out = model_self.language_model(
                    input_ids=None,
                    attention_mask=probe_mask,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=probe_embeddings,
                    labels=None,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                seq_h = probe_out.hidden_states[qk_layer]
            vis_h = seq_h[:, 1 : 1 + total_visual, :]
            tail_h = seq_h[:, 1 + projected_total :, :]
            tail_mask = (~all_actions_mask[:, 1:]).unsqueeze(-1).expand_as(tail_h)
            tail_kept = tail_h[tail_mask].reshape(seq_h.shape[0], -1, seq_h.shape[-1])
            text_emb = torch.cat([seq_h[:, :1, :], tail_kept], dim=1)
            vis_emb = vis_h
        else:
            text_emb = language_embeddings
            vis_emb = visual_embeddings
        with torch.no_grad():
            importance = self._compute_qk_importance_generic(model_self.language_model, text_emb, vis_emb, qk_layer)
        score_ms = (time.perf_counter() - score_t0) * 1000.0

        # Step 4 and Step 5: top-k per view with paper's multi-view 4:6 main/wrist allocation.
        select_t0 = time.perf_counter()
        per_view_keep = self._per_view_keep_counts(total_visual)
        num_images = len(per_view_keep)
        per_image = total_visual // max(1, num_images)
        kept: List[torch.Tensor] = []
        view_masses: List[float] = []
        for view_idx, keep_i in enumerate(per_view_keep):
            start = view_idx * per_image
            end = total_visual if view_idx == num_images - 1 else start + per_image
            local_scores = importance[start:end]
            keep_i = max(1, min(int(keep_i), int(local_scores.numel())))
            top_local = torch.topk(local_scores, k=keep_i, largest=True, sorted=False).indices
            kept.append((top_local + start).to(projected_patch_embeddings.device, dtype=torch.long))
            view_masses.append(float(local_scores.sum().detach().cpu().item()))
        keep_idx = torch.sort(torch.cat(kept, dim=0).long())[0]
        select_ms = (time.perf_counter() - select_t0) * 1000.0

        # Step 6: physically rebuild the visual sequence; all later Transformer layers process fewer tokens.
        gather_t0 = time.perf_counter()
        pruned_visual = visual_embeddings.index_select(dim=1, index=keep_idx)
        if projected_suffix.shape[1] > 0:
            pruned_projected = torch.cat([pruned_visual, projected_suffix], dim=1)
        else:
            pruned_projected = pruned_visual
        gather_ms = (time.perf_counter() - gather_t0) * 1000.0

        overhead_ms = score_ms + select_ms + gather_ms
        self.latest_stats["fastv_score_latency_ms"] = float(score_ms)
        self.latest_stats["fastv_select_latency_ms"] = float(select_ms)
        self.latest_stats["fastv_gather_latency_ms"] = float(gather_ms)
        self.latest_stats["fastv_overhead_latency_ms"] = float(overhead_ms)
        self.latest_stats["compare_pruning_overhead_latency_ms"] = float(overhead_ms)
        self.latest_stats["final_visual_tokens"] = int(pruned_visual.shape[1])
        self.latest_stats["view_keep_counts"] = [int(x) for x in per_view_keep]
        self.latest_stats["view_score_masses"] = view_masses
        self.latest_stats["keep_count_schedule"] = [[int(x) for x in per_view_keep]]
        self.gate.record_forward_decision(True)
        self.latest_stats["vlaadp_executed_pruned_streak"] = int(self.gate.executed_pruned_streak)
        return pruned_projected, {"score_ms": score_ms, "select_ms": select_ms, "gather_ms": gather_ms}

    def _dense_view_counts(self, total_visual: int) -> List[int]:
        num_images = max(1, int(self.config.num_images_in_input))
        per_image = int(total_visual) // num_images
        counts = [per_image] * num_images
        counts[-1] += int(total_visual) - sum(counts)
        return counts

    def _finish_stats(self, full_seq_len: int, actual_seq_len: int, prefill_ms: float, build_ms: float):
        num_layers = max(1, int(self.num_layers))
        actual_by_layer = {i: int(actual_seq_len) for i in range(num_layers)}
        self.latest_stats["original"] = int(full_seq_len)
        self.latest_stats["final"] = int(actual_seq_len)
        self.latest_stats["layer_tokens"] = actual_by_layer
        self.latest_stats["decoder_layers_recorded"] = int(num_layers)
        self.latest_stats["decoder_layers_missing"] = 0
        self.latest_stats["multimodal_build_latency_ms"] = float(build_ms)
        self.latest_stats["prefill_latency_ms"] = float(prefill_ms)
        self.latest_stats["decoder_forward_latency_ms"] = float(prefill_ms)
        if self.latest_stats.get("vlaadp_should_prune"):
            self.latest_stats["pruned_decoder_compute_latency_ms"] = float(prefill_ms)
        dense = int(full_seq_len) * num_layers
        actual = int(actual_seq_len) * num_layers
        saved = max(0, dense - actual)
        self.latest_stats["original_decoder_token_sum"] = int(dense)
        self.latest_stats["actual_decoder_token_sum"] = int(actual)
        self.latest_stats["saved_decoder_token_sum"] = int(saved)
        self.latest_stats["flops_reduction"] = float(saved / max(1, dense))
        if actual_seq_len > 0:
            dense_est = float(prefill_ms) * (float(full_seq_len) / float(actual_seq_len))
            self.latest_stats["estimated_dense_decoder_latency_ms_linear_proxy"] = dense_est
            self.latest_stats["estimated_saved_decoder_compute_latency_ms_linear_proxy"] = max(0.0, dense_est - prefill_ms)
            self.latest_stats["estimated_net_decoder_latency_delta_ms_linear_proxy"] = (
                self.latest_stats["estimated_saved_decoder_compute_latency_ms_linear_proxy"]
                - float(self.latest_stats.get("compare_pruning_overhead_latency_ms", 0.0) or 0.0)
            )

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
        del labels, NUM_PATCHES, NUM_PROMPT_TOKENS
        # Step 1: receive multimodal inputs after image projection and prompt/action token preparation.
        pruned_projected, _ = self._select_and_prune_visual_tokens(
            model_self,
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
        )

        all_actions_mask_3d = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask_3d

        build_t0 = time.perf_counter()
        multimodal_embeddings, multimodal_attention_mask = model_self._build_multimodal_attention(
            input_embeddings,
            pruned_projected,
            attention_mask,
        )
        build_ms = (time.perf_counter() - build_t0) * 1000.0
        full_seq_len = int(multimodal_embeddings.shape[1] + (projected_patch_embeddings.shape[1] - pruned_projected.shape[1]))
        actual_seq_len = int(multimodal_embeddings.shape[1])

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
        prefill_ms = (time.perf_counter() - prefill_t0) * 1000.0
        self._finish_stats(full_seq_len=full_seq_len, actual_seq_len=actual_seq_len, prefill_ms=prefill_ms, build_ms=build_ms)

        # Step 10: output the action chunk from the shortened multimodal sequence.
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
        # The LIBERO OpenVLA-OFT runs here use L1 action heads. For diffusion checkpoints,
        # preserve correctness by delegating to the original dense implementation.
        if self.original_run_diffusion_prediction is None:
            raise RuntimeError("Original diffusion prediction function is unavailable.")
        self._init_stats(int(projected_patch_embeddings.shape[1]), should_prune=False, keep_total=int(projected_patch_embeddings.shape[1]))
        return self.original_run_diffusion_prediction(
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
        )


def maybe_apply_vlaadp_oft_patch(model, config: VLAADPOFTConfig) -> Optional[VLAADPOFTPatch]:
    if not config.enabled:
        return None
    patch = VLAADPOFTPatch(model=model, config=config)
    patch.apply()
    return patch
