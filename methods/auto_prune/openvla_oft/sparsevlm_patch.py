"""
SparseVLM patch for OpenVLA-OFT LIBERO evaluation.

This is a plug-and-play OpenVLA-OFT adaptation of SparseVLM-style progressive
visual token sparsification.  It follows the paper-level stages:

  1. Select image-related text tokens as raters.
  2. At decoder layers, use those raters' attention to score visual tokens.
  3. Decide how many tokens to delete from rank/redundancy signals.
  4. Rescue a subset of deleted tokens, cluster them, and merge each cluster.
  5. Continue later layers with the physically shortened sequence.

For the multi-camera VLA setting, all visual tokens are a single contiguous pool.
There is no per-view allocation or per-view pruning anywhere in this file.
"""

from __future__ import annotations

import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@dataclass
class SparseVLMConfig:
    enabled: bool = False
    target_visual_tokens: int = 128
    visual_tokens: int = 512
    pruning_start_layer: int = 1
    lambda_scale: float = 1.0
    recycle_top_ratio: float = 0.5
    cluster_ratio: float = 0.25
    rank_tolerance: float = 1e-4
    knn_k: int = 8
    min_prune_tokens: int = 1
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


def _safe_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def _effective_rank(tokens: torch.Tensor, tol: float) -> torch.Tensor:
    """Stable effective-rank proxy used for SparseVLM adaptive deletion."""
    if tokens.numel() == 0 or tokens.shape[1] <= 1:
        return tokens.new_tensor(1.0)
    x = tokens[0].float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        sigmas = torch.linalg.svdvals(x)
    except Exception:
        return tokens.new_tensor(float(max(1, min(tokens.shape[1], tokens.shape[-1]))))
    sigmas = sigmas[sigmas > float(tol)]
    if sigmas.numel() == 0:
        return tokens.new_tensor(1.0)
    probs = sigmas / sigmas.sum().clamp(min=1e-8)
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum()
    return torch.exp(entropy).to(tokens.device, dtype=tokens.dtype)


def _cluster_and_merge(tokens: torch.Tensor, cluster_num: int, knn_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Density-peak clustering merge adapted from SparseVLM's released code.

    Returns merged tokens and a representative source index for each cluster so
    position ids / RoPE embeddings can stay synchronized after synthetic tokens
    are inserted.
    """
    bsz, token_num, dim = tokens.shape
    cluster_num = max(1, min(int(cluster_num), int(token_num)))
    if token_num <= cluster_num:
        reps = torch.arange(token_num, device=tokens.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        return tokens, reps

    normalized = _safe_normalize(tokens.float())
    dist_matrix = torch.cdist(normalized, normalized, p=2).to(tokens.dtype)
    k = max(1, min(int(knn_k), token_num))
    nearest_dist, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
    density = torch.exp(-(nearest_dist.float() ** 2).mean(dim=-1)).to(tokens.dtype)
    density = density + torch.rand_like(density) * 1e-6

    higher_density = density[:, None, :] > density[:, :, None]
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist_to_higher, _ = (dist_matrix * higher_density + dist_max * (~higher_density)).min(dim=-1)
    center_score = density * dist_to_higher
    centers = torch.topk(center_score, k=cluster_num, dim=-1).indices

    center_features = tokens.gather(1, centers.unsqueeze(-1).expand(-1, -1, dim)).float()
    assign_dist = torch.cdist(tokens.float(), center_features, p=2)
    assignments = assign_dist.argmin(dim=-1)
    batch_idx = torch.arange(bsz, device=tokens.device)[:, None]
    assignments[batch_idx.expand_as(centers), centers] = torch.arange(cluster_num, device=tokens.device)[None, :]

    merged = tokens.new_zeros(bsz, cluster_num, dim)
    weights = tokens.new_zeros(bsz, cluster_num, 1)
    merged.scatter_add_(1, assignments.unsqueeze(-1).expand(-1, -1, dim), tokens)
    weights.scatter_add_(1, assignments.unsqueeze(-1), tokens.new_ones(bsz, token_num, 1))
    merged = merged / weights.clamp(min=1.0)
    return merged, centers


class SparseVLMPatch:
    def __init__(self, model, config: SparseVLMConfig):
        self.model = model
        self.config = config
        self.layers = _find_decoder_layers(model)
        self.num_layers = len(self.layers)
        self.original_forwards: Dict[int, Any] = {}
        self.original_regression_or_discrete_prediction = None
        self.original_run_diffusion_prediction = None

        self.global_indices: Optional[torch.Tensor] = None
        self.visual_positions: Optional[torch.Tensor] = None
        self.text_rater_positions: Optional[torch.Tensor] = None
        self.latest_stats: Dict[str, Any] = {}
        self._pending_multimodal_build_latency_ms = 0.0
        self.start_layer_idx = max(0, int(config.pruning_start_layer) - 1)

    def apply(self):
        if not self.layers:
            raise RuntimeError("SparseVLM could not find OpenVLA-OFT decoder layers to patch.")

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

        setattr(self.model, "_sparsevlm_patch_ref", self)
        print(
            "✅ SparseVLM applied: "
            f"target_visual_tokens={int(self.config.target_visual_tokens)}, "
            f"visual_tokens={int(self.config.visual_tokens)}, "
            f"start_layer={int(self.config.pruning_start_layer)}"
        )

    def reset_runtime_state(self, seq_len: int, device: torch.device):
        visual_tokens = max(0, min(int(self.config.visual_tokens), max(0, int(seq_len) - 1)))
        self.global_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        # Unified multi-view visual pool: [BOS] + all visual tokens + text/action.
        # No camera/view split is represented here, so pruning is global VLM-style.
        self.visual_positions = torch.arange(1, 1 + visual_tokens, device=device, dtype=torch.long)
        self.text_rater_positions = None
        active_layers = list(range(self.start_layer_idx, self.num_layers))
        self.latest_stats = {
            "method": "sparsevlm",
            "original": int(seq_len),
            "final": int(seq_len),
            "layer_tokens": {},
            "vis_tokens": {},
            "active_pruning_layers": [int(x) for x in active_layers],
            "active_pruning_layer_indices": [int(x) for x in active_layers],
            "keep_count_schedule": [],
            "raw_total_budget": float(self.config.target_visual_tokens),
            "final_visual_tokens": int(visual_tokens),
            "view_keep_counts": [int(visual_tokens)],
            "view_score_masses": [],
            "unified_visual_pool": True,
            "visual_pool_start": 1,
            "visual_pool_length": int(visual_tokens),
            "target_visual_tokens": int(self.config.target_visual_tokens),
            "decoder_layers_recorded": 0,
            "decoder_layers_missing": 0,
            "multimodal_build_latency_ms": float(self._pending_multimodal_build_latency_ms),
            "prefill_latency_ms": 0.0,
            "decode_latency_ms": 0.0,
            "decoder_forward_latency_ms": 0.0,
            "pruned_decoder_compute_latency_ms": 0.0,
            "pruning_layer_latency_ms": 0.0,
            "sparsevlm_rater_latency_ms": 0.0,
            "sparsevlm_priority_latency_ms": 0.0,
            "sparsevlm_rank_budget_latency_ms": 0.0,
            "sparsevlm_recycle_latency_ms": 0.0,
            "sparsevlm_gather_latency_ms": 0.0,
            "sparsevlm_overhead_latency_ms": 0.0,
            "compare_pruning_overhead_latency_ms": 0.0,
            "original_decoder_token_sum": 0,
            "actual_decoder_token_sum": 0,
            "saved_decoder_token_sum": 0,
            "flops_reduction": 0.0,
            "effective_rank": 0.0,
            "redundancy": 0.0,
            "recycled_tokens_total": 0,
            "merged_tokens_total": 0,
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

    def _select_text_raters(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """SparseVLM paper step: select image-related text tokens as raters.

        The released SparseVLM code computes visual-to-text affinities,
        softmaxes over text, averages over visual tokens, then keeps text tokens
        above the mean.  Here non-visual tokens include OpenVLA prompt/action
        tokens, but BOS is excluded as a rater.
        """
        if self.visual_positions is None or self.visual_positions.numel() == 0:
            return torch.empty(0, device=hidden_states.device, dtype=torch.long)
        seq_len = hidden_states.shape[1]
        visual_positions = self.visual_positions[self.visual_positions < seq_len]
        non_visual_mask = torch.ones(seq_len, device=hidden_states.device, dtype=torch.bool)
        non_visual_mask[visual_positions] = False
        non_visual_mask[0] = False
        candidate_positions = torch.where(non_visual_mask)[0]
        if candidate_positions.numel() == 0:
            return torch.empty(0, device=hidden_states.device, dtype=torch.long)

        visual_tokens = hidden_states[:, visual_positions, :]
        candidate_tokens = hidden_states[:, candidate_positions, :]
        affinity = torch.matmul(visual_tokens.float(), candidate_tokens.float().transpose(1, 2))
        relevance = affinity.softmax(dim=-1).mean(dim=1)[0]
        keep_mask = relevance > relevance.mean()
        raters = candidate_positions[keep_mask]
        if raters.numel() == 0:
            raters = candidate_positions[relevance.topk(1).indices]
        return raters

    def _layer_visual_scores(self, attn_weights: torch.Tensor, seq_len: int) -> torch.Tensor:
        visual_positions = self.visual_positions
        if visual_positions is None or visual_positions.numel() == 0:
            return torch.empty((attn_weights.shape[0], 0), device=attn_weights.device, dtype=attn_weights.dtype)
        visual_positions = visual_positions[visual_positions < seq_len]
        raters = self.text_rater_positions
        if raters is None or raters.numel() == 0:
            raters = self._select_text_raters_from_positions(seq_len, attn_weights.device)
        raters = raters[raters < seq_len]
        if raters.numel() == 0:
            raters = torch.tensor([seq_len - 1], device=attn_weights.device, dtype=torch.long)

        relation = attn_weights[:, :, raters, :][:, :, :, visual_positions]
        return relation.mean(dim=1).mean(dim=1)

    def _select_text_raters_from_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        visual_positions = self.visual_positions
        mask = torch.ones(seq_len, device=device, dtype=torch.bool)
        if visual_positions is not None:
            mask[visual_positions[visual_positions < seq_len]] = False
        mask[0] = False
        return torch.where(mask)[0]

    def _adaptive_counts(self, visual_tokens: torch.Tensor, scores: torch.Tensor, layer_idx: int) -> Tuple[int, int, int, float, float]:
        current_v = int(visual_tokens.shape[1])
        target = max(1, min(int(self.config.target_visual_tokens), current_v))
        remaining_prune_layers = max(1, self.num_layers - int(layer_idx))

        rank = _effective_rank(visual_tokens, float(self.config.rank_tolerance))
        redundancy = max(0.0, min(1.0, 1.0 - float(rank.detach().cpu().item()) / float(max(current_v, 1))))

        # SparseVLM adaptive deletion: redundancy controls the deletion pressure.
        # Then we project the keep count so the final budget is reachable without
        # collapsing all visual tokens in one layer.
        raw_delete = int(round(float(self.config.lambda_scale) * redundancy * current_v))
        if current_v > target:
            min_progress = max(int(self.config.min_prune_tokens), int((current_v - target + remaining_prune_layers - 1) // remaining_prune_layers))
            raw_delete = max(raw_delete, min_progress)
        raw_delete = max(0, min(raw_delete, max(0, current_v - target)))

        deleted_count = raw_delete
        rescue_count = int(round(deleted_count * float(self.config.recycle_top_ratio)))
        rescue_count = max(0, min(rescue_count, deleted_count))
        cluster_count = int(round(rescue_count * float(self.config.cluster_ratio)))
        cluster_count = max(0, min(cluster_count, rescue_count))

        keep_direct = current_v - deleted_count
        projected_final = keep_direct + cluster_count
        if projected_final > current_v:
            projected_final = current_v
        return keep_direct, rescue_count, cluster_count, float(rank.detach().cpu().item()), redundancy

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

            attention_mask, position_ids, kwargs = self._align_position_inputs(hidden_states, attention_mask, position_ids, kwargs)

            original_output_attentions = bool(kwargs.get("output_attentions", False))
            can_prune = layer_idx >= self.start_layer_idx
            kwargs["output_attentions"] = bool(original_output_attentions or can_prune)
            original_use_cache = bool(kwargs.get("use_cache", False))

            if layer_idx == self.start_layer_idx and self.text_rater_positions is None:
                rater_t0 = time.perf_counter()
                self.text_rater_positions = self._select_text_raters(hidden_states)
                self.latest_stats["sparsevlm_rater_latency_ms"] += (time.perf_counter() - rater_t0) * 1000.0
                self.latest_stats["text_rater_count"] = int(self.text_rater_positions.numel())

            fwd_t0 = time.perf_counter()
            outputs = self.original_forwards[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            fwd_ms = (time.perf_counter() - fwd_t0) * 1000.0
            self.latest_stats["decoder_forward_latency_ms"] += fwd_ms
            if layer_idx >= self.start_layer_idx:
                self.latest_stats["pruned_decoder_compute_latency_ms"] += fwd_ms

            def clean_outputs(hidden_override=None):
                hidden = outputs[0] if hidden_override is None else hidden_override
                if original_output_attentions:
                    return (hidden,) + outputs[1:]
                if not can_prune:
                    return outputs if hidden_override is None else (hidden,) + outputs[1:]
                if original_use_cache and len(outputs) >= 3:
                    return (hidden, outputs[2])
                return (hidden,)

            if not can_prune or self.visual_positions is None or self.visual_positions.numel() == 0:
                self._finish_layer_stats(layer_idx, layer_t0)
                return clean_outputs()

            current_hidden = outputs[0]
            seq_after = current_hidden.shape[1]
            visual_positions = self.visual_positions[self.visual_positions < seq_after]
            current_v = int(visual_positions.numel())
            target = max(1, int(self.config.target_visual_tokens))
            if current_v <= target:
                self.latest_stats["final_visual_tokens"] = int(current_v)
                self.latest_stats["vis_tokens"][int(layer_idx)] = int(current_v)
                self._finish_layer_stats(layer_idx, layer_t0)
                return clean_outputs()

            attn_weights = outputs[1] if len(outputs) > 1 else None
            if attn_weights is None:
                self._finish_layer_stats(layer_idx, layer_t0)
                return clean_outputs()

            priority_t0 = time.perf_counter()
            scores = self._layer_visual_scores(attn_weights, seq_after)
            priority_ms = (time.perf_counter() - priority_t0) * 1000.0
            self.latest_stats["sparsevlm_priority_latency_ms"] += priority_ms

            visual_tokens = current_hidden[:, visual_positions, :]
            budget_t0 = time.perf_counter()
            keep_direct, rescue_count, cluster_count, rank, redundancy = self._adaptive_counts(visual_tokens, scores, layer_idx)
            rank_budget_ms = (time.perf_counter() - budget_t0) * 1000.0
            self.latest_stats["sparsevlm_rank_budget_latency_ms"] += rank_budget_ms
            self.latest_stats["effective_rank"] = float(rank)
            self.latest_stats["redundancy"] = float(redundancy)

            if keep_direct >= current_v and cluster_count <= 0:
                overhead_ms = priority_ms + rank_budget_ms
                self.latest_stats["sparsevlm_overhead_latency_ms"] += overhead_ms
                self.latest_stats["compare_pruning_overhead_latency_ms"] += overhead_ms
                self.latest_stats["final_visual_tokens"] = int(current_v)
                self.latest_stats["vis_tokens"][int(layer_idx)] = int(current_v)
                self._finish_layer_stats(layer_idx, layer_t0)
                return clean_outputs()

            gather_t0 = time.perf_counter()
            score_order = torch.argsort(scores, dim=-1, descending=True)
            keep_local = score_order[:, :keep_direct]
            deleted_local = score_order[:, keep_direct:]
            keep_local, _ = keep_local.sort(dim=-1)

            kept_visual = visual_positions.unsqueeze(0).expand(bsz, -1).gather(1, keep_local)
            kept_visual_hidden = current_hidden.gather(
                1,
                kept_visual.unsqueeze(-1).expand(-1, -1, current_hidden.size(-1)),
            )
            if self.global_indices is not None:
                current_globals = self.global_indices.expand(bsz, -1) if self.global_indices.shape[0] == 1 and bsz > 1 else self.global_indices
                kept_visual_global = current_globals.gather(1, kept_visual)
            else:
                kept_visual_global = kept_visual

            merged_hidden = current_hidden.new_empty(bsz, 0, current_hidden.size(-1))
            merged_global = kept_visual.new_empty(bsz, 0)
            recycle_ms = 0.0
            if deleted_local.shape[1] > 0 and rescue_count > 0 and cluster_count > 0:
                recycle_t0 = time.perf_counter()
                deleted_scores = scores.gather(1, deleted_local)
                rescue_in_deleted = torch.topk(deleted_scores, k=min(rescue_count, deleted_scores.shape[1]), dim=-1).indices
                rescue_local = deleted_local.gather(1, rescue_in_deleted)
                rescue_positions = visual_positions.unsqueeze(0).expand(bsz, -1).gather(1, rescue_local)
                rescue_hidden = current_hidden.gather(
                    1,
                    rescue_positions.unsqueeze(-1).expand(-1, -1, current_hidden.size(-1)),
                )
                merged_hidden, rep_local = _cluster_and_merge(rescue_hidden, cluster_count, int(self.config.knn_k))
                if self.global_indices is not None:
                    rescue_global = current_globals.gather(1, rescue_positions)
                    merged_global = rescue_global.gather(1, rep_local)
                else:
                    merged_global = rescue_positions.gather(1, rep_local)
                recycle_ms = (time.perf_counter() - recycle_t0) * 1000.0
                self.latest_stats["sparsevlm_recycle_latency_ms"] += recycle_ms
                self.latest_stats["recycled_tokens_total"] += int(rescue_count)
                self.latest_stats["merged_tokens_total"] += int(merged_hidden.shape[1])

            non_visual_mask = torch.ones(seq_after, device=current_hidden.device, dtype=torch.bool)
            non_visual_mask[visual_positions] = False
            non_visual_positions = torch.where(non_visual_mask)[0].unsqueeze(0).expand(bsz, -1)
            non_visual_hidden = current_hidden.gather(
                1,
                non_visual_positions.unsqueeze(-1).expand(-1, -1, current_hidden.size(-1)),
            )
            if self.global_indices is not None:
                non_visual_global = current_globals.gather(1, non_visual_positions)
            else:
                non_visual_global = non_visual_positions

            visual_hidden_all = torch.cat([kept_visual_hidden, merged_hidden], dim=1)
            visual_global_all = torch.cat([kept_visual_global, merged_global], dim=1)
            if visual_global_all.shape[1] > 1:
                visual_order = torch.argsort(visual_global_all, dim=1)
                visual_hidden_all = visual_hidden_all.gather(
                    1,
                    visual_order.unsqueeze(-1).expand(-1, -1, visual_hidden_all.size(-1)),
                )
                visual_global_all = visual_global_all.gather(1, visual_order)

            # Physical sequence shortening: [BOS] + all kept/merged visual
            # tokens + all non-visual suffix.  The visual segment remains
            # contiguous and unified for the next SparseVLM layer.
            bos_hidden = non_visual_hidden[:, :1, :]
            suffix_hidden = non_visual_hidden[:, 1:, :]
            pruned_hidden = torch.cat([bos_hidden, visual_hidden_all, suffix_hidden], dim=1)

            bos_global = non_visual_global[:, :1]
            suffix_global = non_visual_global[:, 1:]
            self.global_indices = torch.cat([bos_global, visual_global_all, suffix_global], dim=1)
            new_visual_count = int(visual_hidden_all.shape[1])
            self.visual_positions = torch.arange(1, 1 + new_visual_count, device=current_hidden.device, dtype=torch.long)
            if self.text_rater_positions is not None:
                self.text_rater_positions = self.text_rater_positions - (current_v - new_visual_count)
                self.text_rater_positions = self.text_rater_positions[self.text_rater_positions > new_visual_count]

            gather_ms = (time.perf_counter() - gather_t0) * 1000.0
            overhead_ms = priority_ms + rank_budget_ms + recycle_ms + gather_ms
            self.latest_stats["sparsevlm_gather_latency_ms"] += gather_ms
            self.latest_stats["sparsevlm_overhead_latency_ms"] += overhead_ms
            self.latest_stats["compare_pruning_overhead_latency_ms"] += overhead_ms
            self.latest_stats["final_visual_tokens"] = int(new_visual_count)
            self.latest_stats["view_keep_counts"] = [int(new_visual_count)]
            self.latest_stats["keep_count_schedule"].append([int(new_visual_count)])
            self.latest_stats["vis_tokens"][int(layer_idx)] = int(new_visual_count)
            self.latest_stats["view_score_masses"] = [float(scores.sum().detach().float().cpu().item())]

            self._finish_layer_stats(layer_idx, layer_t0)
            return clean_outputs(hidden_override=pruned_hidden)

        return new_forward

    def _finish_layer_stats(self, layer_idx: int, layer_t0: float):
        layer_ms = (time.perf_counter() - layer_t0) * 1000.0
        if layer_idx >= self.start_layer_idx:
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
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask

        build_t0 = time.perf_counter()
        multimodal_embeddings, multimodal_attention_mask = model_self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
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
        if self.latest_stats is not None:
            self.latest_stats["prefill_latency_ms"] = (time.perf_counter() - prefill_t0) * 1000.0
            self.latest_stats["multimodal_build_latency_ms"] = float(self._pending_multimodal_build_latency_ms)

        last_hidden_states = language_model_output.hidden_states[-1]
        action_len = ACTION_DIM * NUM_ACTIONS_CHUNK
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
        action_len = ACTION_DIM * NUM_ACTIONS_CHUNK

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


def maybe_apply_sparsevlm_patch(model, config: SparseVLMConfig) -> Optional[SparseVLMPatch]:
    if config is None or not bool(getattr(config, "enabled", False)):
        return None
    patch = SparseVLMPatch(model=model, config=config)
    patch.apply()
    return patch
