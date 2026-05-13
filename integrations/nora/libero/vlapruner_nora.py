from __future__ import annotations

import time
import types
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch


@dataclass
class NoraVLAPrunerConfig:
    enabled: bool = False
    keep_visual_tokens: int = 224
    visual_tokens: int = 512
    num_views: int = 2
    pruning_layer: int = 3
    temporal_window: int = 3
    temporal_gamma: float = 0.8
    warmup_steps: int = 3
    action_attention_layer_start: str = "half"
    action_token_min: int = 151665
    action_token_max: int = 153712
    decode_skip_seq_len: int = 96
    calibrate_average_keep: bool = True
    # run_libero_eval_nora excludes the first policy call in each episode from
    # averaged latency/FLOPs stats.  The budget calibration mirrors that metric
    # convention so logged "Avg final visual tokens" targets keep_visual_tokens.
    measured_skip_initial_calls: int = 1
    # Initial estimate for measured policy calls per episode.  It is updated
    # with completed episodes, but this gives episode 1 a stable fixed budget.
    average_target_measured_steps: int = 40
    debug: bool = False


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


class NoraVLAPrunerPatch:
    """VLA-Pruner adapted to Nora/Qwen2.5-VL autoregressive action chunks.

    This patch intentionally prunes all visual tokens from all camera views as
    one unified VLM-style visual pool.  It does not allocate a separate budget
    per view; view counts are logged only for diagnostics.
    """

    def __init__(self, model: Any, config: NoraVLAPrunerConfig):
        self.model = model
        self.config = config
        self.layers = self._resolve_layers(model)
        self.num_layers = len(self.layers)
        self.original_forwards: Dict[int, Any] = {}
        self.global_indices: Optional[torch.Tensor] = None
        self.input_layout: Dict[str, Any] = {}
        self.runtime_stats: Dict[str, Any] = {}

        self.pruning_layer_idx = max(0, int(config.pruning_layer) - 1)
        self.action_layer_start_idx = self._resolve_action_layer_start(config.action_attention_layer_start)
        self.action_attention_history: Deque[torch.Tensor] = deque(maxlen=max(1, int(config.temporal_window)))

        self._decode_action_vectors_by_step: List[torch.Tensor] = []
        self._current_decode_layer_vectors: List[torch.Tensor] = []
        self._prefill_active = False
        self._has_pruned_this_forward = False
        self._episode_policy_calls = 0
        self._episode_measured_calls = 0
        self._episode_estimated_measured_calls: Optional[float] = None
        self._current_call_measured_seen_before = 0
        self._current_call_is_measured = False
        self._episode_adjusted_keep_count: Optional[int] = None

    @staticmethod
    def _resolve_layers(model):
        candidates = [
            ["model", "layers"],
            ["language_model", "model", "layers"],
            ["language_model", "layers"],
            ["base_model", "model", "model", "layers"],
            ["base_model", "model", "layers"],
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
        raise RuntimeError("Unable to locate decoder layers for VLA-Pruner in the provided Nora model.")

    def _resolve_action_layer_start(self, value: str) -> int:
        if str(value).lower() == "half":
            return max(0, self.num_layers // 2)
        try:
            return max(0, min(self.num_layers - 1, int(value)))
        except Exception:
            return max(0, self.num_layers // 2)

    def apply(self) -> None:
        if not self.config.enabled:
            return
        for idx, layer in enumerate(self.layers):
            self.original_forwards[idx] = layer.forward
            layer.forward = types.MethodType(self._make_new_forward(idx), layer)

    def reset_episode_history(self) -> None:
        if self._episode_measured_calls > 0:
            previous = float(self._episode_measured_calls)
            if self._episode_estimated_measured_calls is None:
                self._episode_estimated_measured_calls = previous
            else:
                self._episode_estimated_measured_calls = (
                    0.7 * float(self._episode_estimated_measured_calls) + 0.3 * previous
                )
        self.action_attention_history.clear()
        self._episode_policy_calls = 0
        self._episode_measured_calls = 0
        self._current_call_measured_seen_before = 0
        self._current_call_is_measured = False
        self._episode_adjusted_keep_count = None

    def reset_runtime_state(self) -> None:
        self.global_indices = None
        self.input_layout = {}
        self._decode_action_vectors_by_step = []
        self._current_decode_layer_vectors = []
        self._prefill_active = False
        self._has_pruned_this_forward = False
        self.runtime_stats = {
            "patch_total_seconds": 0.0,
            "force_attn_seconds": 0.0,
            "force_attn_self_attn_seconds": 0.0,
            "force_attn_mlp_seconds": 0.0,
            "force_attn_residual_norm_seconds": 0.0,
            "regular_forward_seconds": 0.0,
            "regular_self_attn_seconds": 0.0,
            "regular_mlp_seconds": 0.0,
            "regular_residual_norm_seconds": 0.0,
            "manual_probe_seconds": 0.0,
            "metric_seconds": 0.0,
            "rank_seconds": 0.0,
            "budget_schedule_seconds": 0.0,
            "compaction_bookkeeping_seconds": 0.0,
            "topk_seconds": 0.0,
            "gather_seconds": 0.0,
            "last_tokens_before": 0,
            "last_tokens_after": 0,
            "schedule_len": 1,
            "pruned_layers": 0,
            "layout_ready": 0,
            "layout_query_tokens": 0,
            "layout_view0_tokens": 0,
            "layout_view1_tokens": 0,
            "layer0_total_calls": 0,
            "layer0_short_seq_calls": 0,
            "layer0_long_seq_calls": 0,
            "layer0_past_none_calls": 0,
            "layer0_past_present_calls": 0,
            "layer_tokens": {},
            "original_prefill_seq_len": 0,
            "decoder_layers_recorded": 0,
            "decoder_layers_missing": 0,
            "final_visual_tokens": 0,
            "raw_total_budget": float(self.config.keep_visual_tokens),
            "active_pruning_layers": 1.0,
            "flops_reduction": 0.0,
            "attention_mask_gathered": 0,
            "attention_mask_dropped": 0,
            "cache_layer0_seq_len": 0,
            "cache_min_seq_len": 0,
            "cache_max_seq_len": 0,
            "cache_unique_seq_lens": 0,
            "vlapruner_history_size": int(len(self.action_attention_history)),
            "vlapruner_history_ready": int(self._history_ready()),
            "vlapruner_decode_vectors_recorded": 0,
            "vlapruner_action_vectors_used": 0,
            "vlapruner_semantic_score_mass": 0.0,
            "vlapruner_temporal_score_mass": 0.0,
            "vlapruner_unified_visual_pool": 1,
            "vlapruner_requested_keep": int(self.config.keep_visual_tokens),
            "vlapruner_adjusted_keep": int(self.config.keep_visual_tokens),
            "vlapruner_average_target_steps": 0.0,
            "vlapruner_measured_warmup_calls": 0.0,
        }

    @staticmethod
    def _split_contiguous_positions(positions: torch.Tensor) -> List[torch.Tensor]:
        if positions.numel() == 0:
            return []
        if positions.numel() == 1:
            return [positions]
        splits = torch.nonzero(positions[1:] != (positions[:-1] + 1), as_tuple=False).flatten() + 1
        return [chunk for chunk in torch.tensor_split(positions, splits.tolist()) if chunk.numel() > 0]

    def prepare_input_layout(
        self,
        input_ids: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
        num_views: Optional[int] = None,
        dual_view_mode: str = "separate",
        spatial_merge_size: int = 2,
    ) -> None:
        self.input_layout = {}
        if input_ids is None or image_token_id is None or input_ids.ndim != 2 or input_ids.shape[0] < 1:
            return

        row = input_ids[0]
        if attention_mask is not None:
            mask_row = attention_mask[0].to(dtype=torch.bool)
            valid_positions = torch.nonzero(mask_row, as_tuple=False).flatten()
            row = row[mask_row]
        else:
            valid_positions = torch.arange(row.shape[0], device=row.device)

        image_mask = row == int(image_token_id)
        if image_mask.sum().item() <= 0:
            return

        image_positions = valid_positions[image_mask]
        image_groups = self._split_contiguous_positions(image_positions)
        requested_views = max(1, int(num_views or self.config.num_views))

        if len(image_groups) == 1 and requested_views > 1 and dual_view_mode == "stitched":
            stitched_positions = image_groups[0]
            split_sizes = []
            remaining = int(stitched_positions.numel())
            for view_idx in range(requested_views):
                views_left = requested_views - view_idx
                take = remaining // views_left
                if remaining % views_left:
                    take += 1
                split_sizes.append(max(0, take))
                remaining -= take
            image_groups = []
            offset = 0
            for split_size in split_sizes:
                image_groups.append(stitched_positions[offset : offset + split_size])
                offset += split_size
        elif len(image_groups) > requested_views:
            image_groups = image_groups[:requested_views]

        image_groups = [group for group in image_groups if group.numel() > 0]
        if not image_groups:
            return

        expected_counts: List[int] = []
        if image_grid_thw is not None and image_grid_thw.numel() > 0:
            merge_area = max(1, int(spatial_merge_size) ** 2)
            expected_counts = [
                int(((grid[0] * grid[1] * grid[2]).item()) // merge_area)
                for grid in image_grid_thw
            ]

        all_visual_indices = torch.cat(image_groups, dim=0).contiguous()
        last_visual_position = int(all_visual_indices.max().item())
        non_image_positions = valid_positions[~image_mask]
        query_positions = non_image_positions[non_image_positions > last_visual_position].contiguous()

        self.input_layout = {
            "ready": True,
            "view_indices": [group.contiguous() for group in image_groups],
            "all_visual_indices": all_visual_indices,
            "query_positions": query_positions,
            "expected_counts": expected_counts,
            "dual_view_mode": dual_view_mode,
        }
        self.runtime_stats["layout_ready"] = 1
        self.runtime_stats["layout_query_tokens"] = int(query_positions.numel())
        self.runtime_stats["layout_view0_tokens"] = int(image_groups[0].numel()) if len(image_groups) > 0 else 0
        self.runtime_stats["layout_view1_tokens"] = int(image_groups[1].numel()) if len(image_groups) > 1 else 0

    def _locate_current_positions(self, original_positions: torch.Tensor) -> torch.Tensor:
        if self.global_indices is None or original_positions is None or original_positions.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        current_globals = self.global_indices[0]
        original_positions = original_positions.to(device=current_globals.device, dtype=current_globals.dtype)
        mask = torch.isin(current_globals, original_positions)
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _resolve_layout_state(self, seq_len: int, device: torch.device) -> Dict[str, Any]:
        if self.input_layout.get("ready"):
            query_positions = self._locate_current_positions(self.input_layout["query_positions"]).to(device=device)
            view_positions = [
                self._locate_current_positions(view_indices).to(device=device)
                for view_indices in self.input_layout["view_indices"]
            ]
            all_visual_indices = self.input_layout["all_visual_indices"]
            current_visual_positions = self._locate_current_positions(all_visual_indices).to(device=device)
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
            if current_visual_positions.numel() > 0:
                keep_mask[current_visual_positions] = False
            always_keep_positions = torch.nonzero(keep_mask, as_tuple=False).flatten()
            return {
                "query_positions": query_positions,
                "view_positions": view_positions,
                "visual_positions": current_visual_positions,
                "always_keep_positions": always_keep_positions,
            }

        empty = torch.empty(0, dtype=torch.long, device=device)
        return {
            "query_positions": empty,
            "view_positions": [empty, empty],
            "visual_positions": empty,
            "always_keep_positions": torch.arange(seq_len, device=device, dtype=torch.long),
        }

    def _history_ready(self) -> bool:
        return len(self.action_attention_history) >= max(1, int(self.config.warmup_steps))

    def _temporal_action_scores(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self._history_ready():
            return None
        hist = list(self.action_attention_history)
        gamma = float(self.config.temporal_gamma)
        weights = torch.tensor([gamma ** i for i in range(len(hist) - 1, -1, -1)], device=device, dtype=torch.float32)
        stacked = torch.stack([x.to(device=device, dtype=torch.float32) for x in hist], dim=0)
        scores = (stacked * weights[:, None]).sum(dim=0) / weights.sum().clamp(min=1e-6)
        return scores.to(dtype=dtype)

    def _semantic_visual_scores(
        self,
        attn_weights: torch.Tensor,
        visual_positions: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Paper Step 3: current semantic-level visual token importance."""
        if visual_positions.numel() == 0:
            return torch.zeros((attn_weights.shape[0], 0), device=attn_weights.device, dtype=attn_weights.dtype)
        if query_positions.numel() == 0:
            query_mask = torch.ones(attn_weights.shape[-1], dtype=torch.bool, device=attn_weights.device)
            query_mask[visual_positions] = False
            query_positions = torch.nonzero(query_mask, as_tuple=False).flatten()
        rel_attn = attn_weights.index_select(2, query_positions).index_select(3, visual_positions)
        return rel_attn.mean(dim=1).mean(dim=1)

    def _record_decode_action_attention(
        self,
        layer_idx: int,
        attn_weights: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> None:
        """Paper Step 4 support: collect decode-step action-to-vision attention."""
        if attn_weights is None or layer_idx < self.action_layer_start_idx:
            return
        if self.global_indices is None or not self.input_layout.get("ready"):
            return
        visual_original = self.input_layout.get("all_visual_indices")
        if visual_original is None or visual_original.numel() == 0:
            return

        current_globals = self.global_indices[0].to(device=device)
        visual_original = visual_original.to(device=device, dtype=current_globals.dtype)
        key_mask = torch.isin(current_globals, visual_original)
        visual_key_positions = torch.nonzero(key_mask, as_tuple=False).flatten()
        if visual_key_positions.numel() == 0:
            return

        # In HF autoregressive generation, the short decode forward uses the
        # current generated token as query and the KV cache as visual keys.
        query_positions = torch.arange(seq_len, device=device, dtype=torch.long)
        rel_attn = attn_weights.index_select(2, query_positions).index_select(3, visual_key_positions)
        current_scores = rel_attn.float().mean(dim=(0, 1, 2))

        visual_ids = current_globals.index_select(0, visual_key_positions)
        full_scores = torch.zeros(int(self.config.visual_tokens), device=device, dtype=torch.float32)
        order = torch.argsort(visual_original)
        sorted_visual = visual_original.index_select(0, order)
        mapped = torch.searchsorted(sorted_visual, visual_ids)
        valid = mapped < sorted_visual.numel()
        if torch.any(valid):
            matched = sorted_visual.index_select(0, mapped[valid])
            valid_indices = valid.nonzero(as_tuple=False).flatten()
            valid[valid_indices] = matched == visual_ids.index_select(0, valid_indices)
        if not torch.any(valid):
            return
        local_ids = order.index_select(0, mapped[valid])
        full_scores.scatter_add_(0, local_ids, current_scores[valid])
        self._current_decode_layer_vectors.append(full_scores.detach())

    def begin_generate(self) -> None:
        call_idx = int(self._episode_policy_calls)
        self._current_call_is_measured = call_idx >= max(0, int(self.config.measured_skip_initial_calls))
        self._current_call_measured_seen_before = int(self._episode_measured_calls)
        if self._current_call_is_measured:
            self._episode_measured_calls += 1
        self._episode_policy_calls += 1
        self._decode_action_vectors_by_step = []
        self._current_decode_layer_vectors = []

    def finish_decode_step(self) -> None:
        """Average action-to-vision attention over latter-half layers for one decode query token."""
        if not self._current_decode_layer_vectors:
            return
        vec = torch.stack(self._current_decode_layer_vectors, dim=0).mean(dim=0)
        self._decode_action_vectors_by_step.append(vec.detach().cpu())
        self._current_decode_layer_vectors = []
        self.runtime_stats["vlapruner_decode_vectors_recorded"] = int(len(self._decode_action_vectors_by_step))

    def finalize_action_attention_from_generation(self, generated_ids: torch.Tensor, prompt_token_count: int) -> None:
        """Build S_act^t for the completed Nora action chunk and append it to history.

        Nora-LONG/AC generates a variable number of FAST+ tokens for a chunk of
        five future actions.  The paper's autoregressive case averages attention
        vectors from decoding steps; here we filter those recorded decode-query
        vectors to Nora action-token ids and average them into one S_act^t.
        """
        if generated_ids is None or generated_ids.ndim != 2 or generated_ids.shape[0] < 1:
            return
        gen = generated_ids[0, int(prompt_token_count) :]
        if gen.numel() == 0 or not self._decode_action_vectors_by_step:
            return

        action_mask = (gen >= int(self.config.action_token_min)) & (gen <= int(self.config.action_token_max))
        # The first generated token is produced by prefill; decode vectors start
        # from the next autoregressive query, so align vector i with gen[i].
        usable = min(len(self._decode_action_vectors_by_step), int(gen.numel()))
        chosen: List[torch.Tensor] = []
        for idx in range(usable):
            if bool(action_mask[idx].item()):
                chosen.append(self._decode_action_vectors_by_step[idx])
        if not chosen:
            return
        vec = torch.stack(chosen, dim=0).mean(dim=0)
        self.action_attention_history.append(vec.detach().cpu())
        self.runtime_stats["vlapruner_action_vectors_used"] = int(len(chosen))
        self.runtime_stats["vlapruner_history_size"] = int(len(self.action_attention_history))
        self.runtime_stats["vlapruner_history_ready"] = int(self._history_ready())

    def _resolve_average_calibrated_keep_count(self, curr_vis_len: int) -> int:
        """Convert requested average keep into a fixed post-warmup keep.

        VLA-Pruner has an unavoidable temporal warm-start: early control steps
        keep all visual tokens while building S_act history.  The paper reports
        a target token budget, and our LIBERO logger averages final visual token
        counts over measured policy calls.  To make the logged average target
        match keep_visual_tokens, we solve:

            target * N = full * W + post_keep * (N - W)

        where W is the number of already-measured warm-start calls before the
        first pruned call, and N is an episode-level measured-call estimate.
        The solved post_keep is fixed for the rest of the episode, avoiding an
        unstable "first few pruned calls are much smaller, later calls revert"
        schedule.
        """
        target = max(1, min(int(self.config.keep_visual_tokens), int(curr_vis_len)))
        if not bool(getattr(self.config, "calibrate_average_keep", True)):
            return target
        if self._episode_adjusted_keep_count is not None:
            return max(1, min(int(self._episode_adjusted_keep_count), int(curr_vis_len)))

        warmup_measured = max(0, int(self._current_call_measured_seen_before))
        if warmup_measured <= 0:
            adjusted = target
            estimated_total = float(max(1, int(getattr(self.config, "average_target_measured_steps", 40) or 40)))
        else:
            if self._episode_estimated_measured_calls is not None:
                estimated_total = float(self._episode_estimated_measured_calls)
            else:
                estimated_total = float(max(1, int(getattr(self.config, "average_target_measured_steps", 40) or 40)))
            estimated_total = max(float(warmup_measured + 1), estimated_total)
            pruned_calls = max(1.0, estimated_total - float(warmup_measured))
            adjusted_float = (
                float(target) * estimated_total - float(curr_vis_len) * float(warmup_measured)
            ) / pruned_calls
            adjusted = int(round(adjusted_float))
            adjusted = min(target, adjusted)

        adjusted = max(1, min(int(adjusted), int(curr_vis_len)))
        self._episode_adjusted_keep_count = adjusted
        self.runtime_stats["vlapruner_requested_keep"] = int(target)
        self.runtime_stats["vlapruner_adjusted_keep"] = int(adjusted)
        self.runtime_stats["vlapruner_average_target_steps"] = float(estimated_total)
        self.runtime_stats["vlapruner_measured_warmup_calls"] = float(warmup_measured)
        return adjusted

    def _select_vlapruner_indices(
        self,
        hidden_states: torch.Tensor,
        visual_positions: torch.Tensor,
        semantic_scores: torch.Tensor,
        temporal_scores: torch.Tensor,
        keep_count: int,
    ) -> torch.Tensor:
        """Paper Steps 5-7: dual top-k, union, then greedy redundancy filtering."""
        total_visual = int(semantic_scores.shape[-1])
        keep_count = max(1, min(int(keep_count), total_visual))

        # Step 5: select semantically important tokens and action-important tokens.
        sem_topk = torch.topk(semantic_scores[0], k=keep_count, dim=-1).indices
        act_topk = torch.topk(temporal_scores[:total_visual], k=keep_count, dim=-1).indices

        # Step 6: maximize relevance by combining both candidate sets.
        candidate_local = torch.unique(torch.cat([sem_topk, act_topk], dim=0))

        # Step 7: remove redundancy by greedy Max-Min filtering over candidate features.
        if candidate_local.numel() > keep_count:
            visual_hidden = hidden_states[0, visual_positions, :]
            candidate_features = visual_hidden.index_select(0, candidate_local)
            chosen_in_candidate = _max_min_redundancy_filter(candidate_features, keep_count)
            candidate_local = candidate_local.index_select(0, chosen_in_candidate)
        candidate_local, _ = candidate_local.sort()
        return candidate_local

    @staticmethod
    def _build_full_indices(
        batch_size: int,
        seq_len: int,
        always_keep_positions: torch.Tensor,
        selected_visual_positions: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        keep_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        if always_keep_positions.numel() > 0:
            keep_mask[:, always_keep_positions] = True
        if selected_visual_positions.numel() > 0:
            keep_mask.scatter_(1, selected_visual_positions, True)
        base = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return base[keep_mask].view(batch_size, -1)

    def _count_selected_tokens_by_original_view(self, representative_positions: torch.Tensor) -> List[int]:
        if not self.input_layout.get("ready") or representative_positions.numel() == 0:
            return [0, 0]
        first_view = self.input_layout["view_indices"][0]
        first_view_set = first_view.to(device=representative_positions.device)
        view0 = int(torch.isin(representative_positions.long(), first_view_set.long()).sum().item())
        view1 = int(representative_positions.numel() - view0)
        return [view0, view1]

    def _record_prefill_layer_token_count(self, layer_idx: int, seq_len: int) -> None:
        if layer_idx not in self.runtime_stats["layer_tokens"]:
            self.runtime_stats["layer_tokens"][layer_idx] = int(seq_len)
        self._finalize_decoder_flops_stats()

    def get_flops_reduction(self) -> float:
        layer_tokens = self.runtime_stats.get("layer_tokens", {})
        original = int(self.runtime_stats.get("original_prefill_seq_len", 0) or 0)
        if not layer_tokens or original <= 0 or self.num_layers <= 0:
            return 0.0
        total_original = original * self.num_layers
        total_actual = sum(int(v) for v in layer_tokens.values())
        if total_original <= 0:
            return 0.0
        return 1.0 - (float(total_actual) / float(total_original))

    def _finalize_decoder_flops_stats(self) -> None:
        layer_tokens = self.runtime_stats.get("layer_tokens", {})
        if not isinstance(layer_tokens, dict):
            layer_tokens = {}
            self.runtime_stats["layer_tokens"] = layer_tokens
        self.runtime_stats["decoder_layers_recorded"] = int(len(layer_tokens))
        self.runtime_stats["decoder_layers_missing"] = max(0, int(self.num_layers) - int(len(layer_tokens)))
        self.runtime_stats["flops_reduction"] = float(self.get_flops_reduction())

    def _record_cache_lengths(self, past_key_value: Any) -> None:
        key_cache = getattr(past_key_value, "key_cache", None)
        if not isinstance(key_cache, list) or not key_cache:
            return
        lengths: List[int] = []
        for cache_tensor in key_cache:
            if torch.is_tensor(cache_tensor) and cache_tensor.numel() > 0 and cache_tensor.dim() >= 3:
                lengths.append(int(cache_tensor.shape[-2]))
        if not lengths:
            return
        self.runtime_stats["cache_layer0_seq_len"] = int(lengths[0])
        self.runtime_stats["cache_min_seq_len"] = int(min(lengths))
        self.runtime_stats["cache_max_seq_len"] = int(max(lengths))
        self.runtime_stats["cache_unique_seq_lens"] = int(len(set(lengths)))

    def _run_profiled_original_forward(
        self,
        layer_idx: int,
        self_layer: Any,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        forward_kwargs: Dict[str, Any],
        stat_prefix: str,
    ):
        """Use AutoPrune-compatible timing fields for fair latency reporting."""
        component_prefix = stat_prefix[:-8] if stat_prefix.endswith("_forward") else stat_prefix
        self_attn = getattr(self_layer, "self_attn", None)
        mlp = getattr(self_layer, "mlp", None)
        orig_self_attn_forward = getattr(self_attn, "forward", None) if self_attn is not None else None
        orig_mlp_forward = getattr(mlp, "forward", None) if mlp is not None else None

        timed_self_attn_seconds = 0.0
        timed_mlp_seconds = 0.0

        if orig_self_attn_forward is not None:
            def timed_self_attn_forward(module_self, *args, **kwargs):
                nonlocal timed_self_attn_seconds
                t0 = time.perf_counter()
                try:
                    return orig_self_attn_forward(*args, **kwargs)
                finally:
                    timed_self_attn_seconds += time.perf_counter() - t0

            self_attn.forward = types.MethodType(timed_self_attn_forward, self_attn)

        if orig_mlp_forward is not None:
            def timed_mlp_forward(module_self, *args, **kwargs):
                nonlocal timed_mlp_seconds
                t0 = time.perf_counter()
                try:
                    return orig_mlp_forward(*args, **kwargs)
                finally:
                    timed_mlp_seconds += time.perf_counter() - t0

            mlp.forward = types.MethodType(timed_mlp_forward, mlp)

        forward_t0 = time.perf_counter()
        try:
            outputs = self.original_forwards[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **forward_kwargs,
            )
        finally:
            if orig_self_attn_forward is not None:
                self_attn.forward = orig_self_attn_forward
            if orig_mlp_forward is not None:
                mlp.forward = orig_mlp_forward

        total_forward_seconds = time.perf_counter() - forward_t0
        residual_norm_seconds = max(0.0, total_forward_seconds - timed_self_attn_seconds - timed_mlp_seconds)
        self.runtime_stats[f"{stat_prefix}_seconds"] = float(
            self.runtime_stats.get(f"{stat_prefix}_seconds", 0.0) + total_forward_seconds
        )
        self.runtime_stats[f"{component_prefix}_self_attn_seconds"] = float(
            self.runtime_stats.get(f"{component_prefix}_self_attn_seconds", 0.0) + timed_self_attn_seconds
        )
        self.runtime_stats[f"{component_prefix}_mlp_seconds"] = float(
            self.runtime_stats.get(f"{component_prefix}_mlp_seconds", 0.0) + timed_mlp_seconds
        )
        self.runtime_stats[f"{component_prefix}_residual_norm_seconds"] = float(
            self.runtime_stats.get(f"{component_prefix}_residual_norm_seconds", 0.0) + residual_norm_seconds
        )
        return outputs

    def _align_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4 and attention_mask.shape[-2] == seq_len and attention_mask.shape[-1] == seq_len:
            return attention_mask
        if attention_mask.dim() == 2 and attention_mask.shape[-1] == seq_len:
            return attention_mask
        if self.global_indices is not None:
            gather_idx = self.global_indices.to(device=device, dtype=torch.long)
            if gather_idx.shape[0] == 1 and batch_size > 1:
                gather_idx = gather_idx.expand(batch_size, -1)
            max_idx = int(gather_idx.max().item()) if gather_idx.numel() > 0 else -1
            try:
                if (
                    attention_mask.dim() == 4
                    and attention_mask.shape[-2] > max_idx
                    and attention_mask.shape[-1] > max_idx
                ):
                    if attention_mask.shape[0] == 1 and batch_size > 1:
                        attention_mask = attention_mask.expand(batch_size, -1, -1, -1)
                    heads = attention_mask.shape[1]
                    q_idx = gather_idx.view(batch_size, 1, seq_len, 1).expand(
                        batch_size, heads, seq_len, attention_mask.shape[-1]
                    )
                    gathered = attention_mask.gather(2, q_idx)
                    k_idx = gather_idx.view(batch_size, 1, 1, seq_len).expand(batch_size, heads, seq_len, seq_len)
                    gathered = gathered.gather(3, k_idx).contiguous()
                    self.runtime_stats["attention_mask_gathered"] = int(
                        self.runtime_stats.get("attention_mask_gathered", 0) + 1
                    )
                    return gathered
                if attention_mask.dim() == 2 and attention_mask.shape[-1] > max_idx:
                    gathered = attention_mask.gather(1, gather_idx).contiguous()
                    self.runtime_stats["attention_mask_gathered"] = int(
                        self.runtime_stats.get("attention_mask_gathered", 0) + 1
                    )
                    return gathered
            except Exception:
                pass
        self.runtime_stats["attention_mask_dropped"] = int(self.runtime_stats.get("attention_mask_dropped", 0) + 1)
        return None

    def _make_new_forward(self, layer_idx: int):
        def new_forward(self_layer, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            patch_t0 = time.perf_counter()

            def finish(result):
                self.runtime_stats["patch_total_seconds"] = float(
                    self.runtime_stats.get("patch_total_seconds", 0.0) + (time.perf_counter() - patch_t0)
                )
                self._finalize_decoder_flops_stats()
                return result

            batch_size, seq_len, _ = hidden_states.shape
            original_output_attentions = bool(kwargs.get("output_attentions", False))
            original_use_cache = bool(kwargs.get("use_cache", False))

            if layer_idx == 0:
                if self._current_decode_layer_vectors:
                    self.finish_decode_step()
                self.runtime_stats["layer0_total_calls"] = int(self.runtime_stats.get("layer0_total_calls", 0) + 1)
                if seq_len <= int(self.config.decode_skip_seq_len):
                    self.runtime_stats["layer0_short_seq_calls"] = int(self.runtime_stats.get("layer0_short_seq_calls", 0) + 1)
                else:
                    self.runtime_stats["layer0_long_seq_calls"] = int(self.runtime_stats.get("layer0_long_seq_calls", 0) + 1)
                if kwargs.get("past_key_value", None) is None:
                    self.runtime_stats["layer0_past_none_calls"] = int(self.runtime_stats.get("layer0_past_none_calls", 0) + 1)
                else:
                    self.runtime_stats["layer0_past_present_calls"] = int(self.runtime_stats.get("layer0_past_present_calls", 0) + 1)
                if self.global_indices is None or seq_len > int(self.config.decode_skip_seq_len):
                    self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
                if self.runtime_stats.get("original_prefill_seq_len", 0) <= 0 and seq_len > int(self.config.decode_skip_seq_len):
                    self.runtime_stats["original_prefill_seq_len"] = int(seq_len)
                    self._has_pruned_this_forward = False

            short_seq = seq_len <= int(self.config.decode_skip_seq_len)
            if (not short_seq) and (layer_idx == 0 or self.global_indices is None):
                self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

            if position_ids is None and self.global_indices is not None and not short_seq:
                position_ids = self.global_indices
            elif position_ids is not None and self.global_indices is not None and not short_seq:
                gather_idx = self.global_indices
                if gather_idx.shape[0] == 1 and batch_size > 1:
                    gather_idx = gather_idx.expand(batch_size, -1)
                if position_ids.dim() == 2 and position_ids.shape[1] > seq_len:
                    position_ids = position_ids.gather(1, gather_idx)
                elif position_ids.dim() == 3 and position_ids.shape[2] > seq_len:
                    expanded = gather_idx.unsqueeze(0).expand(position_ids.shape[0], -1, -1)
                    position_ids = position_ids.gather(2, expanded)

            attention_mask = self._align_attention_mask(
                attention_mask=attention_mask,
                seq_len=seq_len,
                batch_size=batch_size,
                device=hidden_states.device,
            )
            self._record_cache_lengths(kwargs.get("past_key_value", None))

            if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None and self.global_indices is not None and not short_seq:
                try:
                    cos, sin = kwargs["position_embeddings"]
                    gather_idx = self.global_indices
                    if gather_idx.shape[0] == 1 and batch_size > 1:
                        gather_idx = gather_idx.expand(batch_size, -1)
                    if cos.dim() == 4 and cos.shape[2] > seq_len:
                        expanded = gather_idx.unsqueeze(0).unsqueeze(-1).expand(
                            cos.shape[0], batch_size, gather_idx.shape[1], cos.size(-1)
                        )
                        kwargs["position_embeddings"] = (cos.gather(2, expanded), sin.gather(2, expanded))
                    elif cos.dim() == 3 and cos.shape[1] > seq_len:
                        kwargs["position_embeddings"] = (
                            cos.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, cos.size(-1))),
                            sin.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, sin.size(-1))),
                        )
                except Exception:
                    pass

            # Paper Step 2: early Transformer layers run normally.  We force
            # attentions only at the VLA-Pruner layer for semantic scoring and
            # at deeper decode layers for action-to-vision history recording.
            need_prefill_attn = (not short_seq) and (layer_idx == self.pruning_layer_idx)
            need_decode_attn = short_seq and layer_idx >= self.action_layer_start_idx
            need_attn = bool(need_prefill_attn or need_decode_attn)
            forward_kwargs = dict(kwargs)
            if need_attn and not original_output_attentions:
                forward_kwargs["output_attentions"] = True

            outputs = self._run_profiled_original_forward(
                layer_idx=layer_idx,
                self_layer=self_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                forward_kwargs=forward_kwargs,
                stat_prefix="force_attn" if need_attn else "regular_forward",
            )

            def clean_outputs(current_outputs, hidden_override=None):
                if hidden_override is not None:
                    if isinstance(current_outputs, tuple):
                        current_outputs = (hidden_override,) + current_outputs[1:]
                    else:
                        current_outputs = hidden_override
                if original_output_attentions or not need_attn:
                    return current_outputs
                if not isinstance(current_outputs, tuple):
                    return current_outputs
                if original_use_cache and len(current_outputs) >= 3:
                    return (current_outputs[0], current_outputs[2])
                return (current_outputs[0],) + current_outputs[2:]

            attn_weights = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
            current_layer_output = outputs[0] if isinstance(outputs, tuple) else outputs

            if short_seq:
                self._record_decode_action_attention(layer_idx, attn_weights, seq_len, hidden_states.device)
                return finish(clean_outputs(outputs))

            self._record_prefill_layer_token_count(layer_idx, seq_len)
            if layer_idx != self.pruning_layer_idx or self._has_pruned_this_forward or attn_weights is None:
                return finish(clean_outputs(outputs))

            layout = self._resolve_layout_state(seq_len, hidden_states.device)
            query_positions = layout["query_positions"]
            visual_positions = layout["visual_positions"]
            always_keep_positions = layout["always_keep_positions"]
            if visual_positions.numel() <= 0:
                return finish(clean_outputs(outputs))

            curr_vis_len = int(visual_positions.numel())
            self.runtime_stats["last_tokens_before"] = int(curr_vis_len)

            temporal_scores = self._temporal_action_scores(hidden_states.device, attn_weights.dtype)
            # Paper warm-start: do not prune until w previous control steps have
            # produced action-chunk attention histories.
            if temporal_scores is None:
                self.runtime_stats["last_tokens_after"] = int(curr_vis_len)
                self.runtime_stats["final_visual_tokens"] = int(curr_vis_len)
                self.runtime_stats["raw_total_budget"] = float(curr_vis_len)
                return finish(clean_outputs(outputs))

            keep_count = self._resolve_average_calibrated_keep_count(curr_vis_len)
            if keep_count >= curr_vis_len:
                self.runtime_stats["last_tokens_after"] = int(curr_vis_len)
                self.runtime_stats["final_visual_tokens"] = int(curr_vis_len)
                self.runtime_stats["raw_total_budget"] = float(curr_vis_len)
                return finish(clean_outputs(outputs))

            score_t0 = time.perf_counter()
            semantic_scores = self._semantic_visual_scores(attn_weights, visual_positions, query_positions)
            keep_local = self._select_vlapruner_indices(
                hidden_states=current_layer_output,
                visual_positions=visual_positions,
                semantic_scores=semantic_scores,
                temporal_scores=temporal_scores,
                keep_count=keep_count,
            )
            selected_visual_positions = visual_positions.unsqueeze(0).expand(batch_size, -1).gather(
                1,
                keep_local.unsqueeze(0).expand(batch_size, -1),
            )
            self.runtime_stats["topk_seconds"] = float(
                self.runtime_stats.get("topk_seconds", 0.0) + (time.perf_counter() - score_t0)
            )

            # Paper Step 8: drop unselected visual tokens physically; later
            # prefill layers process the shortened sequence and KV cache records
            # only the retained unified visual pool.
            gather_t0 = time.perf_counter()
            full_indices = self._build_full_indices(
                batch_size=batch_size,
                seq_len=seq_len,
                always_keep_positions=always_keep_positions,
                selected_visual_positions=selected_visual_positions,
                device=hidden_states.device,
            )
            expanded = full_indices.unsqueeze(-1).expand(-1, -1, current_layer_output.size(-1))
            pruned_hidden = current_layer_output.gather(1, expanded).contiguous()
            current_globals = self.global_indices
            if current_globals is None:
                current_globals = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            elif current_globals.shape[0] == 1 and batch_size > 1:
                current_globals = current_globals.expand(batch_size, -1)
            self.global_indices = current_globals.gather(1, full_indices)
            self._has_pruned_this_forward = True
            self.runtime_stats["gather_seconds"] = float(
                self.runtime_stats.get("gather_seconds", 0.0) + (time.perf_counter() - gather_t0)
            )
            self.runtime_stats["last_tokens_after"] = int(keep_count)
            self.runtime_stats["pruned_layers"] = 1
            self.runtime_stats["raw_total_budget"] = float(keep_count)
            self.runtime_stats["final_visual_tokens"] = int(keep_count)
            self.runtime_stats["active_pruning_layers"] = 1.0
            self.runtime_stats["vlapruner_semantic_score_mass"] = float(
                semantic_scores.sum().detach().float().cpu().item()
            )
            self.runtime_stats["vlapruner_temporal_score_mass"] = float(
                temporal_scores.sum().detach().float().cpu().item()
            )
            selected_globals = self.global_indices[0][
                torch.isin(
                    self.global_indices[0],
                    self.input_layout.get("all_visual_indices", torch.empty(0, device=self.global_indices.device)),
                )
            ]
            self.runtime_stats["view_keep_counts"] = self._count_selected_tokens_by_original_view(selected_globals)
            return finish(clean_outputs(outputs, hidden_override=pruned_hidden))

        return new_forward
