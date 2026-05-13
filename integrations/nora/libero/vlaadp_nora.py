from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class NoraVLAADPConfig:
    enabled: bool = False
    keep_visual_tokens: int = 224
    visual_tokens: int = 512
    num_views: int = 2
    qk_layer: int = 0
    main_view_weight: float = 0.4
    wrist_view_weight: float = 0.6
    chunk_size: int = 5
    cold_start_windows: int = 2
    decision_method: str = "adjacent"
    delta_method: str = "net"
    adjacent_extrema_window: int = 3
    adjacent_last_state: bool = False
    limit_consecutive_pruned_enabled: bool = True
    limit_max_consecutive_pruned: int = 3
    min_keep_per_view: int = 1
    debug: bool = False


def _get_text_embedding_layer(model: Any) -> torch.nn.Module:
    candidates: List[Any] = [model]
    for attr in ("base_model", "model"):
        child = getattr(model, attr, None)
        if child is not None and child not in candidates:
            candidates.append(child)
    for root in list(candidates):
        for attr in ("model", "language_model"):
            child = getattr(root, attr, None)
            if child is not None and child not in candidates:
                candidates.append(child)

    for candidate in candidates:
        get_input_embeddings = getattr(candidate, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            embedding = get_input_embeddings()
            if embedding is not None:
                return embedding
        embedding = getattr(candidate, "embed_tokens", None)
        if embedding is not None:
            return embedding
        language_model = getattr(candidate, "language_model", None)
        embedding = getattr(language_model, "embed_tokens", None) if language_model is not None else None
        if embedding is not None:
            return embedding

    raise AttributeError("Could not locate the NORA/Qwen text embedding layer.")


class NoraVLAADPPatch:
    """VLA-ADP adapted to Nora/Qwen2.5-VL.

    The patch applies ADP at the embedding/prefill boundary, before the LLM
    stack.  It intentionally does not replace decoder layer forwards: when the
    gate is open, all LLM layers see the shortened visual sequence; when the gate
    is closed, the original multimodal input is preserved.
    """

    def __init__(self, model: Any, config: NoraVLAADPConfig):
        self.model = model
        self.config = config
        self.layers = self._resolve_layers(model)
        self.num_layers = len(self.layers)
        self.input_layout: Dict[str, Any] = {}
        self.runtime_stats: Dict[str, Any] = {}
        self.current_state = 0
        self.delta_history: List[float] = []
        self.completed_windows = 0
        self.consecutive_pruned = 0
        self._current_forward_state = 0
        self._last_pose_matrix: Optional[np.ndarray] = None
        self.reset_runtime_state()

    @staticmethod
    def _resolve_layers(model: Any):
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
        raise RuntimeError("Unable to locate decoder layers for VLA-ADP in the provided NORA model.")

    def apply(self) -> None:
        return None

    def reset_episode_history(self) -> None:
        self.current_state = 0
        self.delta_history = []
        self.completed_windows = 0
        self.consecutive_pruned = 0
        self._current_forward_state = 0
        self._last_pose_matrix = None
        self.reset_runtime_state()

    def reset_runtime_state(self) -> None:
        self.input_layout = {}
        self._current_forward_state = self._state_for_next_forward()
        self.runtime_stats = {
            "patch_total_seconds": 0.0,
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
            "layer_tokens": {},
            "original_prefill_seq_len": 0,
            "decoder_layers_recorded": 0,
            "decoder_layers_missing": 0,
            "final_visual_tokens": 0,
            "raw_total_budget": 0.0,
            "active_pruning_layers": 0.0,
            "flops_reduction": 0.0,
            "prefill_compressed": 0,
            "prefill_original_visual_tokens": 0,
            "view_keep_counts": [0, 0],
            "vlaadp_state": int(self._current_forward_state),
            "vlaadp_completed_windows": int(self.completed_windows),
            "vlaadp_delta_history_size": int(len(self.delta_history)),
            "vlaadp_consecutive_pruned": int(self.consecutive_pruned),
            "vlaadp_cold_start_active": int(self.completed_windows < int(self.config.cold_start_windows)),
            "vlaadp_virtual_windows_updated": 0,
            "vlaadp_last_delta": 0.0,
        }

    def _state_for_next_forward(self) -> int:
        # Step 7/8: decide whether the current forward should use full vision or pruning.
        if self.completed_windows < int(self.config.cold_start_windows):
            return 0
        return int(self.current_state)

    def begin_generate(self) -> None:
        self._current_forward_state = self._state_for_next_forward()
        self.runtime_stats["vlaadp_state"] = int(self._current_forward_state)
        self.runtime_stats["vlaadp_cold_start_active"] = int(
            self.completed_windows < int(self.config.cold_start_windows)
        )

    def _record_full_prefill_stats(self, generate_kwargs: Dict[str, Any]) -> None:
        input_ids = generate_kwargs.get("input_ids", None)
        if input_ids is None or getattr(input_ids, "ndim", 0) != 2:
            return
        visual_tokens = 0
        if self.input_layout.get("ready"):
            visual_tokens = int(self.input_layout.get("all_visual_indices").numel())
        self.runtime_stats["last_tokens_before"] = visual_tokens
        self.runtime_stats["last_tokens_after"] = visual_tokens
        self.runtime_stats["prefill_original_visual_tokens"] = visual_tokens
        self.runtime_stats["final_visual_tokens"] = visual_tokens
        self.runtime_stats["raw_total_budget"] = float(visual_tokens)
        self.runtime_stats["original_prefill_seq_len"] = int(input_ids.shape[1])
        self.runtime_stats["layer_tokens"] = {idx: int(input_ids.shape[1]) for idx in range(self.num_layers)}
        self.runtime_stats["decoder_layers_recorded"] = int(self.num_layers)
        self.runtime_stats["decoder_layers_missing"] = 0
        self.runtime_stats["pruned_layers"] = 0
        self.runtime_stats["active_pruning_layers"] = 0.0
        self.runtime_stats["flops_reduction"] = 0.0

    def should_compress_generate_inputs(self, generate_kwargs: Dict[str, Any]) -> bool:
        if not self.config.enabled:
            return False
        if int(self._current_forward_state) != 1:
            self._record_full_prefill_stats(generate_kwargs)
            return False
        if generate_kwargs.get("inputs_embeds", None) is not None:
            self._record_full_prefill_stats(generate_kwargs)
            return False
        if generate_kwargs.get("pixel_values", None) is None:
            self._record_full_prefill_stats(generate_kwargs)
            return False
        if generate_kwargs.get("image_grid_thw", None) is None:
            self._record_full_prefill_stats(generate_kwargs)
            return False
        input_ids = generate_kwargs.get("input_ids", None)
        ok = input_ids is not None and getattr(input_ids, "ndim", 0) == 2
        if not ok:
            self._record_full_prefill_stats(generate_kwargs)
        return ok

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

        image_groups = [group.contiguous() for group in image_groups if group.numel() > 0]
        if not image_groups:
            return

        all_visual_indices = torch.cat(image_groups, dim=0).contiguous()
        last_visual_position = int(all_visual_indices.max().item())
        non_image_positions = valid_positions[~image_mask]
        text_positions = non_image_positions[non_image_positions > last_visual_position].contiguous()
        if text_positions.numel() == 0:
            text_positions = non_image_positions.contiguous()

        expected_counts: List[int] = []
        if image_grid_thw is not None and image_grid_thw.numel() > 0:
            merge_area = max(1, int(spatial_merge_size) ** 2)
            expected_counts = [
                int(((grid[0] * grid[1] * grid[2]).item()) // merge_area)
                for grid in image_grid_thw
            ]

        self.input_layout = {
            "ready": True,
            "view_indices": image_groups,
            "all_visual_indices": all_visual_indices,
            "text_positions": text_positions,
            "expected_counts": expected_counts,
        }
        self.runtime_stats["layout_ready"] = 1
        self.runtime_stats["layout_query_tokens"] = int(text_positions.numel())
        self.runtime_stats["layout_view0_tokens"] = int(image_groups[0].numel()) if len(image_groups) > 0 else 0
        self.runtime_stats["layout_view1_tokens"] = int(image_groups[1].numel()) if len(image_groups) > 1 else 0

    def _get_qk_layer_and_modules(self) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
        layer_idx = max(0, min(int(self.config.qk_layer), self.num_layers - 1))
        layer = self.layers[layer_idx]
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None or not hasattr(self_attn, "q_proj") or not hasattr(self_attn, "k_proj"):
            raise AttributeError(f"Layer {layer_idx} does not expose self_attn.q_proj/k_proj.")
        return layer, self_attn.q_proj, self_attn.k_proj

    @staticmethod
    def _infer_attention_heads(
        model: Any,
        q_proj: torch.nn.Module,
        k_proj: torch.nn.Module,
        hidden_size: int,
    ) -> Tuple[int, int, int]:
        config = getattr(model, "config", None)
        num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        num_kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
        if num_heads <= 0:
            text_config = getattr(config, "text_config", None)
            num_heads = int(getattr(text_config, "num_attention_heads", 0) or 0)
            num_kv_heads = int(getattr(text_config, "num_key_value_heads", 0) or 0)
        out_features = int(getattr(q_proj, "out_features", hidden_size))
        if num_heads <= 0 or out_features % num_heads != 0:
            num_heads = 1
        k_out_features = int(getattr(k_proj, "out_features", out_features))
        if num_kv_heads <= 0 or k_out_features % num_kv_heads != 0:
            num_kv_heads = num_heads if k_out_features == out_features else 1
        head_dim = max(1, out_features // num_heads)
        kv_head_dim = max(1, k_out_features // num_kv_heads)
        if kv_head_dim != head_dim:
            raise RuntimeError(
                f"VLA-ADP Q/K head dims do not match: q_head_dim={head_dim}, k_head_dim={kv_head_dim}."
            )
        return num_heads, num_kv_heads, head_dim

    def _compute_qk_importance(
        self,
        text_emb: torch.Tensor,
        visual_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Step 2: before the LLM forward, reuse pretrained W_Q/W_K from qk_layer.
        layer, q_proj, k_proj = self._get_qk_layer_and_modules()
        input_layernorm = getattr(layer, "input_layernorm", None)
        if input_layernorm is not None:
            # Match the official ADP Q/K probe: layer-0 scoring uses the same
            # pre-attention normalization that the LLM layer would apply.
            text_emb = input_layernorm(text_emb)
            visual_emb = input_layernorm(visual_emb)
        q = q_proj(text_emb)
        k = k_proj(visual_emb)
        num_heads, num_kv_heads, head_dim = self._infer_attention_heads(self.model, q_proj, k_proj, q.shape[-1])
        q = q.view(q.shape[0], q.shape[1], num_heads, head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], num_kv_heads, head_dim).transpose(1, 2)
        if num_kv_heads != num_heads:
            if num_heads % num_kv_heads != 0:
                raise RuntimeError(
                    f"VLA-ADP cannot repeat KV heads: num_heads={num_heads}, num_kv_heads={num_kv_heads}."
                )
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        attn_scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(float(head_dim))
        # Step 3: aggregate multi-head and multi-text-token logits into one score per visual token.
        return attn_scores.mean(dim=1).mean(dim=1)[0]

    def _allocate_view_keeps(self, view_lengths: Sequence[int], total_keep: int) -> List[int]:
        # Step 5: allocate the total visual-token budget by view instead of
        # pruning a unified pool; for two views this defaults to main:wrist = 4:6.
        total_capacity = int(sum(view_lengths))
        total_keep = max(1, min(int(total_keep), total_capacity))
        if total_keep >= total_capacity:
            return [int(x) for x in view_lengths]
        non_empty = [idx for idx, length in enumerate(view_lengths) if int(length) > 0]
        counts = [0] * len(view_lengths)
        min_keep = max(1, int(self.config.min_keep_per_view))
        if total_keep >= len(non_empty) * min_keep:
            for idx in non_empty:
                counts[idx] = min(int(view_lengths[idx]), min_keep)
        else:
            for idx in non_empty[:total_keep]:
                counts[idx] = 1
        remaining = total_keep - sum(counts)
        if remaining <= 0:
            return counts

        if len(view_lengths) == 2:
            weights = [float(self.config.main_view_weight), float(self.config.wrist_view_weight)]
        else:
            weights = [1.0] * len(view_lengths)
        weight_sum = sum(max(0.0, w) for w in weights)
        if weight_sum <= 0.0:
            weights = [1.0] * len(view_lengths)
            weight_sum = float(len(view_lengths))
        weights = [max(0.0, w) / weight_sum for w in weights]

        fractional: List[Tuple[float, int]] = []
        assigned_extra = [0] * len(view_lengths)
        for idx, length in enumerate(view_lengths):
            capacity = int(length) - counts[idx]
            if capacity <= 0:
                continue
            raw = remaining * weights[idx]
            take = min(capacity, int(math.floor(raw)))
            assigned_extra[idx] = take
            fractional.append((raw - float(take), idx))
        for idx, take in enumerate(assigned_extra):
            counts[idx] += take
        leftover = total_keep - sum(counts)
        for _frac, idx in sorted(fractional, key=lambda x: x[0], reverse=True):
            if leftover <= 0:
                break
            capacity = int(view_lengths[idx]) - counts[idx]
            if capacity <= 0:
                continue
            counts[idx] += 1
            leftover -= 1
        idx = 0
        while leftover > 0 and idx < len(counts) * 2:
            view_idx = idx % len(counts)
            if counts[view_idx] < int(view_lengths[view_idx]):
                counts[view_idx] += 1
                leftover -= 1
            idx += 1
        return [int(x) for x in counts]

    def _finalize_flops_stats(self, original_seq_len: int, compressed_seq_len: int) -> None:
        if original_seq_len <= 0 or compressed_seq_len <= 0 or self.num_layers <= 0:
            return
        self.runtime_stats["original_prefill_seq_len"] = int(original_seq_len)
        self.runtime_stats["layer_tokens"] = {idx: int(compressed_seq_len) for idx in range(self.num_layers)}
        self.runtime_stats["decoder_layers_recorded"] = int(self.num_layers)
        self.runtime_stats["decoder_layers_missing"] = 0
        original = int(original_seq_len) * int(self.num_layers)
        actual = int(compressed_seq_len) * int(self.num_layers)
        self.runtime_stats["flops_reduction"] = 1.0 - (float(actual) / float(original))

    def compress_generate_inputs(self, model: Any, generate_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # Step 1: input multimodal information is already packed in generate_kwargs.
        if not self.should_compress_generate_inputs(generate_kwargs):
            return generate_kwargs

        patch_t0 = time.perf_counter()
        input_ids = generate_kwargs.get("input_ids", None)
        pixel_values = generate_kwargs.get("pixel_values", None)
        image_grid_thw = generate_kwargs.get("image_grid_thw", None)
        attention_mask = generate_kwargs.get("attention_mask", None)
        image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
        if input_ids is None or pixel_values is None or image_grid_thw is None or image_token_id is None:
            return generate_kwargs

        if not self.input_layout.get("ready"):
            self.prepare_input_layout(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                image_token_id=image_token_id,
                num_views=self.config.num_views,
            )
        if not self.input_layout.get("ready"):
            return generate_kwargs

        text_embeddings = _get_text_embedding_layer(model)
        pixel_values = pixel_values.type(model.visual.dtype)
        visual_output = model.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = visual_output[0] if isinstance(visual_output, tuple) else visual_output
        image_embeds = image_embeds.to(device=input_ids.device, dtype=text_embeddings.weight.dtype)

        image_mask = input_ids == int(image_token_id)
        visual_positions = torch.nonzero(image_mask[0], as_tuple=False).flatten()
        if int(visual_positions.numel()) != int(image_embeds.shape[0]):
            return generate_kwargs
        total_visual = int(image_embeds.shape[0])
        keep_total = max(1, min(int(self.config.keep_visual_tokens), total_visual))
        self.runtime_stats["last_tokens_before"] = total_visual
        self.runtime_stats["prefill_original_visual_tokens"] = total_visual
        if keep_total >= total_visual:
            self.runtime_stats["last_tokens_after"] = total_visual
            self.runtime_stats["final_visual_tokens"] = total_visual
            self.runtime_stats["raw_total_budget"] = float(total_visual)
            return generate_kwargs

        full_embeds = text_embeddings(input_ids)
        full_embeds = full_embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(full_embeds), image_embeds)
        text_positions = self.input_layout["text_positions"].to(device=input_ids.device)
        if text_positions.numel() == 0:
            return generate_kwargs
        text_emb = full_embeds.index_select(1, text_positions)
        visual_emb = image_embeds.unsqueeze(0)
        scores = self._compute_qk_importance(text_emb=text_emb, visual_emb=visual_emb)

        # Step 4: select Top-K visual tokens.
        # Step 5: perform that Top-K independently inside each view quota.
        view_groups = self.input_layout["view_indices"]
        view_lengths = [int(group.numel()) for group in view_groups]
        view_keeps = self._allocate_view_keeps(view_lengths, keep_total)
        selected_visual_positions: List[torch.Tensor] = []
        selected_visual_embed_indices: List[torch.Tensor] = []
        visual_offset = 0
        topk_t0 = time.perf_counter()
        for group, view_len, keep_count in zip(view_groups, view_lengths, view_keeps):
            view_scores = scores[visual_offset : visual_offset + view_len]
            k_i = max(1, min(int(keep_count), int(view_len)))
            local_idx = torch.topk(view_scores, k=k_i, dim=0, largest=True, sorted=False).indices.sort().values
            selected_visual_positions.append(group.to(device=input_ids.device).index_select(0, local_idx))
            selected_visual_embed_indices.append(local_idx + int(visual_offset))
            visual_offset += view_len
        self.runtime_stats["topk_seconds"] = float(time.perf_counter() - topk_t0)

        selected_positions = torch.cat(selected_visual_positions, dim=0).sort().values
        selected_embed_indices = torch.cat(selected_visual_embed_indices, dim=0)
        # Preserve original sequence order after per-view Top-K selection.
        order = torch.argsort(selected_positions)
        selected_positions = selected_positions.index_select(0, order)
        selected_embed_indices = selected_embed_indices.index_select(0, order)

        # Step 6: rebuild the multimodal sequence with the selected visual tokens.
        gather_t0 = time.perf_counter()
        keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        keep_mask[visual_positions] = False
        keep_mask[selected_positions] = True
        full_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
        compressed_input_ids = input_ids.index_select(1, full_indices)
        compressed_embeds = full_embeds.index_select(1, full_indices).contiguous()
        compressed_attention_mask = attention_mask.index_select(1, full_indices) if attention_mask is not None else None

        selected_visual_embeds = image_embeds.index_select(0, selected_embed_indices.to(image_embeds.device))
        rel_visual_positions = torch.searchsorted(full_indices, selected_positions)
        compressed_embeds[:, rel_visual_positions, :] = selected_visual_embeds.to(compressed_embeds.dtype)

        original_position_ids = None
        try:
            original_position_ids, rope_deltas = model.get_rope_index(
                input_ids,
                image_grid_thw,
                generate_kwargs.get("video_grid_thw", None),
                generate_kwargs.get("second_per_grid_ts", None),
                attention_mask,
            )
            model.rope_deltas = rope_deltas
        except Exception:
            original_position_ids = None

        compressed_position_ids = None
        if original_position_ids is not None:
            gather_idx = full_indices.view(1, 1, -1).expand(original_position_ids.shape[0], input_ids.shape[0], -1)
            compressed_position_ids = original_position_ids.gather(2, gather_idx).contiguous()
            try:
                model.rope_deltas = (
                    compressed_position_ids.max(dim=0).values.max(dim=-1, keepdim=True).values
                    + 1
                    - compressed_input_ids.shape[1]
                )
            except Exception:
                pass

        output_kwargs = dict(generate_kwargs)
        output_kwargs["input_ids"] = compressed_input_ids
        output_kwargs["inputs_embeds"] = compressed_embeds
        if compressed_attention_mask is not None:
            output_kwargs["attention_mask"] = compressed_attention_mask
        if compressed_position_ids is not None:
            output_kwargs["position_ids"] = compressed_position_ids
        output_kwargs.pop("pixel_values", None)
        output_kwargs.pop("image_grid_thw", None)

        self.runtime_stats["gather_seconds"] = float(time.perf_counter() - gather_t0)
        self.runtime_stats["patch_total_seconds"] = float(time.perf_counter() - patch_t0)
        self.runtime_stats["prefill_compressed"] = 1
        self.runtime_stats["last_tokens_after"] = int(selected_positions.numel())
        self.runtime_stats["final_visual_tokens"] = int(selected_positions.numel())
        self.runtime_stats["raw_total_budget"] = float(selected_positions.numel())
        self.runtime_stats["pruned_layers"] = int(self.num_layers)
        self.runtime_stats["active_pruning_layers"] = float(self.num_layers)
        self.runtime_stats["view_keep_counts"] = [int(x) for x in view_keeps]
        self._finalize_flops_stats(int(input_ids.shape[1]), int(compressed_input_ids.shape[1]))
        return output_kwargs

    @staticmethod
    def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float64).copy()
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return np.eye(3, dtype=np.float64)
        q = q / norm
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _euler_delta_to_matrix(delta_rpy: np.ndarray) -> np.ndarray:
        rx, ry, rz = [float(x) for x in delta_rpy]
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
        ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
        rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
        return rx_m @ ry_m @ rz_m

    @classmethod
    def _pose_from_obs(cls, obs: Dict[str, Any]) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = cls._quat_to_matrix(np.asarray(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=np.float64))
        pose[:3, 3] = np.asarray(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=np.float64)
        return pose

    @classmethod
    def _apply_action_delta(cls, pose: np.ndarray, action: np.ndarray) -> np.ndarray:
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = cls._euler_delta_to_matrix(np.asarray(action[3:6], dtype=np.float64))
        delta[:3, 3] = np.asarray(action[:3], dtype=np.float64)
        return pose @ delta

    def _split_action_chunks(self, action_sequence: np.ndarray) -> List[np.ndarray]:
        actions = np.asarray(action_sequence, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim == 3:
            return [np.asarray(chunk, dtype=np.float64) for chunk in actions if chunk.size > 0]
        if actions.ndim != 2 or actions.shape[-1] < 6:
            return []
        chunk_size = max(1, int(self.config.chunk_size))
        if actions.shape[0] <= chunk_size:
            return [actions]
        return [actions[start : start + chunk_size] for start in range(0, actions.shape[0], chunk_size)]

    def _compute_chunk_delta(self, chunk: np.ndarray, start_pose: np.ndarray) -> Tuple[float, np.ndarray]:
        # Step 10 feedback for Step 7: compute the windowed EEF motion from the generated action chunk.
        pose = np.array(start_pose, dtype=np.float64, copy=True)
        positions = [pose[:3, 3].copy()]
        for action in chunk:
            pose = self._apply_action_delta(pose, np.asarray(action, dtype=np.float64))
            positions.append(pose[:3, 3].copy())
        method = str(getattr(self.config, "delta_method", "net") or "net").lower()
        if method in {"sum", "arc_sum", "path"}:
            delta = 0.0
            for prev, curr in zip(positions[:-1], positions[1:]):
                delta += float(np.linalg.norm(curr - prev))
        else:
            delta = float(np.linalg.norm(positions[-1] - positions[0]))
        return float(delta), pose

    def _update_gate_from_delta(self, delta: float) -> None:
        # Step 7: feed the latest action-trajectory window into the dynamic gate.
        self.delta_history.append(float(delta))
        self.completed_windows += 1

        # Step 9: cold start; the first windows always use full vision.
        if self.completed_windows < int(self.config.cold_start_windows):
            self.current_state = 0
            self.consecutive_pruned = 0
            return

        # Step 8: coarse action -> pruning, fine action -> full vision.  The
        # adjacent-extrema rule mirrors the paper/open-source ADP gate.
        method = str(self.config.decision_method or "adjacent").lower()
        next_state = int(self.current_state)
        if method == "avg":
            avg_delta = float(sum(self.delta_history) / max(1, len(self.delta_history)))
            next_state = 1 if float(delta) >= avg_delta else 0
        else:
            tau = max(1, int(self.config.adjacent_extrema_window))
            previous = self.delta_history[-tau:-1]
            if len(previous) < max(1, tau - 1):
                next_state = 0
            else:
                upper = max(previous)
                lower = min(previous)
                if float(delta) >= upper:
                    next_state = 1
                elif float(delta) < lower:
                    next_state = 0
                else:
                    next_state = int(self.current_state) if bool(self.config.adjacent_last_state) else 1

        if bool(self.config.limit_consecutive_pruned_enabled):
            # Step 9: avoid over-pruning; after several consecutive pruned
            # windows, force the next window back to full vision.
            if next_state == 1:
                if self.consecutive_pruned >= int(self.config.limit_max_consecutive_pruned):
                    next_state = 0
                    self.consecutive_pruned = 0
                else:
                    self.consecutive_pruned += 1
            else:
                self.consecutive_pruned = 0
        self.current_state = int(next_state)

    def observe_action_sequence(self, action_sequence: np.ndarray, start_obs: Optional[Dict[str, Any]] = None) -> None:
        """Update ADP's action-aware gate after one Nora forward.

        If a future Nora variant returns multiple chunks in one forward, they
        are treated as virtual windows and fed to the gate in temporal order.
        Only the final state is used by the next visual forward, matching the
        fact that the current forward cannot be re-prefilled mid-generation.
        """
        chunks = self._split_action_chunks(action_sequence)
        if not chunks:
            return
        pose = self._pose_from_obs(start_obs) if start_obs is not None else self._last_pose_matrix
        if pose is None:
            pose = np.eye(4, dtype=np.float64)
        updates = 0
        last_delta = 0.0
        for chunk in chunks:
            last_delta, pose = self._compute_chunk_delta(chunk, pose)
            self._update_gate_from_delta(last_delta)
            updates += 1
        self._last_pose_matrix = pose
        self.runtime_stats["vlaadp_virtual_windows_updated"] = int(
            self.runtime_stats.get("vlaadp_virtual_windows_updated", 0) + updates
        )
        self.runtime_stats["vlaadp_last_delta"] = float(last_delta)
        self.runtime_stats["vlaadp_completed_windows"] = int(self.completed_windows)
        self.runtime_stats["vlaadp_delta_history_size"] = int(len(self.delta_history))
        self.runtime_stats["vlaadp_consecutive_pruned"] = int(self.consecutive_pruned)
        self.runtime_stats["vlaadp_next_state"] = int(self.current_state)
