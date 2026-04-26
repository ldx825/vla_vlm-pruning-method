from __future__ import annotations

import math
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class NoraComparePruningConfig:
    enabled: bool = False
    method: str = "off"
    visual_tokens: int = 512
    num_views: int = 2
    fastv_pruning_layer: int = 2
    sparsevlm_pruning_start_layer: int = 2
    sparsevlm_lambda: float = 1.0
    sparsevlm_recycle_top_ratio: float = 0.5
    sparsevlm_cluster_ratio: float = 0.25
    sparsevlm_rank_tolerance: float = 1e-4
    sparsevlm_knn_k: int = 8
    pyramiddrop_pruning_start_layer: int = 2
    pyramiddrop_num_stages: int = 4
    pyramiddrop_lambda: float = 0.0
    detailed_timing: bool = True
    debug: bool = False
    decode_skip_seq_len: int = 96


def compute_fastv_attention_scores(
    attn_weights: torch.Tensor,
    visual_positions: torch.Tensor,
    query_mask: torch.Tensor,
) -> torch.Tensor:
    """
    FastV paper Step 2:
    Score every visual token by the average attention it receives from all non-visual tokens.

    Important experiment convention for NORA:
    - We rank ALL visual tokens jointly across all views.
    - The scoring tokens are all non-visual tokens in the current sequence.
    """
    rel_attn = attn_weights[:, :, query_mask, :]
    if rel_attn.numel() == 0 or visual_positions.numel() == 0:
        return torch.zeros(
            attn_weights.shape[0],
            int(visual_positions.numel()),
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )
    visual_attn = rel_attn.index_select(3, visual_positions)
    return visual_attn.mean(dim=1).mean(dim=1)


def select_fastv_keep_indices(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    keep_count = max(1, min(int(keep_count), int(scores.shape[1])))
    _, idx = torch.topk(scores, keep_count, dim=1)
    idx, _ = idx.sort(dim=1)
    return idx


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if scores.numel() == 0:
        return scores
    scores = scores - scores.min()
    denom = scores.max().clamp(min=1e-6)
    return scores / denom


def _select_evenly_spaced_indices(length: int, count: int, device: torch.device) -> torch.Tensor:
    length = max(0, int(length))
    count = max(0, min(int(count), length))
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count >= length:
        return torch.arange(length, dtype=torch.long, device=device)
    return torch.linspace(0, length - 1, steps=count, device=device).round().long().unique(sorted=True)[:count]


def _get_text_embedding_layer(model: Any) -> torch.nn.Module:
    """Return the LLM token embedding layer across PEFT/Qwen2.5-VL wrappers."""
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

    raise AttributeError(
        "Could not locate text token embedding layer; expected get_input_embeddings(), "
        "embed_tokens, or language_model.embed_tokens on the NORA/Qwen2.5-VL model."
    )


def solve_pyramiddrop_lambda(initial_tokens: int, target_average_tokens: int, stages: int) -> float:
    """Solve lambda so mean_i initial_tokens * lambda**i approximately matches the requested average budget."""
    initial_tokens = max(1, int(initial_tokens))
    target_average_tokens = max(1, min(int(target_average_tokens), initial_tokens))
    stages = max(1, int(stages))
    if target_average_tokens >= initial_tokens:
        return 1.0

    target_ratio = float(target_average_tokens) / float(initial_tokens)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) * 0.5
        avg_ratio = sum(mid ** (idx + 1) for idx in range(stages)) / float(stages)
        if avg_ratio < target_ratio:
            lo = mid
        else:
            hi = mid
    return float(hi)


def compute_sparsevlm_text_relevance_scores(
    attn_weights: torch.Tensor,
    visual_positions: torch.Tensor,
    text_positions: torch.Tensor,
) -> torch.Tensor:
    """
    SparseVLM Eq. (6):
    Estimate which text tokens are actually related to the visual block.
    """
    if visual_positions.numel() == 0 or text_positions.numel() == 0:
        return torch.zeros(
            attn_weights.shape[0],
            int(text_positions.numel()),
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )
    rel_attn = attn_weights.index_select(2, text_positions).index_select(3, visual_positions)
    return rel_attn.mean(dim=1).mean(dim=-1)


def select_sparsevlm_rater_indices(text_relevance_scores: torch.Tensor) -> List[torch.Tensor]:
    """
    SparseVLM Eq. (5):
    Keep text tokens whose relevance is above the per-sample mean.
    """
    raters: List[torch.Tensor] = []
    for batch_idx in range(int(text_relevance_scores.shape[0])):
        row = text_relevance_scores[batch_idx]
        if row.numel() == 0:
            raters.append(torch.empty(0, dtype=torch.long, device=row.device))
            continue
        threshold = row.mean()
        indices = torch.nonzero(row >= threshold, as_tuple=False).flatten()
        if indices.numel() == 0:
            indices = row.topk(1, dim=0).indices.flatten()
        raters.append(indices)
    return raters


def compute_sparsevlm_priority_and_visual_scores(
    attn_weights: torch.Tensor,
    visual_positions: torch.Tensor,
    text_positions: torch.Tensor,
    rater_indices: Sequence[torch.Tensor],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    SparseVLM Eq. (2) + Eq. (3):
    Build the text-to-vision priority matrix P and derive per-visual-token scores.
    """
    batch_size = int(attn_weights.shape[0])
    vis_len = int(visual_positions.numel())
    if vis_len <= 0 or text_positions.numel() <= 0:
        return [], torch.zeros(batch_size, vis_len, device=attn_weights.device, dtype=attn_weights.dtype)

    rel_attn = attn_weights.index_select(2, text_positions).index_select(3, visual_positions)
    priority_mats: List[torch.Tensor] = []
    visual_scores = torch.zeros(batch_size, vis_len, device=attn_weights.device, dtype=attn_weights.dtype)

    for batch_idx in range(batch_size):
        batch_attn = rel_attn[batch_idx].mean(dim=0)  # [Lt, Lv]
        raters = rater_indices[batch_idx]
        if raters.numel() == 0:
            priority = batch_attn[:0]
        else:
            priority = batch_attn.index_select(0, raters)
        priority_mats.append(priority)
        if priority.numel() > 0:
            visual_scores[batch_idx] = priority.mean(dim=0)

    return priority_mats, visual_scores


def estimate_sparsevlm_rank(priority_matrix: torch.Tensor, tolerance: float = 1e-4) -> float:
    """
    SparseVLM Eq. (7):
    Use numerical rank(P) to estimate redundancy.
    """
    if priority_matrix.numel() == 0:
        return 0.0
    try:
        singular_vals = torch.linalg.svdvals(priority_matrix.float())
        if singular_vals.numel() == 0:
            return 0.0
        threshold = float(tolerance) * float(singular_vals.max().item())
        return float((singular_vals > threshold).sum().item())
    except Exception:
        return float(min(priority_matrix.shape[-2], priority_matrix.shape[-1]))


def project_sparsevlm_exact_budget(
    current_visual_tokens: int,
    target_budget: int,
    rank_value: float,
    lambda_scale: float,
    recycle_top_ratio: float,
    cluster_ratio: float,
    remaining_stages: int = 1,
) -> Dict[str, int]:
    """
    Engineering projection of SparseVLM's adaptive deletion rule to an exact final budget.

    Paper alignment:
    - Eq. (7): N = lambda * (Lv - rank(P))
    - top-tau deleted tokens are recycled
    - theta * Lr reconstructed tokens are kept
    """
    current_visual_tokens = max(0, int(current_visual_tokens))
    if current_visual_tokens <= 0:
        return {
            "final_budget": 0,
            "stage_budget": 0,
            "stage_delete": 0,
            "soft_delete": 0,
            "soft_keep": 0,
            "recycle_pool_size": 0,
            "recycle_budget": 0,
            "direct_keep_budget": 0,
        }

    final_budget = max(1, min(int(target_budget), current_visual_tokens))
    remaining_stages = max(1, int(remaining_stages))
    redundancy = max(float(current_visual_tokens) - float(rank_value), 0.0)
    soft_delete = int(round(float(lambda_scale) * redundancy))
    soft_delete = max(0, min(soft_delete, current_visual_tokens - 1))
    soft_keep = current_visual_tokens - soft_delete

    required_delete = max(0, current_visual_tokens - final_budget)
    if required_delete <= 0:
        stage_delete = 0
        stage_budget = final_budget
    else:
        min_delete_now = max(1, int(math.ceil(required_delete / float(remaining_stages))))
        future_stages = max(0, remaining_stages - 1)
        min_future_delete = min(max(required_delete - 1, 0), future_stages)
        max_delete_now = max(min_delete_now, required_delete - min_future_delete)
        stage_delete = max(min_delete_now, min(soft_delete, max_delete_now))
        stage_delete = max(0, min(stage_delete, required_delete))
        stage_budget = current_visual_tokens - stage_delete

    recycle_pool_size = int(math.ceil(float(recycle_top_ratio) * float(soft_delete)))
    recycle_pool_size = max(0, min(recycle_pool_size, soft_delete))
    suggested_recycle = int(math.ceil(float(cluster_ratio) * float(recycle_pool_size)))
    suggested_recycle = max(0, min(suggested_recycle, recycle_pool_size))
    recycle_budget = min(suggested_recycle, max(stage_budget - 1, 0))
    direct_keep_budget = stage_budget - recycle_budget

    if direct_keep_budget <= 0:
        direct_keep_budget = 1
        recycle_budget = stage_budget - 1

    return {
        "final_budget": int(final_budget),
        "stage_budget": int(stage_budget),
        "stage_delete": int(stage_delete),
        "soft_delete": int(soft_delete),
        "soft_keep": int(soft_keep),
        "recycle_pool_size": int(recycle_pool_size),
        "recycle_budget": int(recycle_budget),
        "direct_keep_budget": int(direct_keep_budget),
    }


def recycle_sparsevlm_tokens_with_density_peaks(
    recycled_pool_tokens: torch.Tensor,
    recycled_pool_positions: torch.Tensor,
    recycle_budget: int,
    knn_k: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SparseVLM Eq. (8) / Eq. (9) / Eq. (10):
    Density-peak recycle, cluster-center selection, and token reconstruction.
    """
    if recycle_budget <= 0 or recycled_pool_tokens.numel() == 0:
        return (
            recycled_pool_tokens.new_zeros((0, recycled_pool_tokens.shape[-1])),
            recycled_pool_positions.new_zeros((0,), dtype=torch.long),
        )

    pool_len = int(recycled_pool_tokens.shape[0])
    recycle_budget = max(0, min(int(recycle_budget), pool_len))
    if recycle_budget == 0:
        return (
            recycled_pool_tokens.new_zeros((0, recycled_pool_tokens.shape[-1])),
            recycled_pool_positions.new_zeros((0,), dtype=torch.long),
        )
    if pool_len <= recycle_budget:
        return recycled_pool_tokens[:recycle_budget], recycled_pool_positions[:recycle_budget]

    normalized = F.normalize(recycled_pool_tokens.float(), dim=-1)
    distances = torch.cdist(normalized, normalized, p=2)
    inf = torch.tensor(float("inf"), device=distances.device, dtype=distances.dtype)
    distances_no_diag = distances + torch.eye(pool_len, device=distances.device, dtype=distances.dtype) * inf

    knn_k = max(1, min(int(knn_k), pool_len - 1))
    nearest_distances = torch.topk(distances_no_diag, k=knn_k, dim=1, largest=False).values
    rho = torch.exp(-nearest_distances.pow(2).mean(dim=1) / float(knn_k))

    delta = torch.zeros(pool_len, device=distances.device, dtype=distances.dtype)
    for idx in range(pool_len):
        higher_density = torch.nonzero(rho > rho[idx], as_tuple=False).flatten()
        if higher_density.numel() > 0:
            delta[idx] = distances[idx].index_select(0, higher_density).min()
        else:
            delta[idx] = distances[idx].max()

    center_scores = rho * delta
    center_indices = torch.topk(center_scores, k=recycle_budget, dim=0).indices
    center_indices, _ = torch.sort(center_indices)

    center_tokens = normalized.index_select(0, center_indices)
    similarity = normalized @ center_tokens.transpose(0, 1)
    assignments = similarity.argmax(dim=1)

    reconstructed_tokens = []
    representative_positions = []
    for cluster_idx in range(recycle_budget):
        member_indices = torch.nonzero(assignments == cluster_idx, as_tuple=False).flatten()
        if member_indices.numel() == 0:
            member_indices = center_indices[cluster_idx : cluster_idx + 1]
        cluster_tokens = recycled_pool_tokens.index_select(0, member_indices)
        reconstructed_tokens.append(cluster_tokens.sum(dim=0, keepdim=True))
        representative_positions.append(recycled_pool_positions[center_indices[cluster_idx]].view(1))

    return (
        torch.cat(reconstructed_tokens, dim=0).to(recycled_pool_tokens.dtype),
        torch.cat(representative_positions, dim=0).to(dtype=torch.long),
    )


class NoraComparePruningPatch:
    """
    Unified baseline patch for NORA decoder-side pruning baselines.

    Design principle:
    - Keep NORA AutoPrune's decoder-forward patching skeleton intact.
    - Only replace the token-scoring / keep-budget logic with FastV, SparseVLM, or PyramidDrop.
    - Treat all visual tokens from all views as one joint block during pruning.
    """

    def __init__(self, model, config: NoraComparePruningConfig):
        self.model = model
        self.config = config
        self.layers = self._resolve_layers(model)
        self.original_forwards: Dict[int, Any] = {}
        self.global_indices: Optional[torch.Tensor] = None
        self.input_layout: Dict[str, Any] = {}
        self.runtime_stats: Dict[str, Any] = {}
        self.num_layers = len(self.layers)
        self.active_stage_map = self._resolve_active_stage_map()
        self.prefill_methods = {"fastervlm", "fastvlm_prefill", "visionzip"}
        self._prefill_original_visual_tokens = 0

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
        raise RuntimeError("Unable to locate decoder layers for compare pruning in the provided model.")

    def _resolve_active_stage_map(self) -> Dict[int, int]:
        method = str(self.config.method or "off").lower()
        if method == "fastv":
            prune_idx = max(0, int(self.config.fastv_pruning_layer) - 1)
            return {prune_idx: 0}
        if method == "sparsevlm":
            start_idx = max(0, int(self.config.sparsevlm_pruning_start_layer) - 1)
            return {idx: idx - start_idx for idx in range(start_idx, self.num_layers)}
        if method == "pyramiddrop":
            start_idx = max(0, int(self.config.pyramiddrop_pruning_start_layer) - 1)
            if start_idx >= self.num_layers:
                return {}
            # PyramidDrop prunes at stage boundaries, not after every decoder
            # layer.  Split layers from pruning_start_layer to the end into a
            # small number of stages, then prune only at each stage end.
            num_stages = max(1, min(int(self.config.pyramiddrop_num_stages), self.num_layers - start_idx))
            span = self.num_layers - start_idx
            boundaries: List[int] = []
            for stage_idx in range(num_stages):
                end_exclusive = start_idx + int(math.ceil(float(stage_idx + 1) * float(span) / float(num_stages)))
                boundaries.append(min(self.num_layers - 1, max(start_idx, end_exclusive - 1)))
            unique_boundaries = sorted(set(boundaries))
            return {layer_idx: stage_idx for stage_idx, layer_idx in enumerate(unique_boundaries)}
        return {}

    def reset_runtime_state(self) -> None:
        self.global_indices = None
        self.input_layout = {}
        self.runtime_stats = {
            "patch_total_seconds": 0.0,
            "topk_seconds": 0.0,
            "gather_seconds": 0.0,
            "last_tokens_before": 0,
            "last_tokens_after": 0,
            "schedule_len": 0,
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
            "raw_total_budget": 0.0,
            "active_pruning_layers": 0.0,
            "flops_reduction": 0.0,
            "prefill_compressed": 0,
            "prefill_original_visual_tokens": 0,
            "fastervlm_reduction_ratio": 0.0,
            "visionzip_dominant_tokens": 0,
            "visionzip_contextual_tokens": 0,
            "pyramiddrop_lambda": 0.0,
        }

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

    def apply(self) -> None:
        if not self.config.enabled or not self.active_stage_map:
            return
        for idx, layer in enumerate(self.layers):
            self.original_forwards[idx] = layer.forward
            layer.forward = types.MethodType(self._make_new_forward(idx), layer)

    @staticmethod
    def _is_empty_cache(past_key_values: Any) -> bool:
        if past_key_values is None:
            return True
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                return int(get_seq_length()) == 0
            except Exception:
                return False
        return False

    def should_compress_generate_inputs(self, generate_kwargs: Dict[str, Any]) -> bool:
        """True only for the single prefill input of one generate() call.

        FasterVLM/VisionZip are prefill-time methods.  During autoregressive
        decode, HF generate feeds cached states and no fresh pixel tensors; in
        that phase this must be false so KV cache decoding is left untouched.
        """
        method = str(self.config.method or "off").lower()
        if method == "fastv" or method not in self.prefill_methods:
            return False
        if generate_kwargs.get("inputs_embeds", None) is not None:
            return False
        input_ids = generate_kwargs.get("input_ids", None)
        if input_ids is None or getattr(input_ids, "ndim", 0) != 2:
            return False
        if generate_kwargs.get("pixel_values", None) is None:
            return False
        if generate_kwargs.get("image_grid_thw", None) is None:
            return False
        if not self._is_empty_cache(generate_kwargs.get("past_key_values", None)):
            return False
        return True

    def _estimate_decoder_flops_after_prefill_compression(self, original_seq_len: int, compressed_seq_len: int) -> None:
        if original_seq_len <= 0 or compressed_seq_len <= 0 or self.num_layers <= 0:
            return
        self.runtime_stats["original_prefill_seq_len"] = int(original_seq_len)
        self.runtime_stats["layer_tokens"] = {idx: int(compressed_seq_len) for idx in range(self.num_layers)}
        self._finalize_decoder_flops_stats()

    def _compress_visionzip_tokens(
        self,
        image_embeds: torch.Tensor,
        scores: torch.Tensor,
        metric: torch.Tensor,
        target_keep: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        total_visual = int(image_embeds.shape[0])
        target_keep = max(1, min(int(target_keep), total_visual))
        if target_keep >= total_visual:
            keep = torch.arange(total_visual, device=image_embeds.device, dtype=torch.long)
            return image_embeds, keep, {"dominant": total_visual, "contextual": 0, "global": 0}

        # VisionZip reports a final budget of K + M + 1.  Qwen2.5-VL exposes merged
        # patch tokens rather than a CLIP-style CLS output, so the +1 slot is a
        # global summary token built from the unified visual block.
        dominant_count = max(1, int(round(0.7 * max(target_keep - 1, 1))))
        dominant_count = min(dominant_count, target_keep - 1)
        contextual_count = max(0, target_keep - 1 - dominant_count)

        dominant_idx = torch.topk(scores, k=dominant_count, dim=0).indices.sort().values
        dominant_mask = torch.zeros(total_visual, dtype=torch.bool, device=image_embeds.device)
        dominant_mask[dominant_idx] = True
        residual_idx = torch.nonzero(~dominant_mask, as_tuple=False).flatten()

        global_token = image_embeds.mean(dim=0, keepdim=True)
        if contextual_count <= 0 or residual_idx.numel() == 0:
            selected_idx = torch.cat([dominant_idx[: max(target_keep - 1, 0)], residual_idx[:1]], dim=0)
            selected_idx = selected_idx[:target_keep].sort().values
            tokens = image_embeds.index_select(0, selected_idx)
            if tokens.shape[0] < target_keep:
                tokens = torch.cat([global_token, tokens], dim=0)[:target_keep]
                selected_idx = torch.cat([residual_idx[:1], selected_idx], dim=0)[:target_keep].sort().values
            return tokens, selected_idx, {"dominant": int(dominant_count), "contextual": 0, "global": 1}

        residual_metric = F.normalize(metric.index_select(0, residual_idx).float(), dim=-1)
        center_local = _select_evenly_spaced_indices(int(residual_idx.numel()), contextual_count, image_embeds.device)
        center_idx = residual_idx.index_select(0, center_local)
        center_metric = residual_metric.index_select(0, center_local)
        non_center_mask = torch.ones(int(residual_idx.numel()), dtype=torch.bool, device=image_embeds.device)
        non_center_mask[center_local] = False
        merge_metric = residual_metric[non_center_mask]
        merge_tokens = image_embeds.index_select(0, residual_idx[non_center_mask])
        center_tokens = image_embeds.index_select(0, center_idx)
        if merge_metric.numel() > 0:
            similarity = merge_metric @ center_metric.transpose(0, 1)
            assign = similarity.argmax(dim=1)
            aggregated = torch.zeros_like(center_tokens)
            counts = torch.zeros(contextual_count, 1, dtype=center_tokens.dtype, device=center_tokens.device)
            aggregated.index_add_(0, assign, merge_tokens)
            counts.index_add_(0, assign, torch.ones(assign.shape[0], 1, dtype=center_tokens.dtype, device=center_tokens.device))
            contextual_tokens = center_tokens + aggregated / counts.clamp(min=1.0)
        else:
            contextual_tokens = center_tokens

        merged_positions = torch.cat([dominant_idx, center_idx, residual_idx[:1]], dim=0)
        merged_tokens = torch.cat(
            [image_embeds.index_select(0, dominant_idx), contextual_tokens.to(image_embeds.dtype), global_token],
            dim=0,
        )
        order = torch.argsort(merged_positions[: merged_tokens.shape[0]])
        return (
            merged_tokens.index_select(0, order)[:target_keep].contiguous(),
            merged_positions.index_select(0, order)[:target_keep].contiguous(),
            {"dominant": int(dominant_count), "contextual": int(contextual_count), "global": 1},
        )

    def compress_generate_inputs(self, model: Any, generate_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply FasterVLM/VisionZip before LLM prefill by replacing image placeholders with a shorter embed sequence."""
        if not self.should_compress_generate_inputs(generate_kwargs):
            return generate_kwargs
        method = str(self.config.method or "off").lower()
        input_ids = generate_kwargs.get("input_ids", None)
        pixel_values = generate_kwargs.get("pixel_values", None)
        image_grid_thw = generate_kwargs.get("image_grid_thw", None)

        patch_t0 = time.perf_counter()
        image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
        if image_token_id is None:
            return generate_kwargs
        image_mask = input_ids == int(image_token_id)
        visual_positions = torch.nonzero(image_mask[0], as_tuple=False).flatten()
        if visual_positions.numel() <= 0:
            return generate_kwargs

        pixel_values = pixel_values.type(model.visual.dtype)
        visual_output = model.visual(pixel_values, grid_thw=image_grid_thw)
        if isinstance(visual_output, tuple):
            image_embeds = visual_output[0]
            attn_logits = visual_output[1] if len(visual_output) > 1 else None
            attn_key = visual_output[2] if len(visual_output) > 2 else None
        else:
            image_embeds = visual_output
            attn_logits = None
            attn_key = None

        text_embeddings = _get_text_embedding_layer(model)
        image_embeds = image_embeds.to(device=input_ids.device, dtype=text_embeddings.weight.dtype)
        total_visual = int(image_embeds.shape[0])
        if int(visual_positions.numel()) != total_visual:
            return generate_kwargs
        target_keep = max(1, min(int(self.config.visual_tokens), total_visual))
        if target_keep >= total_visual:
            self.runtime_stats["last_tokens_before"] = total_visual
            self.runtime_stats["last_tokens_after"] = total_visual
            self.runtime_stats["final_visual_tokens"] = total_visual
            self.runtime_stats["raw_total_budget"] = float(total_visual)
            return generate_kwargs

        # FasterVLM/VisionZip paper alignment: both rank the unified visual block
        # before LLM prefill.  If the local Qwen vision tower exposes final-layer
        # CLS/patch attention (as in the reference implementation), use it;
        # otherwise fall back to patch embedding norm so the pruning is still real.
        if attn_logits is not None and attn_logits.numel() == total_visual:
            scores = _normalize_scores(attn_logits.flatten().to(image_embeds.device))
        else:
            scores = _normalize_scores(image_embeds.float().norm(dim=-1))

        if method in {"fastervlm", "fastvlm_prefill"}:
            keep_idx = torch.topk(scores, k=target_keep, dim=0).indices.sort().values
            compressed_visual = image_embeds.index_select(0, keep_idx)
            self.runtime_stats["fastervlm_reduction_ratio"] = float(1.0 - target_keep / float(total_visual))
        else:
            metric = attn_key[0].to(image_embeds.device) if attn_key is not None and attn_key.ndim == 3 else image_embeds
            compressed_visual, keep_idx, vz_stats = self._compress_visionzip_tokens(
                image_embeds=image_embeds,
                scores=scores,
                metric=metric,
                target_keep=target_keep,
            )
            self.runtime_stats["visionzip_dominant_tokens"] = int(vz_stats["dominant"])
            self.runtime_stats["visionzip_contextual_tokens"] = int(vz_stats["contextual"])

        original_position_ids = None
        try:
            original_position_ids, rope_deltas = model.get_rope_index(
                input_ids,
                image_grid_thw,
                generate_kwargs.get("video_grid_thw", None),
                generate_kwargs.get("second_per_grid_ts", None),
                generate_kwargs.get("attention_mask", None),
            )
            model.rope_deltas = rope_deltas
        except Exception:
            original_position_ids = None

        full_embeds = text_embeddings(input_ids)
        full_embeds = full_embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(full_embeds), image_embeds)
        selected_visual_positions = visual_positions.index_select(0, keep_idx.to(visual_positions.device))
        keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        keep_mask[visual_positions] = False
        keep_mask[selected_visual_positions] = True
        full_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
        compressed_input_ids = input_ids.index_select(1, full_indices)
        compressed_attention_mask = None
        if generate_kwargs.get("attention_mask", None) is not None:
            compressed_attention_mask = generate_kwargs["attention_mask"].index_select(1, full_indices)
        compressed_embeds = full_embeds.index_select(1, full_indices).contiguous()

        rel_visual_positions = torch.searchsorted(full_indices, selected_visual_positions.sort().values)
        compressed_embeds[:, rel_visual_positions, :] = compressed_visual.to(compressed_embeds.dtype)

        generate_kwargs = dict(generate_kwargs)
        generate_kwargs["input_ids"] = compressed_input_ids
        generate_kwargs["inputs_embeds"] = compressed_embeds
        if compressed_attention_mask is not None:
            generate_kwargs["attention_mask"] = compressed_attention_mask
        if original_position_ids is not None:
            gather_idx = full_indices.view(1, 1, -1).expand(original_position_ids.shape[0], input_ids.shape[0], -1)
            compressed_position_ids = original_position_ids.gather(2, gather_idx).contiguous()
            generate_kwargs["position_ids"] = compressed_position_ids
            try:
                model.rope_deltas = (
                    compressed_position_ids.max(dim=0).values.max(dim=-1, keepdim=True).values
                    + 1
                    - compressed_input_ids.shape[1]
                )
            except Exception:
                pass
        generate_kwargs.pop("pixel_values", None)
        generate_kwargs.pop("image_grid_thw", None)

        original_seq_len = int(input_ids.shape[1])
        compressed_seq_len = int(compressed_input_ids.shape[1])
        self.runtime_stats["prefill_compressed"] = 1
        self.runtime_stats["prefill_original_visual_tokens"] = total_visual
        self.runtime_stats["last_tokens_before"] = total_visual
        self.runtime_stats["last_tokens_after"] = target_keep
        self.runtime_stats["final_visual_tokens"] = target_keep
        self.runtime_stats["raw_total_budget"] = float(target_keep)
        self.runtime_stats["pruned_layers"] = 1
        self.runtime_stats["active_pruning_layers"] = 1.0
        self.runtime_stats["patch_total_seconds"] = float(
            self.runtime_stats.get("patch_total_seconds", 0.0) + (time.perf_counter() - patch_t0)
        )
        self._estimate_decoder_flops_after_prefill_compression(original_seq_len, compressed_seq_len)
        return generate_kwargs

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
        requested_views = max(1, int(num_views or 2))

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

        all_visual_indices = torch.cat(image_groups, dim=0).contiguous()
        last_visual_position = int(all_visual_indices.max().item())
        non_image_positions = valid_positions[~image_mask]
        query_positions = non_image_positions[non_image_positions > last_visual_position].contiguous()

        expected_counts: List[int] = []
        if image_grid_thw is not None and image_grid_thw.numel() > 0:
            merge_area = max(1, int(spatial_merge_size) ** 2)
            expected_counts = [
                int(((grid[0] * grid[1] * grid[2]).item()) // merge_area)
                for grid in image_grid_thw
            ]

        self.input_layout = {
            "ready": True,
            "view_indices": [group.contiguous() for group in image_groups],
            "all_visual_indices": all_visual_indices,
            "query_positions": query_positions,
            "expected_counts": expected_counts,
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

    def _run_profiled_original_forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        forward_kwargs: Dict[str, Any],
    ):
        return self.original_forwards[layer_idx](
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **forward_kwargs,
        )

    @staticmethod
    def _gather_shared_positions(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.numel() <= 0:
            return hidden_states[:, :0, :]
        expanded = positions.view(1, -1, 1).expand(hidden_states.shape[0], -1, hidden_states.shape[-1])
        return hidden_states.gather(1, expanded)

    def _record_prefill_layer_token_count(self, layer_idx: int, seq_len: int) -> None:
        if layer_idx not in self.runtime_stats["layer_tokens"]:
            self.runtime_stats["layer_tokens"][layer_idx] = int(seq_len)
        self._finalize_decoder_flops_stats()

    def _run_sparsevlm_joint_pruning(
        self,
        current_layer_output: torch.Tensor,
        attn_weights: torch.Tensor,
        visual_positions: torch.Tensor,
        text_positions: torch.Tensor,
        current_globals: torch.Tensor,
        remaining_stages: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        SparseVLM on NORA with one important VLA-specific convention:
        all visual tokens from all views are pruned jointly as one block.
        """
        batch_size = int(current_layer_output.shape[0])
        current_visual_tokens = int(visual_positions.numel())
        final_budget = min(int(self.config.visual_tokens), current_visual_tokens)
        if final_budget <= 0:
            empty_tokens = current_layer_output[:, :0, :]
            empty_positions = current_globals[:, :0]
            return empty_tokens, empty_positions, {
                "raw_total_budget": 0.0,
                "final_visual_tokens": 0,
                "active_pruning_layers": float(len(self.active_stage_map)),
            }

        if current_visual_tokens <= final_budget:
            identity_tokens = self._gather_shared_positions(current_layer_output, visual_positions)
            identity_positions = current_globals.index_select(1, visual_positions)
            return identity_tokens, identity_positions, {
                "raw_total_budget": float(current_visual_tokens),
                "final_visual_tokens": int(current_visual_tokens),
                "active_pruning_layers": float(len(self.active_stage_map)),
                "sparsevlm_rater_count_mean": 0.0,
                "sparsevlm_rank_mean": 0.0,
            }

        text_relevance = compute_sparsevlm_text_relevance_scores(
            attn_weights=attn_weights,
            visual_positions=visual_positions,
            text_positions=text_positions,
        )
        rater_indices = select_sparsevlm_rater_indices(text_relevance)
        priority_mats, visual_scores = compute_sparsevlm_priority_and_visual_scores(
            attn_weights=attn_weights,
            visual_positions=visual_positions,
            text_positions=text_positions,
            rater_indices=rater_indices,
        )

        visual_tokens = self._gather_shared_positions(current_layer_output, visual_positions)
        visual_global_positions = current_globals.index_select(1, visual_positions)

        new_visual_tokens: List[torch.Tensor] = []
        new_visual_positions: List[torch.Tensor] = []
        rater_counts: List[int] = []
        rank_values: List[float] = []
        soft_delete_values: List[int] = []
        recycle_budget_values: List[int] = []

        for batch_idx in range(batch_size):
            batch_tokens = visual_tokens[batch_idx]
            batch_scores = visual_scores[batch_idx]
            batch_globals = visual_global_positions[batch_idx]
            priority = priority_mats[batch_idx]

            rank_value = estimate_sparsevlm_rank(priority, tolerance=self.config.sparsevlm_rank_tolerance)
            budget_info = project_sparsevlm_exact_budget(
                current_visual_tokens=batch_tokens.shape[0],
                target_budget=final_budget,
                rank_value=rank_value,
                lambda_scale=self.config.sparsevlm_lambda,
                recycle_top_ratio=self.config.sparsevlm_recycle_top_ratio,
                cluster_ratio=self.config.sparsevlm_cluster_ratio,
                remaining_stages=remaining_stages,
            )

            sorted_indices = torch.argsort(batch_scores, descending=True)
            direct_keep_budget = int(budget_info["direct_keep_budget"])
            recycle_budget = int(budget_info["recycle_budget"])
            recycle_pool_size = int(budget_info["recycle_pool_size"])
            stage_budget = int(budget_info["stage_budget"])

            direct_keep_idx = sorted_indices[:direct_keep_budget]
            deleted_idx = sorted_indices[direct_keep_budget:]
            recycle_pool_idx = deleted_idx[:recycle_pool_size]

            direct_tokens = batch_tokens.index_select(0, direct_keep_idx)
            direct_positions = batch_globals.index_select(0, direct_keep_idx)

            recycled_tokens, recycled_positions = recycle_sparsevlm_tokens_with_density_peaks(
                recycled_pool_tokens=batch_tokens.index_select(0, recycle_pool_idx) if recycle_pool_idx.numel() > 0 else batch_tokens[:0],
                recycled_pool_positions=batch_globals.index_select(0, recycle_pool_idx) if recycle_pool_idx.numel() > 0 else batch_globals[:0],
                recycle_budget=recycle_budget,
                knn_k=self.config.sparsevlm_knn_k,
            )

            merged_tokens = torch.cat([direct_tokens, recycled_tokens.to(dtype=batch_tokens.dtype)], dim=0)
            merged_positions = torch.cat([direct_positions, recycled_positions.to(device=batch_tokens.device)], dim=0)

            deficit = stage_budget - int(merged_tokens.shape[0])
            if deficit > 0:
                refill_idx = deleted_idx[recycle_pool_size : recycle_pool_size + deficit]
                if refill_idx.numel() > 0:
                    merged_tokens = torch.cat([merged_tokens, batch_tokens.index_select(0, refill_idx)], dim=0)
                    merged_positions = torch.cat([merged_positions, batch_globals.index_select(0, refill_idx)], dim=0)

            if merged_tokens.shape[0] > stage_budget:
                merged_tokens = merged_tokens[:stage_budget]
                merged_positions = merged_positions[:stage_budget]

            order = torch.argsort(merged_positions)
            new_visual_tokens.append(merged_tokens.index_select(0, order))
            new_visual_positions.append(merged_positions.index_select(0, order))
            rater_counts.append(int(rater_indices[batch_idx].numel()))
            rank_values.append(float(rank_value))
            soft_delete_values.append(int(budget_info["soft_delete"]))
            recycle_budget_values.append(int(recycle_budget))

        stacked_visual_tokens = torch.stack(new_visual_tokens, dim=0).contiguous()
        stacked_visual_positions = torch.stack(new_visual_positions, dim=0).contiguous()
        keep_counts = self._count_selected_tokens_by_original_view(stacked_visual_positions[0])

        return stacked_visual_tokens, stacked_visual_positions, {
            "raw_total_budget": float(stacked_visual_tokens.shape[1]),
            "final_visual_tokens": int(stacked_visual_tokens.shape[1]),
            "active_pruning_layers": float(len(self.active_stage_map)),
            "sparsevlm_rater_count_mean": float(sum(rater_counts) / max(len(rater_counts), 1)),
            "sparsevlm_rank_mean": float(sum(rank_values) / max(len(rank_values), 1)),
            "sparsevlm_soft_delete_mean": float(sum(soft_delete_values) / max(len(soft_delete_values), 1)),
            "sparsevlm_recycle_budget_mean": float(sum(recycle_budget_values) / max(len(recycle_budget_values), 1)),
            "view_keep_counts": [int(x) for x in keep_counts],
        }

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
                self.runtime_stats["layer0_total_calls"] = int(self.runtime_stats.get("layer0_total_calls", 0) + 1)
                if seq_len <= int(self.config.decode_skip_seq_len):
                    self.runtime_stats["layer0_short_seq_calls"] = int(self.runtime_stats.get("layer0_short_seq_calls", 0) + 1)
                else:
                    self.runtime_stats["layer0_long_seq_calls"] = int(self.runtime_stats.get("layer0_long_seq_calls", 0) + 1)
                if kwargs.get("past_key_value", None) is None:
                    self.runtime_stats["layer0_past_none_calls"] = int(self.runtime_stats.get("layer0_past_none_calls", 0) + 1)
                else:
                    self.runtime_stats["layer0_past_present_calls"] = int(self.runtime_stats.get("layer0_past_present_calls", 0) + 1)
                if self.global_indices is None:
                    self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
                if self.runtime_stats.get("original_prefill_seq_len", 0) <= 0 and seq_len > int(self.config.decode_skip_seq_len):
                    self.runtime_stats["original_prefill_seq_len"] = int(seq_len)

            short_seq_skip = seq_len <= int(self.config.decode_skip_seq_len)
            if layer_idx == 0 or self.global_indices is None:
                self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

            if position_ids is None:
                position_ids = self.global_indices
            else:
                if position_ids.dim() == 2 and position_ids.shape[1] > seq_len:
                    gather_idx = self.global_indices
                    if gather_idx.shape[0] == 1 and batch_size > 1:
                        gather_idx = gather_idx.expand(batch_size, -1)
                    position_ids = position_ids.gather(1, gather_idx)
                elif position_ids.dim() == 3 and position_ids.shape[2] > seq_len:
                    gather_idx = self.global_indices
                    if gather_idx.shape[0] == 1 and batch_size > 1:
                        gather_idx = gather_idx.expand(batch_size, -1)
                    expanded = gather_idx.unsqueeze(0).expand(position_ids.shape[0], -1, -1)
                    position_ids = position_ids.gather(2, expanded)

            if attention_mask is not None and attention_mask.shape[-1] != seq_len:
                attention_mask = None

            if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
                try:
                    cos, sin = kwargs["position_embeddings"]
                    if cos.dim() == 4 and cos.shape[2] > seq_len:
                        gather_idx = self.global_indices
                        if gather_idx.shape[0] == 1 and batch_size > 1:
                            gather_idx = gather_idx.expand(batch_size, -1)
                        expanded = gather_idx.unsqueeze(0).unsqueeze(-1).expand(
                            cos.shape[0], batch_size, gather_idx.shape[1], cos.size(-1)
                        )
                        cos = cos.gather(2, expanded)
                        sin = sin.gather(2, expanded)
                        kwargs["position_embeddings"] = (cos, sin)
                    elif cos.dim() == 3 and cos.shape[1] > seq_len:
                        gather_idx = self.global_indices
                        if gather_idx.shape[0] == 1 and batch_size > 1:
                            gather_idx = gather_idx.expand(batch_size, -1)
                        cos = cos.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, cos.size(-1)))
                        sin = sin.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, sin.size(-1)))
                        kwargs["position_embeddings"] = (cos, sin)
                except Exception:
                    pass

            need_pruning_attn = (not short_seq_skip) and (layer_idx in self.active_stage_map)
            forward_kwargs = dict(kwargs)
            if need_pruning_attn and not original_output_attentions:
                forward_kwargs["output_attentions"] = True

            outputs = self._run_profiled_original_forward(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                forward_kwargs=forward_kwargs,
            )

            def clean_outputs(current_outputs, hidden_override=None):
                if hidden_override is not None:
                    if isinstance(current_outputs, tuple):
                        current_outputs = (hidden_override,) + current_outputs[1:]
                    else:
                        current_outputs = hidden_override
                if original_output_attentions or not need_pruning_attn:
                    return current_outputs
                if not isinstance(current_outputs, tuple):
                    return current_outputs
                if original_use_cache and len(current_outputs) >= 3:
                    return (current_outputs[0], current_outputs[2])
                return (current_outputs[0],) + current_outputs[2:]

            if short_seq_skip:
                return finish(clean_outputs(outputs))

            self._record_prefill_layer_token_count(layer_idx, seq_len)

            if layer_idx < min(self.active_stage_map.keys()):
                return finish(clean_outputs(outputs))

            attn_weights = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
            if attn_weights is None:
                return finish(clean_outputs(outputs))

            layout = self._resolve_layout_state(seq_len, hidden_states.device)
            query_positions = layout["query_positions"]
            visual_positions = layout["visual_positions"]
            always_keep_positions = layout["always_keep_positions"]
            if visual_positions.numel() <= 0:
                return finish(clean_outputs(outputs))

            current_globals = self.global_indices
            if current_globals is None:
                current_globals = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            elif current_globals.shape[0] == 1 and batch_size > 1:
                current_globals = current_globals.expand(batch_size, -1)

            current_layer_output = outputs[0]
            curr_vis_len = int(visual_positions.numel())
            self.runtime_stats["last_tokens_before"] = int(curr_vis_len)

            method = str(self.config.method or "off").lower()
            if method == "fastv":
                # FastV Step 1: run layers 0..K-1 normally. We only enter this branch at layer K.
                # FastV Step 2: use all non-visual tokens as scorers for the joint visual block.
                total_keep = min(int(self.config.visual_tokens), int(curr_vis_len))
                if total_keep >= curr_vis_len:
                    self.runtime_stats["last_tokens_after"] = int(curr_vis_len)
                    self.runtime_stats["raw_total_budget"] = float(curr_vis_len)
                    self.runtime_stats["final_visual_tokens"] = int(curr_vis_len)
                    self.runtime_stats["active_pruning_layers"] = 1.0
                    return finish(clean_outputs(outputs))

                query_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
                query_mask[visual_positions] = False
                all_scores = compute_fastv_attention_scores(
                    attn_weights=attn_weights,
                    visual_positions=visual_positions,
                    query_mask=query_mask,
                )
                kept_visual_idx = select_fastv_keep_indices(all_scores, total_keep)
                visual_base = visual_positions.unsqueeze(0).expand(batch_size, -1)
                selected_visual_positions = visual_base.gather(1, kept_visual_idx)
                topk_elapsed = time.perf_counter() - patch_t0
                full_indices = self._build_full_indices(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    always_keep_positions=always_keep_positions,
                    selected_visual_positions=selected_visual_positions,
                    device=hidden_states.device,
                )
                self.runtime_stats["topk_seconds"] = float(
                    self.runtime_stats.get("topk_seconds", 0.0) + topk_elapsed
                )
                gather_t0 = time.perf_counter()
                expanded = full_indices.unsqueeze(-1).expand(-1, -1, current_layer_output.size(-1))
                pruned_hidden = current_layer_output.gather(1, expanded).contiguous()
                self.global_indices = current_globals.gather(1, full_indices)
                self.runtime_stats["gather_seconds"] = float(
                    self.runtime_stats.get("gather_seconds", 0.0) + (time.perf_counter() - gather_t0)
                )
                self.runtime_stats["last_tokens_after"] = int(total_keep)
                self.runtime_stats["pruned_layers"] = int(self.runtime_stats.get("pruned_layers", 0) + 1)
                self.runtime_stats["raw_total_budget"] = float(total_keep)
                self.runtime_stats["final_visual_tokens"] = int(total_keep)
                self.runtime_stats["active_pruning_layers"] = 1.0
                keep_counts = self._count_selected_tokens_by_original_view(self.global_indices[0][torch.isin(self.global_indices[0], self.input_layout.get("all_visual_indices", torch.empty(0, device=self.global_indices.device)))])
                self.runtime_stats["view_keep_counts"] = [int(x) for x in keep_counts]
                return finish(clean_outputs(outputs, hidden_override=pruned_hidden))

            if method == "sparsevlm":
                # SparseVLM Step 1: select text raters from text-only query tokens.
                # Decode short steps are already excluded above, so these query tokens stay prompt-side.
                stage_idx = int(self.active_stage_map.get(layer_idx, 0))
                remaining_stages = max(1, len(self.active_stage_map) - stage_idx)
                new_visual_tokens, new_visual_positions, sparse_stats = self._run_sparsevlm_joint_pruning(
                    current_layer_output=current_layer_output,
                    attn_weights=attn_weights,
                    visual_positions=visual_positions,
                    text_positions=query_positions,
                    current_globals=current_globals,
                    remaining_stages=remaining_stages,
                )

                visual_start = int(visual_positions.min().item())
                visual_end = int(visual_positions.max().item()) + 1
                prefix_positions = always_keep_positions[always_keep_positions < visual_start]
                suffix_positions = always_keep_positions[always_keep_positions >= visual_end]
                prefix_tokens = self._gather_shared_positions(current_layer_output, prefix_positions)
                suffix_tokens = self._gather_shared_positions(current_layer_output, suffix_positions)
                prefix_globals = current_globals.index_select(1, prefix_positions) if prefix_positions.numel() > 0 else current_globals[:, :0]
                suffix_globals = current_globals.index_select(1, suffix_positions) if suffix_positions.numel() > 0 else current_globals[:, :0]

                gather_t0 = time.perf_counter()
                pruned_hidden = torch.cat([prefix_tokens, new_visual_tokens, suffix_tokens], dim=1)
                self.global_indices = torch.cat([prefix_globals, new_visual_positions, suffix_globals], dim=1)
                self.runtime_stats["gather_seconds"] = float(
                    self.runtime_stats.get("gather_seconds", 0.0) + (time.perf_counter() - gather_t0)
                )
                self.runtime_stats["last_tokens_after"] = int(new_visual_tokens.shape[1])
                self.runtime_stats["pruned_layers"] = int(self.runtime_stats.get("pruned_layers", 0) + 1)
                self.runtime_stats.update(sparse_stats)
                return finish(clean_outputs(outputs, hidden_override=pruned_hidden))

            if method == "pyramiddrop":
                # PyramidDrop Step 2/3: run the LLM by stages and prune visual
                # tokens at each stage boundary.  For VLA, use the prompt-side
                # final instruction token as the query, never the generated
                # action tokens appended later in decode.
                stage_idx = int(self.active_stage_map.get(layer_idx, 0))
                total_stages = max(1, len(self.active_stage_map))
                initial_visual = int(self.runtime_stats.get("pyramiddrop_initial_visual_tokens", 0) or 0)
                if initial_visual <= 0:
                    initial_visual = int(curr_vis_len)
                    self.runtime_stats["pyramiddrop_initial_visual_tokens"] = initial_visual

                lambda_value = float(getattr(self.config, "pyramiddrop_lambda", 0.0) or 0.0)
                if lambda_value <= 0.0:
                    lambda_value = solve_pyramiddrop_lambda(
                        initial_tokens=initial_visual,
                        target_average_tokens=int(self.config.visual_tokens),
                        stages=total_stages,
                    )
                lambda_value = max(0.0, min(1.0, lambda_value))
                target_keep = int(round(float(initial_visual) * (lambda_value ** float(stage_idx + 1))))
                target_keep = max(1, min(int(target_keep), int(curr_vis_len)))
                if stage_idx == total_stages - 1:
                    target_keep = min(target_keep, int(self.config.visual_tokens))
                    target_keep = max(1, min(target_keep, int(curr_vis_len)))

                if target_keep >= curr_vis_len:
                    self.runtime_stats["last_tokens_after"] = int(curr_vis_len)
                    self.runtime_stats["raw_total_budget"] = float(curr_vis_len)
                    self.runtime_stats["final_visual_tokens"] = int(curr_vis_len)
                    self.runtime_stats["active_pruning_layers"] = float(total_stages)
                    self.runtime_stats["pyramiddrop_lambda"] = float(lambda_value)
                    return finish(clean_outputs(outputs))

                if query_positions.numel() > 0:
                    # The input layout records only non-image prompt positions after
                    # the visual block; decode calls are skipped above, so this is
                    # the instruction/generation-prompt tail, not an action token.
                    pdrop_query_positions = query_positions[-1:].contiguous()
                else:
                    pdrop_query_positions = always_keep_positions[-1:].contiguous()
                if pdrop_query_positions.numel() <= 0:
                    return finish(clean_outputs(outputs))

                rel_attn = attn_weights.index_select(2, pdrop_query_positions).index_select(3, visual_positions)
                visual_scores = rel_attn.mean(dim=1).mean(dim=1)
                kept_visual_idx = select_fastv_keep_indices(visual_scores, target_keep)
                visual_base = visual_positions.unsqueeze(0).expand(batch_size, -1)
                selected_visual_positions = visual_base.gather(1, kept_visual_idx)
                full_indices = self._build_full_indices(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    always_keep_positions=always_keep_positions,
                    selected_visual_positions=selected_visual_positions,
                    device=hidden_states.device,
                )
                gather_t0 = time.perf_counter()
                expanded = full_indices.unsqueeze(-1).expand(-1, -1, current_layer_output.size(-1))
                pruned_hidden = current_layer_output.gather(1, expanded).contiguous()
                self.global_indices = current_globals.gather(1, full_indices)
                self.runtime_stats["gather_seconds"] = float(
                    self.runtime_stats.get("gather_seconds", 0.0) + (time.perf_counter() - gather_t0)
                )
                self.runtime_stats["last_tokens_after"] = int(target_keep)
                self.runtime_stats["pruned_layers"] = int(self.runtime_stats.get("pruned_layers", 0) + 1)
                self.runtime_stats["raw_total_budget"] = float(target_keep)
                self.runtime_stats["final_visual_tokens"] = int(target_keep)
                self.runtime_stats["active_pruning_layers"] = float(total_stages)
                self.runtime_stats["pyramiddrop_lambda"] = float(lambda_value)
                return finish(clean_outputs(outputs, hidden_override=pruned_hidden))

            return finish(clean_outputs(outputs))

        return new_forward
