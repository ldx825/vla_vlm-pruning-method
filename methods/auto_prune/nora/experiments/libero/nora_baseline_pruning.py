import math
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv
except Exception:
    apply_multimodal_rotary_pos_emb = None
    repeat_kv = None


@dataclass
class NoraBaselinePruningConfig:
    enabled: bool = False
    method: str = "fastervlm"
    target_token_num: int = 128
    visual_tokens: int = 512
    start_layer: int = 2
    max_pruning_layers: int = 6
    layer_indices: Tuple[int, ...] = ()
    visionzip_dominant_ratio: float = 0.7
    sparsevlm_lambda: float = 1.0
    sparsevlm_recycle_top_ratio: float = 0.5
    sparsevlm_cluster_ratio: float = 0.25
    sparsevlm_rank_tolerance: float = 1e-4
    sparsevlm_knn_k: int = 8
    sparsevlm_min_prune_tokens: int = 1
    action_token_min: int = 151665
    action_token_max: int = 153712
    detailed_timing: bool = True
    debug: bool = False
    disable_kv_cache: bool = False


def _normalize_method(method: str) -> str:
    value = str(method or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "fastervlm": "fastervlm",
        "fastvlm": "fastervlm",
        "visionzip": "visionzip",
        "pyramiddrop": "pyramiddrop",
        "pdrop": "pyramiddrop",
        "fastv": "fastv",
        "sparsevlm": "sparsevlm",
        "sparsevlms": "sparsevlm",
    }
    if value not in aliases:
        raise ValueError(f"Unknown NORA baseline pruning method: {method!r}")
    return aliases[value]


def _runtime_stats(method: str, target_token_num: int) -> Dict[str, Any]:
    return {
        "baseline_method": method,
        "prefill_seq_len": None,
        "schedule_len": 0,
        "pruned_layers": 0,
        "last_tokens_before": 0,
        "last_tokens_after": 0,
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
        "topk_seconds": 0.0,
        "gather_seconds": 0.0,
        "compaction_bookkeeping_seconds": 0.0,
        "layout_ready": 0,
        "layout_query_tokens": 0,
        "layout_view0_tokens": 0,
        "layout_view1_tokens": 0,
        "flops_reduction": 0.0,
        "final_visual_tokens": 0.0,
        "raw_total_budget": float(target_token_num),
        "active_pruning_layers": 0.0,
        "method_overhead_seconds": 0.0,
        "unified_visual_pruning": 1,
    }


def _clamp_keep(target: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(1, min(int(target), int(total)))


def _estimate_flops_reduction(prefill_len: int, before_visual: int, after_visual: int) -> float:
    dense = max(1.0, float(prefill_len) ** 2)
    pruned_len = max(1.0, float(prefill_len - max(0, before_visual - after_visual)))
    return max(0.0, min(1.0, 1.0 - (pruned_len * pruned_len) / dense))


def _scores_from_visual_outputs(
    image_embeds: torch.Tensor,
    attn_scores: Optional[torch.Tensor],
) -> torch.Tensor:
    total = int(image_embeds.shape[0])
    if attn_scores is not None:
        score = attn_scores.float().reshape(-1)
        if score.numel() >= total:
            return torch.nan_to_num(score[:total], nan=0.0, posinf=0.0, neginf=0.0)
    # Qwen2.5-VL vision backbones do not expose a CLIP-style CLS token in the
    # stock HF forward. When the patched vision forward cannot expose
    # [CLS]->patch attention, this fallback ranks by final visual feature energy
    # but keeps the same prefill-before-projector semantics.
    return torch.nan_to_num(image_embeds.float().norm(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)


def _fastervlm_select(image_embeds: torch.Tensor, scores: torch.Tensor, target: int) -> Tuple[torch.Tensor, torch.Tensor]:
    total = int(image_embeds.shape[0])
    keep = _clamp_keep(target, total)
    if keep >= total:
        idx = torch.arange(total, device=image_embeds.device)
        return image_embeds, idx
    # FasterVLM's paper parameter is reduction ratio R. The experiment passes
    # final keep count, so R is implicitly 1 - keep / total; top-k implements
    # the threshold induced by that R.
    _, idx = torch.topk(scores.to(image_embeds.device), keep, dim=0)
    idx = torch.sort(idx).values
    return image_embeds.index_select(0, idx).contiguous(), idx


def _contextual_tokens(features: torch.Tensor, remaining_idx: torch.Tensor, contextual_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if contextual_num <= 0 or remaining_idx.numel() <= 0:
        return features[:0], remaining_idx[:0]
    contextual_num = min(int(contextual_num), int(remaining_idx.numel()))
    remaining = features.index_select(0, remaining_idx)
    if contextual_num >= remaining.shape[0]:
        return remaining, remaining_idx

    step = max(1, remaining.shape[0] // contextual_num)
    anchor_local = torch.arange(0, remaining.shape[0], step, device=features.device)[:contextual_num]
    anchor_tokens = remaining.index_select(0, anchor_local)
    rest_mask = torch.ones(remaining.shape[0], dtype=torch.bool, device=features.device)
    rest_mask[anchor_local] = False
    rest = remaining[rest_mask]
    if rest.numel() == 0:
        return anchor_tokens, remaining_idx.index_select(0, anchor_local)

    rest_norm = F.normalize(rest.float(), dim=-1).unsqueeze(0)
    anchor_norm = F.normalize(anchor_tokens.float(), dim=-1).unsqueeze(0)
    assign = torch.bmm(rest_norm, anchor_norm.transpose(1, 2)).argmax(dim=-1)
    one_hot = torch.zeros(1, rest.shape[0], contextual_num, device=features.device, dtype=features.dtype)
    one_hot.scatter_(2, assign.unsqueeze(-1), 1)
    counts = one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    merged = torch.bmm(one_hot.transpose(1, 2), rest.unsqueeze(0)) / counts
    contextual = anchor_tokens.unsqueeze(0) + merged
    return contextual.squeeze(0).to(features.dtype), remaining_idx.index_select(0, anchor_local)


def _visionzip_select(
    image_embeds: torch.Tensor,
    scores: torch.Tensor,
    target: int,
    dominant_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    total = int(image_embeds.shape[0])
    keep = _clamp_keep(target, total)
    if keep >= total:
        idx = torch.arange(total, device=image_embeds.device)
        return image_embeds, idx, total, 0

    # VisionZip final count is K + M + 1 in the paper. In Qwen/NORA the visual
    # sequence after the vision merger has no standalone CLS token, so the
    # experiment-facing keep count is split as K/M over the real visual tokens.
    dominant_num = max(1, min(keep, int(round(keep * float(dominant_ratio)))))
    contextual_num = max(0, keep - dominant_num)
    _, dominant_idx = torch.topk(scores.to(image_embeds.device), dominant_num, dim=0)
    dominant_idx = torch.sort(dominant_idx).values
    all_idx = torch.arange(total, device=image_embeds.device)
    remain_mask = torch.ones(total, dtype=torch.bool, device=image_embeds.device)
    remain_mask[dominant_idx] = False
    contextual, contextual_idx = _contextual_tokens(image_embeds, all_idx[remain_mask], contextual_num)
    pruned = torch.cat([image_embeds.index_select(0, dominant_idx), contextual], dim=0).contiguous()
    keep_idx = torch.cat([dominant_idx, contextual_idx], dim=0)
    return pruned, keep_idx, int(dominant_num), int(contextual.shape[0])


def _solve_lambda(initial_tokens: int, target_avg_tokens: int, stages: int) -> float:
    initial_tokens = max(1, int(initial_tokens))
    target_avg_tokens = max(1, min(int(target_avg_tokens), initial_tokens))
    stages = max(1, int(stages))
    if target_avg_tokens >= initial_tokens:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(48):
        mid = (lo + hi) * 0.5
        avg = sum(initial_tokens * (mid ** (idx + 1)) for idx in range(stages)) / float(stages)
        if avg < target_avg_tokens:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _pyramiddrop_schedule(initial_tokens: int, target_avg_tokens: int, stages: int) -> Tuple[List[int], float]:
    lam = _solve_lambda(initial_tokens, target_avg_tokens, stages)
    current = int(initial_tokens)
    counts: List[int] = []
    for idx in range(max(1, int(stages))):
        count = int(round(initial_tokens * (lam ** (idx + 1))))
        count = max(1, min(current, count))
        counts.append(count)
        current = count
    return counts, lam


def _effective_rank(tokens: torch.Tensor, tolerance: float) -> torch.Tensor:
    if tokens.numel() == 0 or tokens.shape[1] <= 1:
        return tokens.new_tensor(1.0)
    x = tokens[0].float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        sigmas = torch.linalg.svdvals(x)
    except Exception:
        return tokens.new_tensor(float(max(1, min(tokens.shape[1], tokens.shape[-1]))))
    sigmas = sigmas[sigmas > float(tolerance)]
    if sigmas.numel() == 0:
        return tokens.new_tensor(1.0)
    probs = sigmas / sigmas.sum().clamp(min=1e-8)
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum()
    return torch.exp(entropy).to(device=tokens.device, dtype=tokens.dtype)


def _cluster_and_merge(tokens: torch.Tensor, cluster_num: int, knn_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, token_num, dim = tokens.shape
    cluster_num = max(1, min(int(cluster_num), int(token_num)))
    if token_num <= cluster_num:
        reps = torch.arange(token_num, dtype=torch.long, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        return tokens, reps

    normalized = F.normalize(tokens.float(), dim=-1)
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

    center_tokens = tokens.gather(1, centers.unsqueeze(-1).expand(-1, -1, dim)).float()
    assign_dist = torch.cdist(tokens.float(), center_tokens, p=2)
    assignments = assign_dist.argmin(dim=-1)
    batch_idx = torch.arange(bsz, device=tokens.device)[:, None]
    assignments[batch_idx.expand_as(centers), centers] = torch.arange(cluster_num, device=tokens.device)[None, :]

    merged = tokens.new_zeros(bsz, cluster_num, dim)
    weights = tokens.new_zeros(bsz, cluster_num, 1)
    merged.scatter_add_(1, assignments.unsqueeze(-1).expand(-1, -1, dim), tokens)
    weights.scatter_add_(1, assignments.unsqueeze(-1), tokens.new_ones(bsz, token_num, 1))
    merged = merged / weights.clamp(min=1.0)
    return merged, centers


class NoraBaselinePruningPatch:
    def __init__(self, model, config: NoraBaselinePruningConfig):
        self.model = model
        self.config = config
        self.method = _normalize_method(config.method)
        self.layers = self._resolve_layers(model)
        self.original_forward = None
        self.original_layer_forwards: Dict[int, Any] = {}
        self.global_indices: Optional[torch.Tensor] = None
        self.input_layout: Dict[str, Any] = {}
        self.keep_schedule: List[int] = []
        self.pruning_layers = self._build_pruning_layers()
        self.reset_runtime_state()

    @staticmethod
    def _resolve_layers(model):
        for path in (
            ("model", "layers"),
            ("language_model", "model", "layers"),
            ("language_model", "layers"),
            ("base_model", "model", "model", "layers"),
            ("base_model", "model", "layers"),
        ):
            obj = model
            for attr in path:
                if not hasattr(obj, attr):
                    obj = None
                    break
                obj = getattr(obj, attr)
            if isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
                return obj
        raise RuntimeError("Unable to locate decoder layers for NORA baseline pruning.")

    def _build_pruning_layers(self) -> List[int]:
        if self.method == "fastv":
            # FastV paper uses K as "run first K layers, prune after layer K".
            # Externally we keep that 1-based convention for FastV.
            idx = max(0, min(int(self.config.start_layer) - 1, len(self.layers) - 1))
            return [idx]
        if self.config.layer_indices:
            return [int(idx) for idx in self.config.layer_indices if 0 <= int(idx) < len(self.layers)]
        candidates = list(range(max(0, int(self.config.start_layer)), len(self.layers)))
        max_layers = int(self.config.max_pruning_layers)
        if max_layers <= 0 or max_layers >= len(candidates):
            return candidates
        if max_layers == 1:
            return candidates[:1]
        last = len(candidates) - 1
        selected = [candidates[int(round(offset * last / max(1, max_layers - 1)))] for offset in range(max_layers)]
        return sorted(set(selected))

    def reset_runtime_state(self) -> None:
        self.global_indices = None
        self.input_layout = {}
        self.keep_schedule = []
        self.runtime_stats = _runtime_stats(self.method, self.config.target_token_num)
        self.runtime_stats["active_pruning_layers"] = float(len(self.pruning_layers))
        self.runtime_stats["active_pruning_layer_indices"] = [int(idx) for idx in self.pruning_layers]

    def apply(self) -> None:
        if not self.config.enabled:
            return
        self.original_forward = self.model.forward
        self.model.forward = types.MethodType(self._make_model_forward(), self.model)
        if self.method == "pyramiddrop":
            for idx, layer in enumerate(self.layers):
                self.original_layer_forwards[idx] = layer.forward
                layer.forward = types.MethodType(self._make_layer_forward(idx), layer)
        if self.method in {"fastv", "sparsevlm"}:
            for idx, layer in enumerate(self.layers):
                self.original_layer_forwards[idx] = layer.forward
                layer.forward = types.MethodType(self._make_layer_forward(idx), layer)
        self.model._nora_baseline_pruning_patch = self
        self.model._autoprune_patch = self
        if self.config.disable_kv_cache:
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.use_cache = False

    def prepare_input_layout(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
    ) -> None:
        self.input_layout = {}
        if input_ids is None or image_token_id is None or input_ids.ndim != 2 or input_ids.shape[0] < 1:
            return
        row = input_ids[0]
        valid = attention_mask[0].bool() if attention_mask is not None else torch.ones_like(row, dtype=torch.bool)
        valid_pos = torch.nonzero(valid, as_tuple=False).flatten()
        valid_row = row[valid]
        image_mask = valid_row == int(image_token_id)
        visual_positions = valid_pos[image_mask]
        if visual_positions.numel() <= 0:
            return
        non_visual = valid_pos[~image_mask]
        after_visual = non_visual[non_visual > int(visual_positions.max().item())]
        action_mask = (row[after_visual] >= int(self.config.action_token_min)) & (
            row[after_visual] <= int(self.config.action_token_max)
        )
        query_positions = after_visual[~action_mask]
        if query_positions.numel() <= 0:
            query_positions = non_visual[~((row[non_visual] >= int(self.config.action_token_min)) & (row[non_visual] <= int(self.config.action_token_max)))]
        keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        keep_mask[visual_positions] = False
        self.input_layout = {
            "ready": True,
            "visual_positions": visual_positions.contiguous(),
            "always_keep_positions": torch.nonzero(keep_mask, as_tuple=False).flatten(),
            "query_positions": query_positions.contiguous(),
        }
        self.runtime_stats["layout_ready"] = 1
        self.runtime_stats["layout_query_tokens"] = int(query_positions.numel())
        self.runtime_stats["layout_view0_tokens"] = int(visual_positions.numel())
        self.runtime_stats["layout_view1_tokens"] = 0

    def _locate_current_positions(self, original_positions: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.global_indices is None or original_positions is None or original_positions.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        current = self.global_indices[0].to(device=device)
        positions = original_positions.to(device=device, dtype=current.dtype)
        return torch.nonzero(torch.isin(current, positions), as_tuple=False).flatten()

    def _layout_state(self, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if self.input_layout.get("ready"):
            visual = self._locate_current_positions(self.input_layout["visual_positions"], device)
            always = self._locate_current_positions(self.input_layout["always_keep_positions"], device)
            query = self._locate_current_positions(self.input_layout["query_positions"], device)
            return {"visual_positions": visual, "always_keep_positions": always, "query_positions": query}
        vis_start = 1
        vis_end = min(1 + int(self.config.visual_tokens), seq_len - 1)
        visual = torch.arange(vis_start, vis_end, dtype=torch.long, device=device)
        always = torch.cat(
            [
                torch.arange(0, vis_start, dtype=torch.long, device=device),
                torch.arange(vis_end, seq_len, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        query = torch.arange(vis_end, seq_len, dtype=torch.long, device=device)
        return {"visual_positions": visual, "always_keep_positions": always, "query_positions": query}

    @staticmethod
    def _build_full_indices(batch_size: int, seq_len: int, always_keep: torch.Tensor, selected_visual: torch.Tensor, device) -> torch.Tensor:
        keep = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        if always_keep.numel() > 0:
            keep[:, always_keep] = True
        if selected_visual.numel() > 0:
            keep.scatter_(1, selected_visual, True)
        base = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        return base[keep].view(batch_size, -1)

    @staticmethod
    def _align_position_ids(global_indices, position_ids, batch_size: int, seq_len: int):
        if global_indices is None:
            return position_ids
        gather_idx = global_indices
        if gather_idx.shape[0] == 1 and batch_size > 1:
            gather_idx = gather_idx.expand(batch_size, -1)
        if position_ids is None:
            return gather_idx
        if position_ids.dim() == 2 and position_ids.shape[1] != seq_len:
            return position_ids.gather(1, gather_idx)
        if position_ids.dim() == 3 and position_ids.shape[2] != seq_len:
            expanded = gather_idx.unsqueeze(0).expand(position_ids.shape[0], -1, -1)
            return position_ids.gather(2, expanded)
        return position_ids

    @staticmethod
    def _align_position_embeddings(global_indices, kwargs: Dict[str, Any], batch_size: int, seq_len: int) -> None:
        if global_indices is None or kwargs.get("position_embeddings", None) is None:
            return
        try:
            gather_idx = global_indices
            if gather_idx.shape[0] == 1 and batch_size > 1:
                gather_idx = gather_idx.expand(batch_size, -1)
            cos, sin = kwargs["position_embeddings"]
            if cos.dim() == 4 and cos.shape[2] != seq_len:
                expanded = gather_idx.unsqueeze(0).unsqueeze(-1).expand(cos.shape[0], batch_size, gather_idx.shape[1], cos.size(-1))
                kwargs["position_embeddings"] = (cos.gather(2, expanded), sin.gather(2, expanded))
            elif cos.dim() == 3 and cos.shape[1] != seq_len:
                expanded = gather_idx.unsqueeze(-1).expand(-1, -1, cos.size(-1))
                kwargs["position_embeddings"] = (cos.gather(1, expanded), sin.gather(1, expanded))
        except Exception:
            pass

    def _make_model_forward(self):
        def new_forward(model_self, *args, **kwargs):
            if self.method not in {"fastervlm", "visionzip"}:
                input_ids = kwargs.get("input_ids", args[0] if args else None)
                attention_mask = kwargs.get("attention_mask", None)
                if input_ids is not None and input_ids.ndim == 2 and input_ids.shape[1] > 1:
                    self.reset_runtime_state()
                    self.prepare_input_layout(input_ids, attention_mask, int(model_self.config.image_token_id))
                return self.original_forward(*args, **kwargs)

            input_ids = kwargs.get("input_ids", args[0] if args else None)
            pixel_values = kwargs.get("pixel_values", None)
            attention_mask = kwargs.get("attention_mask", None)
            past_key_values = kwargs.get("past_key_values", None)
            is_prefill = (
                input_ids is not None
                and input_ids.ndim == 2
                and input_ids.shape[1] > 1
                and pixel_values is not None
                and (past_key_values is None or not hasattr(past_key_values, "get_seq_length") or past_key_values.get_seq_length() == 0)
            )
            if not is_prefill or kwargs.get("inputs_embeds", None) is not None:
                return self.original_forward(*args, **kwargs)

            patch_t0 = time.perf_counter()
            self.reset_runtime_state()
            self.prepare_input_layout(input_ids, attention_mask, int(model_self.config.image_token_id))
            if not self.input_layout.get("ready"):
                return self.original_forward(*args, **kwargs)

            pixel_values = pixel_values.type(model_self.visual.dtype)
            visual_t0 = time.perf_counter()
            visual_out = model_self.visual(pixel_values, grid_thw=kwargs.get("image_grid_thw", None))
            self.runtime_stats["regular_forward_seconds"] += time.perf_counter() - visual_t0
            if isinstance(visual_out, tuple):
                image_embeds = visual_out[0]
                attn_scores = visual_out[1] if len(visual_out) > 1 else None
            else:
                image_embeds = visual_out
                attn_scores = None

            n_image_tokens = int((input_ids == model_self.config.image_token_id).sum().item())
            if n_image_tokens != int(image_embeds.shape[0]):
                return self.original_forward(*args, **kwargs)

            score = _scores_from_visual_outputs(image_embeds, attn_scores)
            select_t0 = time.perf_counter()
            if self.method == "fastervlm":
                pruned_embeds, keep_visual_idx = _fastervlm_select(image_embeds, score, self.config.target_token_num)
                dominant_tokens = int(pruned_embeds.shape[0])
                contextual_tokens = 0
            else:
                pruned_embeds, keep_visual_idx, dominant_tokens, contextual_tokens = _visionzip_select(
                    image_embeds,
                    score,
                    self.config.target_token_num,
                    self.config.visionzip_dominant_ratio,
                )
            # Keep the physical sequence ordered by the retained visual-token
            # positions so attention masks, mROPE position_ids, and embeddings
            # all describe the same shortened sequence.
            if keep_visual_idx.numel() > 0:
                order = torch.argsort(keep_visual_idx)
                keep_visual_idx = keep_visual_idx.index_select(0, order)
                pruned_embeds = pruned_embeds.index_select(0, order)
            self.runtime_stats["topk_seconds"] += time.perf_counter() - select_t0

            gather_t0 = time.perf_counter()
            image_positions = self.input_layout["visual_positions"].to(device=input_ids.device)
            keep_visual_positions = image_positions.index_select(0, keep_visual_idx.to(device=image_positions.device))
            keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
            keep_mask[image_positions] = False
            keep_mask[keep_visual_positions] = True
            full_indices = torch.nonzero(keep_mask, as_tuple=False).flatten().unsqueeze(0)

            inputs_embeds = model_self.model.embed_tokens(input_ids)
            image_mask = input_ids == model_self.config.image_token_id
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[image_mask] = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            compact_embeds = inputs_embeds.gather(1, full_indices.unsqueeze(-1).expand(-1, -1, inputs_embeds.size(-1))).contiguous()
            compact_input_ids = input_ids.gather(1, full_indices)
            compact_attention = attention_mask.gather(1, full_indices) if attention_mask is not None else None
            labels = kwargs.get("labels", None)
            compact_labels = labels.gather(1, full_indices) if labels is not None and labels.dim() == 2 else labels
            cache_position = kwargs.get("cache_position", None)
            compact_cache_position = cache_position
            if isinstance(cache_position, torch.Tensor) and cache_position.dim() == 1 and cache_position.numel() == input_ids.shape[1]:
                compact_cache_position = cache_position.index_select(0, full_indices[0].to(cache_position.device))

            compact_image_mask = compact_input_ids == model_self.config.image_token_id
            compact_embeds[compact_image_mask] = pruned_embeds.to(compact_embeds.device, compact_embeds.dtype)

            position_ids = kwargs.get("position_ids", None)
            if position_ids is None and hasattr(model_self, "get_rope_index"):
                position_ids, rope_deltas = model_self.get_rope_index(
                    input_ids,
                    kwargs.get("image_grid_thw", None),
                    kwargs.get("video_grid_thw", None),
                    kwargs.get("second_per_grid_ts", None),
                    attention_mask,
                )
                kwargs["rope_deltas"] = rope_deltas
            if position_ids is not None:
                if position_ids.dim() == 2:
                    compact_position_ids = position_ids.gather(1, full_indices)
                elif position_ids.dim() == 3:
                    compact_position_ids = position_ids.gather(2, full_indices.unsqueeze(0).expand(position_ids.shape[0], -1, -1))
                else:
                    compact_position_ids = position_ids
            else:
                base_mask = compact_attention if compact_attention is not None else torch.ones_like(compact_input_ids)
                compact_position_ids = base_mask.long().cumsum(-1) - 1
                compact_position_ids.masked_fill_(base_mask == 0, 1)
                compact_position_ids = compact_position_ids.unsqueeze(0).expand(3, -1, -1)

            self.runtime_stats["gather_seconds"] += time.perf_counter() - gather_t0
            before = int(image_positions.numel())
            after = int(pruned_embeds.shape[0])
            self.runtime_stats["prefill_seq_len"] = int(input_ids.shape[1])
            self.runtime_stats["last_tokens_before"] = before
            self.runtime_stats["last_tokens_after"] = after
            self.runtime_stats["final_visual_tokens"] = float(after)
            self.runtime_stats["raw_total_budget"] = float(self.config.target_token_num)
            self.runtime_stats["pruned_layers"] = 1
            self.runtime_stats["dominant_tokens"] = int(dominant_tokens)
            self.runtime_stats["contextual_tokens"] = int(contextual_tokens)
            self.runtime_stats["reduction_ratio"] = 1.0 - float(after) / float(max(1, before))
            self.runtime_stats["flops_reduction"] = _estimate_flops_reduction(int(input_ids.shape[1]), before, after)
            self.runtime_stats["method_overhead_seconds"] = self.runtime_stats["topk_seconds"] + self.runtime_stats["gather_seconds"]
            self.runtime_stats["patch_total_seconds"] += time.perf_counter() - patch_t0

            next_kwargs = dict(kwargs)
            next_kwargs.update(
                {
                    "input_ids": compact_input_ids,
                    "attention_mask": compact_attention,
                    "position_ids": compact_position_ids,
                    "inputs_embeds": compact_embeds,
                    "pixel_values": None,
                    "image_grid_thw": None,
                    "labels": compact_labels,
                    "cache_position": compact_cache_position,
                }
            )
            if self.config.debug:
                print(f"[NoraBaseline:{self.method}] visual tokens {before} -> {after}")
            return self.original_forward(**next_kwargs)

        return new_forward

    def _score_visual_tokens(self, layer, hidden_states, attention_mask, position_ids, kwargs, visual_positions, query_pos):
        with torch.no_grad():
            normed = layer.input_layernorm(hidden_states)
            self_attn = layer.self_attn
            q = self_attn.q_proj(normed)
            k = self_attn.k_proj(normed)
            bsz, seq_len, _ = hidden_states.shape
            num_heads = self_attn.num_heads
            num_kv_heads = self_attn.num_key_value_heads
            head_dim = self_attn.head_dim
            q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            position_embeddings = kwargs.get("position_embeddings", None)
            if apply_multimodal_rotary_pos_emb is not None and position_embeddings is not None:
                try:
                    cos, sin = position_embeddings
                    q, k = apply_multimodal_rotary_pos_emb(
                        q,
                        k,
                        cos,
                        sin,
                        self_attn.rope_scaling["mrope_section"],
                    )
                except Exception:
                    pass
            if repeat_kv is not None and num_kv_heads != num_heads:
                k = repeat_kv(k, num_heads // max(1, num_kv_heads))
            elif num_kv_heads != num_heads:
                k = k.repeat_interleave(max(1, num_heads // max(1, num_kv_heads)), dim=1)
            batch = torch.arange(bsz, device=hidden_states.device)
            q_text = q[batch, :, query_pos, :].unsqueeze(2)
            k_vis = k.index_select(2, visual_positions)
            score = torch.matmul(q_text, k_vis.transpose(-1, -2)).squeeze(2).mean(dim=1) / math.sqrt(head_dim)
            if attention_mask is not None and attention_mask.dim() == 2:
                valid_visual = attention_mask.index_select(1, visual_positions).bool()
                score = score.masked_fill(~valid_visual, torch.finfo(score.dtype).min)
            return torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)

    def _select_sparsevlm_raters(self, hidden_states: torch.Tensor, visual_positions: torch.Tensor, text_positions: torch.Tensor) -> torch.Tensor:
        # SparseVLM Step 1: select text raters that are truly image-related.
        # Only non-action instruction/text tokens are allowed here. This differs
        # from FastV, whose attention score intentionally includes action tokens.
        if visual_positions.numel() <= 0 or text_positions.numel() <= 0:
            return text_positions
        visual_tokens = hidden_states[:, visual_positions, :]
        text_tokens = hidden_states[:, text_positions, :]
        affinity = torch.matmul(visual_tokens.float(), text_tokens.float().transpose(1, 2))
        relevance = affinity.softmax(dim=-1).mean(dim=1)[0]
        keep = relevance > relevance.mean()
        raters = text_positions[keep]
        if raters.numel() <= 0:
            raters = text_positions[relevance.topk(1).indices]
        return raters.contiguous()

    @staticmethod
    def _attention_scores_from_raters(attn_weights: torch.Tensor, visual_positions: torch.Tensor, rater_positions: torch.Tensor) -> torch.Tensor:
        # attn_weights: [B, Heads, Query, Key]. Average heads and selected raters.
        if visual_positions.numel() <= 0 or rater_positions.numel() <= 0:
            return torch.empty((attn_weights.shape[0], 0), dtype=attn_weights.dtype, device=attn_weights.device)
        relation = attn_weights[:, :, rater_positions, :][:, :, :, visual_positions]
        return torch.nan_to_num(relation.mean(dim=1).mean(dim=1).float(), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _build_full_indices_from_visual(batch_size: int, seq_len: int, visual_positions: torch.Tensor, selected_visual: torch.Tensor, device) -> torch.Tensor:
        keep = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        if visual_positions.numel() > 0:
            keep[:, visual_positions] = False
        if selected_visual.numel() > 0:
            keep.scatter_(1, selected_visual, True)
        base = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        return base[keep].view(batch_size, -1)

    def _sparsevlm_counts(self, visual_tokens: torch.Tensor, layer_idx: int) -> Tuple[int, int, int, float, float]:
        current_v = int(visual_tokens.shape[1])
        target = max(1, min(int(self.config.target_token_num), current_v))
        remaining_layers = max(1, len([idx for idx in self.pruning_layers if idx >= layer_idx]))
        rank = _effective_rank(visual_tokens, float(self.config.sparsevlm_rank_tolerance))
        rank_value = float(rank.detach().cpu().item())
        redundancy = max(0.0, min(1.0, 1.0 - rank_value / float(max(1, current_v))))
        # SparseVLM Step 3: adaptive deletion from rank/redundancy, then a
        # budget-constrained projection so experiments can sweep final budgets.
        delete_count = int(round(float(self.config.sparsevlm_lambda) * redundancy * current_v))
        if current_v > target:
            min_progress = max(
                int(self.config.sparsevlm_min_prune_tokens),
                int(math.ceil((current_v - target) / float(remaining_layers))),
            )
            delete_count = max(delete_count, min_progress)
        delete_count = max(0, min(delete_count, max(0, current_v - target)))
        rescue_count = max(0, min(delete_count, int(round(delete_count * float(self.config.sparsevlm_recycle_top_ratio)))))
        cluster_count = max(0, min(rescue_count, int(round(rescue_count * float(self.config.sparsevlm_cluster_ratio)))))
        keep_direct = current_v - delete_count
        return keep_direct, rescue_count, cluster_count, rank_value, redundancy

    def _make_layer_forward(self, layer_idx: int):
        def new_forward(layer_self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            patch_t0 = time.perf_counter()
            batch_size, seq_len, _ = hidden_states.shape
            is_decode = seq_len <= 96
            if layer_idx == 0:
                if not is_decode or self.global_indices is None:
                    self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
                if self.runtime_stats.get("prefill_seq_len") is None and seq_len > 1:
                    self.runtime_stats["prefill_seq_len"] = int(seq_len)

            if not is_decode:
                position_ids = self._align_position_ids(self.global_indices, position_ids, batch_size, seq_len)
                if attention_mask is not None and attention_mask.shape[-1] != seq_len:
                    attention_mask = None
                self._align_position_embeddings(self.global_indices, kwargs, batch_size, seq_len)

            original_output_attentions = bool(kwargs.get("output_attentions", False))
            need_attention = (
                not is_decode
                and self.method in {"fastv", "sparsevlm"}
                and layer_idx in self.pruning_layers
            )
            original_use_cache = bool(kwargs.get("use_cache", False))
            kwargs["output_attentions"] = bool(original_output_attentions or need_attention)

            forward_t0 = time.perf_counter()
            outputs = self.original_layer_forwards[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
            self.runtime_stats["regular_forward_seconds"] += time.perf_counter() - forward_t0
            self.runtime_stats["patch_total_seconds"] += time.perf_counter() - patch_t0

            def clean_outputs(hidden_override=None):
                hidden = outputs[0] if hidden_override is None else hidden_override
                if original_output_attentions:
                    return (hidden,) + tuple(outputs[1:])
                if not need_attention:
                    return outputs if hidden_override is None else (hidden,) + tuple(outputs[1:])
                if original_use_cache and len(outputs) >= 3:
                    return (hidden, outputs[2])
                return (hidden,)

            if is_decode or layer_idx not in self.pruning_layers:
                return clean_outputs()

            layout = self._layout_state(seq_len, hidden_states.device)
            visual_positions = layout["visual_positions"]
            always_keep = layout["always_keep_positions"]
            query_positions = layout["query_positions"]
            if visual_positions.numel() <= 0:
                return clean_outputs()

            if self.method == "fastv":
                # FastV Step 1/2: run the first K layers normally; at layer K,
                # score each visual token by attention from every other token in
                # the current sequence. This includes action tokens, as requested.
                attn_weights = outputs[1] if len(outputs) > 1 else None
                if attn_weights is None:
                    return clean_outputs()
                score_t0 = time.perf_counter()
                all_query = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
                all_query[visual_positions] = False
                query_all = torch.nonzero(all_query, as_tuple=False).flatten()
                score = self._attention_scores_from_raters(attn_weights, visual_positions, query_all)
                keep_count = _clamp_keep(int(self.config.target_token_num), int(visual_positions.numel()))
                if keep_count >= int(visual_positions.numel()):
                    self.runtime_stats["last_tokens_before"] = int(visual_positions.numel())
                    self.runtime_stats["last_tokens_after"] = int(visual_positions.numel())
                    return clean_outputs()
                _, topk = torch.topk(score, keep_count, dim=1)
                topk = torch.sort(topk, dim=1).values
                selected = visual_positions.unsqueeze(0).expand(batch_size, -1).gather(1, topk)
                self.runtime_stats["topk_seconds"] += time.perf_counter() - score_t0

                # FastV Step 3/4: convert paper ratio R into a final keep count
                # and physically compact the sequence for all later layers.
                gather_t0 = time.perf_counter()
                full_indices = self._build_full_indices_from_visual(batch_size, seq_len, visual_positions, selected, hidden_states.device)
                new_hidden = outputs[0].gather(1, full_indices.unsqueeze(-1).expand(-1, -1, outputs[0].size(-1))).contiguous()
                if self.global_indices is not None:
                    current = self.global_indices
                    if current.shape[0] == 1 and batch_size > 1:
                        current = current.expand(batch_size, -1)
                    self.global_indices = current.gather(1, full_indices)
                self.runtime_stats["gather_seconds"] += time.perf_counter() - gather_t0
                before = int(visual_positions.numel())
                after = int(keep_count)
                self.runtime_stats["last_tokens_before"] = before
                self.runtime_stats["last_tokens_after"] = after
                self.runtime_stats["final_visual_tokens"] = float(after)
                self.runtime_stats["reduction_ratio"] = 1.0 - float(after) / float(max(1, before))
                self.runtime_stats["pruned_layers"] = 1
                self.runtime_stats["method_overhead_seconds"] = self.runtime_stats["topk_seconds"] + self.runtime_stats["gather_seconds"]
                self.runtime_stats["flops_reduction"] = _estimate_flops_reduction(seq_len, before, after)
                if self.config.debug:
                    print(f"[NoraBaseline:fastv] layer={layer_idx} visual {before}->{after}, seq {seq_len}->{full_indices.shape[1]}")
                return clean_outputs(hidden_override=new_hidden)

            if self.method == "sparsevlm":
                # SparseVLM Step 1: use only text/instruction tokens as raters.
                # `query_positions` was prepared from input_ids with action token
                # ids excluded, so action tokens are not mixed into SparseVLM's
                # rater set.
                attn_weights = outputs[1] if len(outputs) > 1 else None
                if attn_weights is None or query_positions.numel() <= 0:
                    return clean_outputs()
                rater_t0 = time.perf_counter()
                raters = self._select_sparsevlm_raters(outputs[0], visual_positions, query_positions)
                self.runtime_stats["metric_seconds"] += time.perf_counter() - rater_t0
                if raters.numel() <= 0:
                    return clean_outputs()

                # SparseVLM Step 2: text raters score visual tokens through
                # decoder attention.
                score_t0 = time.perf_counter()
                score = self._attention_scores_from_raters(attn_weights, visual_positions, raters)
                visual_tokens = outputs[0].index_select(1, visual_positions)
                keep_direct, rescue_count, cluster_count, rank_value, redundancy = self._sparsevlm_counts(visual_tokens, layer_idx)
                self.runtime_stats["effective_rank"] = float(rank_value)
                self.runtime_stats["redundancy"] = float(redundancy)
                self.runtime_stats["rank_seconds"] += time.perf_counter() - score_t0
                if keep_direct >= int(visual_positions.numel()) and cluster_count <= 0:
                    self.runtime_stats["last_tokens_before"] = int(visual_positions.numel())
                    self.runtime_stats["last_tokens_after"] = int(visual_positions.numel())
                    return clean_outputs()

                # SparseVLM Step 3/4: keep high-priority visual tokens, rescue a
                # subset of deleted tokens, cluster/merge rescued tokens, then
                # rebuild a physically shorter sequence.
                gather_t0 = time.perf_counter()
                order = torch.argsort(score, dim=1, descending=True)
                keep_local = torch.sort(order[:, :keep_direct], dim=1).values
                deleted_local = order[:, keep_direct:]
                kept_visual = visual_positions.unsqueeze(0).expand(batch_size, -1).gather(1, keep_local)
                kept_hidden = outputs[0].gather(1, kept_visual.unsqueeze(-1).expand(-1, -1, outputs[0].size(-1)))
                current = self.global_indices
                if current is not None and current.shape[0] == 1 and batch_size > 1:
                    current = current.expand(batch_size, -1)
                kept_global = current.gather(1, kept_visual) if current is not None else kept_visual

                merged_hidden = outputs[0].new_empty(batch_size, 0, outputs[0].size(-1))
                merged_global = kept_visual.new_empty(batch_size, 0)
                if deleted_local.shape[1] > 0 and rescue_count > 0 and cluster_count > 0:
                    deleted_scores = score.gather(1, deleted_local)
                    rescue_in_deleted = torch.topk(deleted_scores, k=min(rescue_count, deleted_scores.shape[1]), dim=1).indices
                    rescue_local = deleted_local.gather(1, rescue_in_deleted)
                    rescue_pos = visual_positions.unsqueeze(0).expand(batch_size, -1).gather(1, rescue_local)
                    rescue_hidden = outputs[0].gather(1, rescue_pos.unsqueeze(-1).expand(-1, -1, outputs[0].size(-1)))
                    merged_hidden, rep_local = _cluster_and_merge(rescue_hidden, cluster_count, int(self.config.sparsevlm_knn_k))
                    rescue_global = current.gather(1, rescue_pos) if current is not None else rescue_pos
                    merged_global = rescue_global.gather(1, rep_local)
                    self.runtime_stats["recycled_tokens"] = int(self.runtime_stats.get("recycled_tokens", 0) + rescue_count)
                    self.runtime_stats["merged_tokens"] = int(self.runtime_stats.get("merged_tokens", 0) + merged_hidden.shape[1])

                non_visual_mask = torch.ones(seq_len, dtype=torch.bool, device=hidden_states.device)
                non_visual_mask[visual_positions] = False
                non_visual_pos = torch.nonzero(non_visual_mask, as_tuple=False).flatten().unsqueeze(0).expand(batch_size, -1)
                non_visual_hidden = outputs[0].gather(1, non_visual_pos.unsqueeze(-1).expand(-1, -1, outputs[0].size(-1)))
                non_visual_global = current.gather(1, non_visual_pos) if current is not None else non_visual_pos
                visual_hidden = torch.cat([kept_hidden, merged_hidden], dim=1)
                visual_global = torch.cat([kept_global, merged_global], dim=1)
                if visual_global.shape[1] > 1:
                    visual_order = torch.argsort(visual_global, dim=1)
                    visual_hidden = visual_hidden.gather(1, visual_order.unsqueeze(-1).expand(-1, -1, visual_hidden.size(-1)))
                    visual_global = visual_global.gather(1, visual_order)
                bos_hidden = non_visual_hidden[:, :1, :]
                suffix_hidden = non_visual_hidden[:, 1:, :]
                new_hidden = torch.cat([bos_hidden, visual_hidden, suffix_hidden], dim=1).contiguous()
                self.global_indices = torch.cat([non_visual_global[:, :1], visual_global, non_visual_global[:, 1:]], dim=1)

                before = int(visual_positions.numel())
                after = int(visual_hidden.shape[1])
                self.runtime_stats["gather_seconds"] += time.perf_counter() - gather_t0
                self.runtime_stats["last_tokens_before"] = before
                self.runtime_stats["last_tokens_after"] = after
                self.runtime_stats["final_visual_tokens"] = float(after)
                self.runtime_stats["pruned_layers"] = int(self.runtime_stats.get("pruned_layers", 0) + 1)
                self.runtime_stats["keep_count_schedule"] = list(self.runtime_stats.get("keep_count_schedule", [])) + [int(after)]
                self.runtime_stats["method_overhead_seconds"] = (
                    self.runtime_stats["metric_seconds"] + self.runtime_stats["rank_seconds"] + self.runtime_stats["gather_seconds"]
                )
                self.runtime_stats["flops_reduction"] = _estimate_flops_reduction(seq_len, before, after)
                if self.config.debug:
                    print(f"[NoraBaseline:sparsevlm] layer={layer_idx} visual {before}->{after}, seq {seq_len}->{new_hidden.shape[1]}")
                return clean_outputs(hidden_override=new_hidden)

            if query_positions.numel() <= 0:
                return clean_outputs()

            if not self.keep_schedule:
                self.keep_schedule, lam = _pyramiddrop_schedule(
                    int(visual_positions.numel()),
                    int(self.config.target_token_num),
                    len(self.pruning_layers),
                )
                self.runtime_stats["schedule_len"] = len(self.keep_schedule)
                self.runtime_stats["keep_count_schedule"] = [int(v) for v in self.keep_schedule]
                self.runtime_stats["pyramiddrop_lambda"] = float(lam)
                self.runtime_stats["raw_total_budget"] = float(self.config.target_token_num)

            stage_idx = self.pruning_layers.index(layer_idx)
            keep_count = int(self.keep_schedule[min(stage_idx, len(self.keep_schedule) - 1)])
            if keep_count >= int(visual_positions.numel()):
                self.runtime_stats["last_tokens_before"] = int(visual_positions.numel())
                self.runtime_stats["last_tokens_after"] = int(visual_positions.numel())
                return clean_outputs()

            score_t0 = time.perf_counter()
            # PyramidDrop uses the last instruction token as query. We explicitly
            # choose the final non-action text token recorded from input_ids, not
            # the sequence tail, because VLA generation may later append action tokens.
            query_pos = query_positions[-1].expand(batch_size)
            score = self._score_visual_tokens(layer_self, outputs[0], attention_mask, position_ids, kwargs, visual_positions, query_pos)
            _, topk = torch.topk(score, keep_count, dim=1)
            topk = torch.sort(topk, dim=1).values
            selected = visual_positions.unsqueeze(0).expand(batch_size, -1).gather(1, topk)
            self.runtime_stats["topk_seconds"] += time.perf_counter() - score_t0

            gather_t0 = time.perf_counter()
            full_indices = self._build_full_indices(batch_size, seq_len, always_keep, selected, hidden_states.device)
            new_hidden = outputs[0].gather(1, full_indices.unsqueeze(-1).expand(-1, -1, outputs[0].size(-1))).contiguous()
            if self.global_indices is not None:
                current = self.global_indices
                if current.shape[0] == 1 and batch_size > 1:
                    current = current.expand(batch_size, -1)
                self.global_indices = current.gather(1, full_indices)
            self.runtime_stats["gather_seconds"] += time.perf_counter() - gather_t0
            before = int(visual_positions.numel())
            after = int(keep_count)
            self.runtime_stats["last_tokens_before"] = before
            self.runtime_stats["last_tokens_after"] = after
            self.runtime_stats["final_visual_tokens"] = float(after)
            self.runtime_stats["pruned_layers"] = int(self.runtime_stats.get("pruned_layers", 0) + 1)
            self.runtime_stats["method_overhead_seconds"] = self.runtime_stats["topk_seconds"] + self.runtime_stats["gather_seconds"]
            self.runtime_stats["flops_reduction"] = _estimate_flops_reduction(seq_len, before, after)
            if self.config.debug:
                print(f"[NoraBaseline:pyramiddrop] layer={layer_idx} visual {before}->{after}, seq {seq_len}->{full_indices.shape[1]}")
            return clean_outputs(hidden_override=new_hidden)

        return new_forward

    def get_runtime_stats(self) -> Dict[str, Any]:
        return dict(self.runtime_stats)

    def get_flops_reduction(self) -> float:
        return float(self.runtime_stats.get("flops_reduction", 0.0) or 0.0)


def maybe_apply_nora_baseline_pruning(model, config: Optional[NoraBaselinePruningConfig]) -> Optional[NoraBaselinePruningPatch]:
    if config is None or not config.enabled:
        return None
    patch = NoraBaselinePruningPatch(model, config)
    patch.apply()
    return patch


__all__ = [
    "NoraBaselinePruningConfig",
    "NoraBaselinePruningPatch",
    "maybe_apply_nora_baseline_pruning",
]
