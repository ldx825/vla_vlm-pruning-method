import math
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv
except Exception:
    apply_multimodal_rotary_pos_emb = None
    repeat_kv = None


@dataclass
class NoraAutoPruneConfig:
    enabled: bool = False
    # Legacy final-layer hard cap. In absolute/area semantics this should usually stay 0.
    target_token_num: int = 0
    visual_tokens: int = 512
    start_layer: int = 2
    num_views: int = 2
    allocation_temp: float = 0.4
    x0: float = 12.0
    k_shape: float = 0.3
    b_min: int = 96
    b_max: int = 156
    mu: float = 4.0
    tau: float = 0.5
    min_keep_per_view: int = 24
    budget_mode: str = "absolute"
    absolute_anchor_mode: str = "elbow"
    absolute_budget_scale: float = 1.0
    absolute_auto_bounds: bool = True
    enable_adaptive_layer_policy: bool = True
    enable_oneshot_fallback: bool = True
    adaptive_layer_choices: Tuple[int, ...] = (1, 3, 6)
    adaptive_layer_mu: float = 4.0
    adaptive_layer_tau: float = 0.5
    oneshot_fallback_max_layers: int = 3
    oneshot_fallback_budget_margin: float = 0.15
    oneshot_fallback_frontload_ratio: float = 0.5
    monitor_only: bool = False
    # Ablation: keep attention/statistic collection but disable per-layer token pruning branch.
    disable_token_prune: bool = False
    # Ablation: apply token pruning only at start_layer, then keep token count fixed for later layers.
    disable_progressive_prune: bool = False
    # Pure latency ablation: directly drop this many visual tokens at start_layer.
    # This path does not require attention-based token scoring.
    static_drop_tokens_at_start: int = 0
    # Experimental: keep flash attention for the actual forward pass and compute
    # attention scores manually on pruning layers when native attentions are
    # unavailable or too expensive to materialize.
    prefer_flash_attention: bool = False
    manual_attention_probe: bool = False
    manual_attention_probe_stride: int = 1
    manual_attention_probe_max_layers: int = 0
    prune_only_on_anchor_layers: bool = False
    piecewise_linear_keep_schedule: bool = False
    piecewise_linear_hard_final_cap: bool = True
    budget_semantics: str = "final_cap"
    budget_aware_x0_shift: float = 0.0
    compaction_delta_threshold_tokens: int = 0
    physical_compaction_max_shots: int = 0
    detailed_timing: bool = True
    debug: bool = False
    disable_kv_cache: bool = False


class AdaptiveBudgetStrategy:
    def __init__(self, b_min: int, b_max: int, mu: float, tau: float):
        self.b_min = b_min
        self.b_max = b_max
        self.mu = mu
        self.tau = tau

    def decide_budget(
        self,
        metric_head: torch.Tensor,
        metric_wrist: torch.Tensor,
        effective_rank: torch.Tensor,
    ) -> torch.Tensor:
        nmi_avg = (metric_head + metric_wrist) / 2.0 + 1e-6
        signal = effective_rank / nmi_avg
        value = torch.clamp((signal - self.mu) / self.tau, min=-20, max=20)
        return self.b_min + (self.b_max - self.b_min) * torch.sigmoid(value)


def _estimate_flops_reduction(prefill_len: int, before_visual: int, after_visual: int) -> float:
    dense = max(1.0, float(prefill_len) ** 2)
    pruned_len = max(1.0, float(prefill_len - max(0, before_visual - after_visual)))
    return max(0.0, min(1.0, 1.0 - (pruned_len * pruned_len) / dense))


def _finite_scalar(
    value: torch.Tensor,
    *,
    default: float,
    min_value: Optional[float] = None,
) -> torch.Tensor:
    value = torch.nan_to_num(
        value.float(),
        nan=float(default),
        posinf=float(default),
        neginf=float(default),
    )
    if min_value is not None:
        value = torch.clamp(value, min=float(min_value))
    return value


def logistic_complexity_score(metric: torch.Tensor, mu: float, tau: float) -> torch.Tensor:
    value = torch.clamp((metric - float(mu)) / max(float(tau), 1e-6), min=-20.0, max=20.0)
    return torch.sigmoid(value)


def select_progressive_pruning_layers(pruning_loc: List[int], target_layers: int) -> List[int]:
    if not pruning_loc:
        return []
    target_layers = max(1, min(int(target_layers), len(pruning_loc)))
    if target_layers >= len(pruning_loc):
        return [int(layer_idx) for layer_idx in pruning_loc]
    if target_layers == 1:
        return [int(pruning_loc[0])]

    last_idx = len(pruning_loc) - 1
    chosen: List[int] = []
    for step in range(target_layers):
        dense_idx = int(round(last_idx * step / float(target_layers - 1)))
        chosen.append(int(pruning_loc[dense_idx]))

    ordered: List[int] = []
    for layer_idx in chosen:
        if layer_idx not in ordered:
            ordered.append(layer_idx)
    if pruning_loc[0] not in ordered:
        ordered.insert(0, int(pruning_loc[0]))
    return ordered


def resolve_adaptive_pruning_layer_count(
    config: NoraAutoPruneConfig,
    pruning_loc: List[int],
    complexity_signal: torch.Tensor,
) -> Tuple[List[int], float]:
    if not pruning_loc:
        return [], 0.0

    filtered_choices: List[int] = []
    for choice in getattr(config, "adaptive_layer_choices", (1, 3, 6)):
        choice = max(1, min(int(choice), len(pruning_loc)))
        if choice not in filtered_choices:
            filtered_choices.append(choice)
    if not filtered_choices:
        filtered_choices = [1, len(pruning_loc)]
    filtered_choices = sorted(filtered_choices)

    complexity_score = float(
        logistic_complexity_score(
            complexity_signal,
            getattr(config, "adaptive_layer_mu", config.mu),
            getattr(config, "adaptive_layer_tau", config.tau),
        ).item()
    )
    if len(filtered_choices) == 1:
        return select_progressive_pruning_layers(pruning_loc, filtered_choices[0]), complexity_score

    bucket_index = min(
        len(filtered_choices) - 1,
        int(math.floor(complexity_score * len(filtered_choices))),
    )
    return select_progressive_pruning_layers(pruning_loc, filtered_choices[bucket_index]), complexity_score


def compress_keep_count_schedule(
    pruning_loc: List[int],
    keep_count_schedule: List[List[int]],
) -> Tuple[List[int], List[List[int]]]:
    if not pruning_loc or not keep_count_schedule:
        return [], []

    active_pruning_loc = [int(pruning_loc[0])]
    active_schedule = [list(map(int, keep_count_schedule[0]))]
    for layer_idx, keep_counts in zip(pruning_loc[1:], keep_count_schedule[1:]):
        keep_counts = list(map(int, keep_counts))
        if keep_counts == active_schedule[-1]:
            continue
        active_pruning_loc.append(int(layer_idx))
        active_schedule.append(keep_counts)
    return active_pruning_loc, active_schedule


def should_use_oneshot_fallback(
    config: NoraAutoPruneConfig,
    compressed_schedule: List[List[int]],
    target_keep_per_view: List[int],
) -> bool:
    if not compressed_schedule:
        return False
    if len(compressed_schedule) > int(getattr(config, "oneshot_fallback_max_layers", 3) or 3):
        return False

    target_total = int(sum(target_keep_per_view))
    if target_total <= 0:
        return False

    first_total = int(sum(compressed_schedule[0]))
    if first_total <= 0:
        return False

    second_stage = compressed_schedule[1] if len(compressed_schedule) > 1 else compressed_schedule[0]
    second_total = int(sum(second_stage))
    budget_margin = max(
        2,
        int(round(target_total * float(getattr(config, "oneshot_fallback_budget_margin", 0.15)))),
    )
    near_target_early = second_total <= target_total + budget_margin

    total_drop = max(1, first_total - target_total)
    frontloaded_drop = max(0, first_total - second_total)
    frontload_ratio = frontloaded_drop / float(total_drop)
    enough_frontloaded = frontload_ratio >= float(
        getattr(config, "oneshot_fallback_frontload_ratio", 0.5)
    )
    return bool(near_target_early and enough_frontloaded)


def allocate_keep_counts(
    view_lengths: List[int],
    total_keep: int,
    *,
    min_keep_per_view: int,
    weights: Optional[torch.Tensor] = None,
) -> List[int]:
    total_capacity = int(sum(view_lengths))
    total_keep = max(1, min(int(total_keep), total_capacity))
    counts = [0] * len(view_lengths)

    non_empty = [idx for idx, length in enumerate(view_lengths) if int(length) > 0]
    if not non_empty:
        return counts

    min_tokens = min_keep_per_view if total_keep >= len(non_empty) * min_keep_per_view else 1
    for idx in non_empty:
        counts[idx] = min(int(view_lengths[idx]), int(min_tokens))

    assigned = sum(counts)
    if assigned >= total_keep:
        return [min(int(counts[idx]), int(view_lengths[idx])) for idx in range(len(view_lengths))]

    remaining = int(total_keep - assigned)
    if weights is None:
        weights = torch.ones(len(view_lengths), dtype=torch.float32) / max(1, len(view_lengths))
    else:
        weights = torch.nan_to_num(weights.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if float(weights.sum().item()) <= 0.0:
            weights = torch.ones(len(view_lengths), dtype=torch.float32) / max(1, len(view_lengths))
        else:
            weights = weights / weights.sum()

    fractional: List[Tuple[float, int]] = []
    for idx, length in enumerate(view_lengths):
        capacity = int(length) - int(counts[idx])
        if capacity <= 0:
            continue
        raw = float(remaining) * float(weights[idx].item())
        take = min(capacity, int(math.floor(raw)))
        counts[idx] += int(take)
        fractional.append((raw - float(take), idx))

    leftover = int(total_keep - sum(counts))
    fractional.sort(reverse=True)
    for _, idx in fractional:
        if leftover <= 0:
            break
        capacity = int(view_lengths[idx]) - int(counts[idx])
        if capacity <= 0:
            continue
        counts[idx] += 1
        leftover -= 1

    return [min(int(counts[idx]), int(view_lengths[idx])) for idx in range(len(view_lengths))]


def allocate_keep_counts_by_elbow_two_views(
    attn_head: torch.Tensor,
    attn_wrist: torch.Tensor,
    min_keep_per_view: int,
) -> Tuple[List[int], List[float]]:
    keep_counts: List[int] = []
    elbow_strengths: List[float] = []
    for view_attn in (attn_head, attn_wrist):
        view_len = int(view_attn.shape[-1])
        if view_len <= 0:
            keep_counts.append(0)
            elbow_strengths.append(0.0)
            continue

        score = view_attn.mean(dim=1).mean(dim=1)[0].float()
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score = torch.clamp(score, min=0.0)
        score_sum = float(score.sum().item())
        if score_sum <= 1e-9:
            keep_counts.append(max(1, min(view_len, int(min_keep_per_view))))
            elbow_strengths.append(0.0)
            continue

        sorted_score, _ = torch.sort(score, descending=True)
        normalized = sorted_score / max(score_sum, 1e-9)
        cdf = torch.cumsum(normalized, dim=0)
        linear = torch.linspace(1.0 / view_len, 1.0, view_len, device=cdf.device, dtype=cdf.dtype)
        gain = cdf - linear
        knee_idx = int(torch.argmax(gain).item())
        keep_count = max(1, knee_idx + 1)
        keep_count = max(min(view_len, int(min_keep_per_view)), keep_count)
        keep_counts.append(int(keep_count))
        elbow_strengths.append(float(gain[knee_idx].item()))
    return keep_counts, elbow_strengths


def decide_absolute_budget_v2(
    config: NoraAutoPruneConfig,
    complexity_signal: torch.Tensor,
    effective_rank: torch.Tensor,
    view_lengths: List[int],
    total_visual: int,
    *,
    anchor_budget: Optional[float] = None,
) -> Tuple[float, int, int, float, float, Optional[float], float]:
    non_empty_views = sum(1 for length in view_lengths if int(length) > 0)
    auto_floor = sum(min(int(config.min_keep_per_view), int(length)) for length in view_lengths if int(length) > 0)
    auto_floor = max(non_empty_views, auto_floor)
    auto_ceiling = min(
        int(total_visual),
        max(
            auto_floor + 8,
            int(math.ceil(float(effective_rank.item()) * float(max(1.0, config.x0 / max(1.0, float(config.start_layer + 1)))))),
        ),
    )

    normalized_signal = torch.log1p(complexity_signal / max(1.0, float(total_visual)))
    reference_signal = torch.log1p(_finite_scalar(effective_rank, default=1.0, min_value=1.0))
    delta = (normalized_signal - reference_signal) / max(float(config.tau), 1e-6)
    score = torch.sigmoid(delta)

    anchor_budget_value: Optional[float] = None
    controller_ratio = 1.0
    if anchor_budget is not None and math.isfinite(float(anchor_budget)):
        anchor_budget_value = float(anchor_budget)
        lower_bound = max(1, auto_floor)
        upper_bound = max(lower_bound, int(total_visual))
        controller_log_ratio = float(torch.clamp(0.5 * delta, min=-0.5, max=0.5).item())
        controller_ratio = float(math.exp(controller_log_ratio))
        scale = max(0.05, float(getattr(config, "absolute_budget_scale", 1.0)))
        raw_budget = float(anchor_budget_value * controller_ratio * scale)
        raw_budget = float(max(lower_bound, min(upper_bound, raw_budget)))
    else:
        if bool(getattr(config, "absolute_auto_bounds", True)):
            lower_bound = max(1, auto_floor)
            upper_bound = max(lower_bound, auto_ceiling)
        else:
            lower_bound = max(1, int(config.b_min))
            upper_bound = max(lower_bound, min(int(total_visual), int(config.b_max)))
        raw_budget = float((lower_bound + (upper_bound - lower_bound) * score).item())

    return (
        raw_budget,
        int(lower_bound),
        int(upper_bound),
        float(normalized_signal.item()),
        float(reference_signal.item()),
        anchor_budget_value,
        float(controller_ratio),
    )


def calculate_mi_and_kl(attn: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns (MI, KL) where KL is D(p_y || Uniform). We use NMI = MI / KL.
    num_text = attn.shape[-2]
    num_vision = attn.shape[-1]

    p_x = 1.0 / num_text
    p_xy = attn * p_x
    p_y = p_xy.sum(dim=-2)

    ratio = attn / (p_y.unsqueeze(-2) + eps)
    mi = (p_xy * torch.log2(torch.clamp(ratio, min=eps))).sum(dim=(-2, -1))

    uniform = 1.0 / num_vision
    kl = (p_y * torch.log2(torch.clamp(p_y / uniform, min=eps))).sum(dim=-1)
    return mi, kl


class NoraAutoPrunePatch:
    """Inference-only AutoPrune patch for Qwen2.5-VL decoder layers."""

    def __init__(self, model, config: NoraAutoPruneConfig):
        self.model = model
        self.config = config
        self.strategy = AdaptiveBudgetStrategy(config.b_min, config.b_max, config.mu, config.tau)

        self.layers = self._resolve_layers(model)
        self.original_forwards: Dict[int, Any] = {}
        self.proj_hook_handles: List[Any] = []
        self.proj_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        self.keep_percent_schedule = []
        self.keep_count_schedule = []
        self.compaction_layer_flags: List[bool] = []
        self.global_indices = None
        self.cached_rank = None
        self.view_boundary_tracker: Dict[str, int] = {}
        self.runtime_stats: Dict[str, Any] = {}
        self.input_layout: Dict[str, Any] = {}
        self.cached_view_score_ids: List[torch.Tensor] = []
        self.cached_view_score_values: List[torch.Tensor] = []
        self.manual_probe_anchor_layers: set[int] = set()
        self.noop_control_mode = False

        self.num_layers = len(self.layers)
        self.tokens_per_view = max(1, config.visual_tokens // max(1, config.num_views))
        self.pruning_loc = list(range(config.start_layer, self.num_layers))
        self.manual_probe_anchor_layers = self._resolve_manual_probe_anchor_layers()

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

        raise RuntimeError("Unable to locate decoder layers for AutoPrune in the provided model.")

    def reset_runtime_state(self) -> None:
        self.keep_percent_schedule = []
        self.keep_count_schedule = []
        self.compaction_layer_flags = []
        self.global_indices = None
        self.cached_rank = None
        self.view_boundary_tracker = {}
        self.input_layout = {}
        self.proj_cache = {}
        self.cached_view_score_ids = []
        self.cached_view_score_values = []
        self.noop_control_mode = False
        self.runtime_stats = {
            "prefill_seq_len": None,
            "schedule_len": 0,
            "nmi_head": None,
            "nmi_wrist": None,
            "effective_rank": None,
            "total_budget": None,
            "effective_x0_head": None,
            "effective_x0_wrist": None,
            "pruned_layers": 0,
            "last_tokens_before": None,
            "last_tokens_after": None,
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
            "compaction_skipped_layers": 0,
            "planned_compaction_layers": 0,
            "layer0_total_calls": 0,
            "layer0_short_seq_calls": 0,
            "layer0_long_seq_calls": 0,
            "layer0_past_none_calls": 0,
            "layer0_past_present_calls": 0,
            "layout_ready": 0,
            "layout_query_tokens": 0,
            "layout_view0_tokens": 0,
            "layout_view1_tokens": 0,
            "flops_reduction": 0.0,
            "final_visual_tokens": 0.0,
            "raw_total_budget": 0.0,
            "active_pruning_layers": float(len(self.pruning_loc)),
            "manual_attention_probe_layers": 0,
            "manual_probe_qk_cache_hits": 0,
            "manual_probe_qk_cache_misses": 0,
            "manual_probe_reused_layers": 0,
        }

    def _resolve_manual_probe_anchor_layers(self, x0_override: Optional[float] = None) -> set[int]:
        if not self.pruning_loc:
            return set()
        max_layers = int(getattr(self.config, "manual_attention_probe_max_layers", 0) or 0)
        if max_layers <= 0 or max_layers >= len(self.pruning_loc):
            return set(self.pruning_loc)
        if max_layers == 1:
            return {self.pruning_loc[0]}
        chosen: List[int] = [self.pruning_loc[0], self.pruning_loc[-1]]
        if max_layers >= 3:
            x0_value = float(self.config.x0 if x0_override is None else x0_override)
            knee_layer = min(self.pruning_loc, key=lambda layer: abs(float(layer) - x0_value))
            chosen.append(knee_layer)

        if len(set(chosen)) < max_layers:
            last_idx = len(self.pruning_loc) - 1
            for step in range(max_layers):
                chosen.append(self.pruning_loc[int(round(last_idx * step / float(max_layers - 1)))])

        ordered: List[int] = []
        for layer_idx in sorted(set(chosen)):
            if layer_idx not in ordered:
                ordered.append(layer_idx)
        if len(ordered) > max_layers:
            ordered = ordered[:max_layers]
        return set(ordered)

    def _effective_x0_for_target(self, target_keep: int, start_count: int) -> float:
        base_x0 = float(self.config.x0)
        shift_strength = float(getattr(self.config, "budget_aware_x0_shift", 0.0) or 0.0)
        if shift_strength <= 0.0 or start_count <= 0:
            return base_x0

        keep_fraction = max(0.0, min(1.0, float(target_keep) / float(start_count)))
        x0_eff = base_x0 - shift_strength * (1.0 - keep_fraction)
        min_x0 = float(self.config.start_layer)
        max_x0 = float(max(self.pruning_loc[-1], self.config.start_layer))
        return max(min_x0, min(max_x0, x0_eff))

    def _is_no_prune_budget(self, current_vis_len: int) -> bool:
        if current_vis_len <= 0:
            return True
        if self.config.monitor_only or self.config.disable_token_prune:
            return True
        return int(self.config.b_min) >= current_vis_len and int(self.config.b_max) >= current_vis_len

    def apply(self) -> None:
        if not self.config.enabled:
            return

        for i, layer in enumerate(self.layers):
            self.original_forwards[i] = layer.forward
            layer.forward = types.MethodType(self._make_new_forward(i), layer)
            self._register_projection_hooks(i, layer)
        self.model._autoprune_patch = self

    def _register_projection_hooks(self, layer_idx: int, layer: Any) -> None:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            return

        def make_hook(cache_key: str):
            def hook(_module, _inputs, output):
                layer_cache = self.proj_cache.setdefault(layer_idx, {})
                layer_cache[cache_key] = output
            return hook

        for module_name, cache_key in (("q_proj", "q_proj"), ("k_proj", "k_proj")):
            module = getattr(self_attn, module_name, None)
            if module is None:
                continue
            self.proj_hook_handles.append(module.register_forward_hook(make_hook(cache_key)))

    def _pop_cached_proj_states(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        layer_cache = self.proj_cache.pop(layer_idx, {})
        return layer_cache.get("q_proj"), layer_cache.get("k_proj")

    def _run_profiled_original_forward(
        self,
        layer_idx: int,
        self_layer: Any,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        forward_kwargs: Dict[str, Any],
        stat_prefix: str = "force_attn",
    ):
        component_prefix = stat_prefix[:-8] if stat_prefix.endswith("_forward") else stat_prefix
        detailed_timing = bool(getattr(self.config, "detailed_timing", True))
        if not detailed_timing:
            forward_t0 = time.perf_counter()
            outputs = self.original_forwards[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **forward_kwargs,
            )
            self.runtime_stats[f"{stat_prefix}_seconds"] = float(
                self.runtime_stats.get(f"{stat_prefix}_seconds", 0.0) + (time.perf_counter() - forward_t0)
            )
            return outputs

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
                    timed_self_attn_seconds += (time.perf_counter() - t0)

            self_attn.forward = types.MethodType(timed_self_attn_forward, self_attn)

        if orig_mlp_forward is not None:
            def timed_mlp_forward(module_self, *args, **kwargs):
                nonlocal timed_mlp_seconds
                t0 = time.perf_counter()
                try:
                    return orig_mlp_forward(*args, **kwargs)
                finally:
                    timed_mlp_seconds += (time.perf_counter() - t0)

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

    def _cache_view_scores(
        self,
        view_positions: List[torch.Tensor],
        score_h: torch.Tensor,
        score_w: torch.Tensor,
    ) -> None:
        if self.global_indices is None or len(view_positions) < 2:
            return

        current_globals = self.global_indices[0]
        self.cached_view_score_ids = []
        self.cached_view_score_values = []

        for view_idx, scores in enumerate((score_h, score_w)):
            positions = view_positions[view_idx]
            if positions.numel() == 0:
                self.cached_view_score_ids.append(torch.empty(0, dtype=torch.long, device=current_globals.device))
                self.cached_view_score_values.append(torch.empty(0, dtype=scores.dtype, device=scores.device))
                continue
            global_ids = current_globals.index_select(0, positions).detach()
            values = scores[0].detach()
            self.cached_view_score_ids.append(global_ids)
            self.cached_view_score_values.append(values)

    def _reuse_cached_view_scores(
        self,
        batch_size: int,
        view_positions: List[torch.Tensor],
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if (
            self.global_indices is None
            or len(self.cached_view_score_ids) < 2
            or len(view_positions) < 2
        ):
            return None

        current_globals = self.global_indices[0]
        reused_scores: List[torch.Tensor] = []
        for view_idx in range(2):
            positions = view_positions[view_idx]
            cached_ids = self.cached_view_score_ids[view_idx]
            cached_vals = self.cached_view_score_values[view_idx]
            if positions.numel() == 0:
                reused_scores.append(torch.empty((batch_size, 0), dtype=torch.float32, device=device))
                continue
            if cached_ids.numel() == 0:
                return None
            current_ids = current_globals.index_select(0, positions)
            gather_idx = torch.searchsorted(cached_ids, current_ids)
            if torch.any(gather_idx >= cached_ids.numel()):
                return None
            matched_ids = cached_ids.index_select(0, gather_idx)
            if not torch.equal(matched_ids, current_ids):
                return None
            values = cached_vals.index_select(0, gather_idx).to(device=device)
            reused_scores.append(values.unsqueeze(0).expand(batch_size, -1))

        self.runtime_stats["manual_probe_reused_layers"] = int(
            self.runtime_stats.get("manual_probe_reused_layers", 0) + 1
        )
        return reused_scores[0], reused_scores[1]

    def _calculate_effective_rank(self, hidden_states: torch.Tensor, text_slice: slice) -> torch.Tensor:
        try:
            text_features = hidden_states[0, text_slice, :].float()
            if text_features.numel() == 0:
                return torch.tensor(1.0, device=hidden_states.device)
            sigmas = torch.linalg.svdvals(text_features)
            sigma_sum = sigmas.sum()
            if sigma_sum < 1e-9:
                return torch.tensor(1.0, device=hidden_states.device)
            probs = sigmas / sigma_sum
            entropy = -torch.sum(probs * torch.log(probs + 1e-12))
            return torch.exp(entropy)
        except Exception:
            return torch.tensor(1.0, device=hidden_states.device)

    def _generate_schedule_aligned(self, target_total: torch.Tensor, x0: float, k: float) -> torch.Tensor:
        num_pruning_layers = max(1, len(self.pruning_loc))
        device = target_total.device

        layer_indices = torch.arange(self.config.start_layer, self.num_layers, device=device, dtype=torch.float32)
        raw_curve = 1.0 / (1.0 + torch.exp(k * (layer_indices - x0)))

        scales = torch.linspace(0.01, 100.0, 2000, device=device)
        clamped = torch.clamp(scales.unsqueeze(1) * raw_curve.unsqueeze(0), max=1.0)
        totals = clamped.sum(dim=1) * self.tokens_per_view

        target = torch.clamp(target_total, min=1.0, max=float(self.tokens_per_view * num_pruning_layers))
        best_idx = torch.argmin((totals - target).abs())
        best_curve = clamped[best_idx]

        prev_curve = torch.cat([torch.ones(1, device=device), best_curve[:-1]])
        return torch.clamp(best_curve / (prev_curve + 1e-9), min=0.001, max=1.0)

    def _generate_area_calibrated_schedule(
        self,
        start_count: int,
        target_keep: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        num_pruning_layers = max(1, len(self.pruning_loc))
        if num_pruning_layers <= 0:
            return torch.ones(1, device=device), float(self.config.x0)

        target_keep = max(1, min(int(target_keep), int(start_count)))
        target_total = torch.tensor(float(target_keep * num_pruning_layers), device=device)
        min_keep = max(1, min(int(self.config.min_keep_per_view), int(start_count)))
        x0_candidates = torch.linspace(
            float(self.config.start_layer),
            float(max(self.pruning_loc[-1], self.config.start_layer)),
            steps=max(16, min(128, num_pruning_layers * 4)),
            device=device,
        )

        best_ratios: Optional[torch.Tensor] = None
        best_x0 = float(self.config.x0)
        best_score: Optional[float] = None

        for x0_tensor in x0_candidates:
            x0_value = float(x0_tensor.item())
            ratios = self._generate_schedule_aligned(target_total, x0_value, self.config.k_shape)
            dense_counts = self._simulate_dense_keep_counts(start_count, ratios, min_keep)
            avg_keep = float(sum(dense_counts)) / float(len(dense_counts))
            final_keep = float(dense_counts[-1])
            avg_err = abs(avg_keep - float(target_keep))
            tail_err = abs(final_keep - float(target_keep))
            floor_penalty = 0.0
            if target_keep > int(min_keep * 1.5) and final_keep <= float(min_keep + 1):
                floor_penalty = float(target_keep - min_keep)
            score = avg_err + (0.20 * tail_err) + floor_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_ratios = ratios
                best_x0 = x0_value

        if best_ratios is None:
            return self._generate_schedule_aligned(target_total, float(self.config.x0), self.config.k_shape), float(self.config.x0)
        return best_ratios, best_x0

    def _simulate_dense_keep_counts(
        self,
        start_count: int,
        step_ratios: torch.Tensor,
        min_keep: int,
    ) -> List[int]:
        current = int(start_count)
        dense_counts: List[int] = []
        lower_bound = max(1, min(int(min_keep), int(start_count)))
        for ratio in step_ratios.tolist():
            next_count = int(current * float(ratio))
            next_count = max(lower_bound, next_count)
            next_count = min(next_count, current)
            dense_counts.append(int(next_count))
            current = int(next_count)
        return dense_counts

    def _build_piecewise_linear_keep_counts(
        self,
        dense_counts: List[int],
        start_count: int,
        final_target: int,
    ) -> List[int]:
        if not dense_counts:
            return []

        num_layers = len(dense_counts)
        lower_bound = max(1, min(int(self.config.min_keep_per_view), int(start_count)))
        final_target = max(lower_bound, min(int(final_target), int(start_count)))

        anchor_layers = sorted(
            max(0, min(num_layers - 1, int(layer_idx) - self.config.start_layer))
            for layer_idx in (self.manual_probe_anchor_layers or set(self.pruning_loc))
        )
        anchor_indices = sorted(set([0, num_layers - 1] + anchor_layers))

        anchor_values: Dict[int, int] = {}
        for dense_idx in anchor_indices:
            value = int(dense_counts[dense_idx])
            if dense_idx == num_layers - 1 and self.config.piecewise_linear_hard_final_cap:
                value = final_target
            anchor_values[dense_idx] = max(lower_bound, min(value, int(start_count)))

        anchor_values[0] = max(anchor_values[0], final_target)
        anchor_values[0] = min(anchor_values[0], int(start_count))
        if self.config.piecewise_linear_hard_final_cap:
            anchor_values[num_layers - 1] = final_target

        prev_anchor_idx = anchor_indices[0]
        for dense_idx in anchor_indices[1:]:
            anchor_values[dense_idx] = min(anchor_values[dense_idx], anchor_values[prev_anchor_idx])
            if dense_idx == num_layers - 1 and self.config.piecewise_linear_hard_final_cap:
                anchor_values[dense_idx] = final_target
            prev_anchor_idx = dense_idx

        schedule = [int(start_count)] * num_layers
        for left_idx, right_idx in zip(anchor_indices[:-1], anchor_indices[1:]):
            left_val = anchor_values[left_idx]
            right_val = anchor_values[right_idx]
            span = max(1, right_idx - left_idx)
            for dense_idx in range(left_idx, right_idx + 1):
                ratio = (dense_idx - left_idx) / float(span)
                value = int(round(left_val + (right_val - left_val) * ratio))
                schedule[dense_idx] = value

        prev_value = int(start_count)
        for dense_idx in range(num_layers):
            floor = final_target if dense_idx == num_layers - 1 and self.config.piecewise_linear_hard_final_cap else lower_bound
            value = max(floor, min(int(schedule[dense_idx]), prev_value))
            schedule[dense_idx] = value
            prev_value = value

        if self.config.piecewise_linear_hard_final_cap:
            schedule[-1] = final_target
            prev_value = int(start_count)
            for dense_idx in range(num_layers):
                floor = final_target if dense_idx == num_layers - 1 else lower_bound
                value = max(floor, min(int(schedule[dense_idx]), prev_value))
                schedule[dense_idx] = value
                prev_value = value

        return schedule

    @staticmethod
    def _keep_schedule_error(
        dense_h: List[int],
        dense_w: List[int],
        sched_h: List[int],
        sched_w: List[int],
    ) -> float:
        total_dense = [int(h) + int(w) for h, w in zip(dense_h, dense_w)]
        total_sched = [int(h) + int(w) for h, w in zip(sched_h, sched_w)]
        total_shape = sum((float(s) - float(d)) ** 2 for s, d in zip(total_sched, total_dense))
        view_shape = 0.25 * (
            sum((float(s) - float(d)) ** 2 for s, d in zip(sched_h, dense_h))
            + sum((float(s) - float(d)) ** 2 for s, d in zip(sched_w, dense_w))
        )
        area_penalty = 4.0 * abs(sum(total_sched) - sum(total_dense))
        return float(total_shape + view_shape + area_penalty)

    def _build_sparse_compaction_keep_counts(
        self,
        dense_h: List[int],
        dense_w: List[int],
        start_h: int,
        start_w: int,
        final_h: int,
        final_w: int,
        max_shots: int,
    ) -> Tuple[List[List[int]], List[bool]]:
        num_layers = min(len(dense_h), len(dense_w))
        if num_layers <= 0 or max_shots <= 0:
            return [], []

        start_h = int(start_h)
        start_w = int(start_w)
        final_h = int(max(0, min(final_h, start_h)))
        final_w = int(max(0, min(final_w, start_w)))
        start_total = int(start_h + start_w)
        final_total = int(final_h + final_w)
        dense_total = [int(h) + int(w) for h, w in zip(dense_h, dense_w)]
        target_area_total = float(sum(dense_total))

        best_schedule: Optional[Tuple[List[int], List[int], List[bool], float]] = None

        def consider_candidate(sched_h: List[int], sched_w: List[int], flags: List[bool]) -> None:
            nonlocal best_schedule
            err = self._keep_schedule_error(dense_h, dense_w, sched_h, sched_w)
            candidate = (sched_h, sched_w, flags, err)
            if best_schedule is None or err < best_schedule[3]:
                best_schedule = candidate

        for shot_idx in range(num_layers):
            sched_h = [start_h] * shot_idx + [final_h] * (num_layers - shot_idx)
            sched_w = [start_w] * shot_idx + [final_w] * (num_layers - shot_idx)
            flags = [False] * num_layers
            if shot_idx < num_layers and (final_h < start_h or final_w < start_w):
                flags[shot_idx] = True
            consider_candidate(sched_h, sched_w, flags)

        if max_shots >= 2 and num_layers >= 2:
            for first_idx in range(num_layers - 1):
                for second_idx in range(first_idx + 1, num_layers):
                    span = int(second_idx - first_idx)
                    if span <= 0:
                        continue

                    remaining_area = target_area_total - (first_idx * start_total) - ((num_layers - second_idx) * final_total)
                    mid_total = int(round(remaining_area / float(span)))
                    mid_total = max(final_total, min(mid_total, start_total))

                    mid_slice_h = dense_h[first_idx:second_idx]
                    mid_slice_w = dense_w[first_idx:second_idx]
                    desired_h = float(sum(mid_slice_h) / float(len(mid_slice_h))) if mid_slice_h else float(final_h)
                    desired_w = float(sum(mid_slice_w) / float(len(mid_slice_w))) if mid_slice_w else float(final_w)
                    mid_h, mid_w = self._allocate_keep_counts_from_targets(
                        [start_h, start_w],
                        [desired_h, desired_w],
                        int(mid_total),
                    )
                    mid_h = max(final_h, min(int(mid_h), start_h))
                    mid_w = max(final_w, min(int(mid_w), start_w))

                    sched_h = (
                        [start_h] * first_idx
                        + [mid_h] * (second_idx - first_idx)
                        + [final_h] * (num_layers - second_idx)
                    )
                    sched_w = (
                        [start_w] * first_idx
                        + [mid_w] * (second_idx - first_idx)
                        + [final_w] * (num_layers - second_idx)
                    )
                    flags = [False] * num_layers
                    if mid_h < start_h or mid_w < start_w:
                        flags[first_idx] = True
                    if final_h < mid_h or final_w < mid_w:
                        flags[second_idx] = True
                    consider_candidate(sched_h, sched_w, flags)

        if best_schedule is None:
            return [], []

        sched_h, sched_w, flags, _ = best_schedule
        keep_schedule = [[int(h), int(w)] for h, w in zip(sched_h, sched_w)]
        return keep_schedule, [bool(flag) for flag in flags]

    def _build_sparse_area_matched_keep_counts(
        self,
        dense_h: List[int],
        dense_w: List[int],
        start_h: int,
        start_w: int,
        max_shots: int,
    ) -> Tuple[List[List[int]], List[bool]]:
        num_layers = min(len(dense_h), len(dense_w))
        if num_layers <= 0 or max_shots <= 0:
            return [], []

        start_h = int(start_h)
        start_w = int(start_w)
        lower_h = max(1, min(int(self.config.min_keep_per_view), start_h))
        lower_w = max(1, min(int(self.config.min_keep_per_view), start_w))
        dense_total = [int(h) + int(w) for h, w in zip(dense_h, dense_w)]
        start_total = int(start_h + start_w)
        end_total = int(dense_total[-1]) if dense_total else int(lower_h + lower_w)
        total_drop = max(0, start_total - end_total)
        shots = max(1, min(int(max_shots), num_layers))

        compaction_starts: List[int] = []
        if total_drop > 0:
            for step in range(1, shots + 1):
                target_total = start_total - (total_drop * step / float(shots))
                start_idx = next(
                    (idx for idx, value in enumerate(dense_total) if float(value) <= float(target_total)),
                    num_layers - 1,
                )
                compaction_starts.append(int(start_idx))

        if not compaction_starts:
            return [[int(start_h), int(start_w)] for _ in range(num_layers)], [False] * num_layers

        unique_starts = sorted(set(idx for idx in compaction_starts if 0 <= idx < num_layers))
        if len(unique_starts) < shots:
            for idx in range(num_layers):
                if idx not in unique_starts:
                    unique_starts.append(idx)
                if len(unique_starts) >= shots:
                    break
            unique_starts = sorted(set(unique_starts))
        if len(unique_starts) > shots:
            unique_starts = unique_starts[:shots]

        sched_h = [start_h] * num_layers
        sched_w = [start_w] * num_layers
        flags = [False] * num_layers
        prev_h, prev_w = int(start_h), int(start_w)

        for start_idx_pos, start_idx in enumerate(unique_starts):
            end_idx = unique_starts[start_idx_pos + 1] if start_idx_pos + 1 < len(unique_starts) else num_layers
            seg_h = dense_h[start_idx:end_idx] or [dense_h[-1]]
            seg_w = dense_w[start_idx:end_idx] or [dense_w[-1]]
            target_h = int(round(sum(seg_h) / float(len(seg_h))))
            target_w = int(round(sum(seg_w) / float(len(seg_w))))
            target_h = max(lower_h, min(target_h, prev_h))
            target_w = max(lower_w, min(target_w, prev_w))
            if target_h < prev_h or target_w < prev_w:
                flags[start_idx] = True
            for dense_idx in range(start_idx, end_idx):
                sched_h[dense_idx] = target_h
                sched_w[dense_idx] = target_w
            prev_h, prev_w = target_h, target_w

        keep_schedule = [[int(h), int(w)] for h, w in zip(sched_h, sched_w)]
        return keep_schedule, [bool(flag) for flag in flags]

    @staticmethod
    def _split_contiguous_positions(positions: torch.Tensor) -> List[torch.Tensor]:
        if positions.numel() == 0:
            return []
        if positions.numel() == 1:
            return [positions]

        splits = torch.nonzero(positions[1:] != (positions[:-1] + 1), as_tuple=False).flatten() + 1
        chunks = torch.tensor_split(positions, splits.tolist())
        return [chunk for chunk in chunks if chunk.numel() > 0]

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

        expected_counts: List[int] = []
        if image_grid_thw is not None and image_grid_thw.numel() > 0:
            merge_area = max(1, int(spatial_merge_size) ** 2)
            expected_counts = [
                int(((grid[0] * grid[1] * grid[2]).item()) // merge_area)
                for grid in image_grid_thw
            ]

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

        self.input_layout = {
            "ready": True,
            "dual_view_mode": dual_view_mode,
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
                "mode": "qwen_input_ids",
                "query_positions": query_positions,
                "view_positions": view_positions,
                "visual_positions": current_visual_positions,
                "always_keep_positions": always_keep_positions,
            }

        vis_start = 1
        vis_end = min(1 + self.config.visual_tokens, seq_len - 1)
        if vis_end <= vis_start:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {
                "mode": "legacy",
                "query_positions": empty,
                "view_positions": [empty, empty],
                "visual_positions": empty,
                "always_keep_positions": torch.arange(seq_len, device=device, dtype=torch.long),
            }

        current_vis_len = vis_end - vis_start
        current_per_view = current_vis_len // 2
        head_positions = torch.arange(vis_start, vis_start + current_per_view, device=device, dtype=torch.long)
        wrist_positions = torch.arange(
            vis_start + current_per_view,
            vis_start + current_vis_len,
            device=device,
            dtype=torch.long,
        )
        query_positions = torch.arange(vis_end, seq_len, device=device, dtype=torch.long)
        keep_before = torch.arange(0, vis_start, device=device, dtype=torch.long)
        always_keep_positions = torch.cat([keep_before, query_positions], dim=0)

        return {
            "mode": "legacy",
            "query_positions": query_positions,
            "view_positions": [head_positions, wrist_positions],
            "visual_positions": torch.cat([head_positions, wrist_positions], dim=0),
            "always_keep_positions": always_keep_positions,
        }

    @staticmethod
    def _allocate_keep_counts(view_lengths: List[int], total_keep: int) -> List[int]:
        total_capacity = sum(view_lengths)
        total_keep = max(1, min(int(total_keep), total_capacity))
        counts = [0] * len(view_lengths)

        non_empty = [idx for idx, length in enumerate(view_lengths) if length > 0]
        if not non_empty:
            return counts

        if total_keep >= len(non_empty):
            for idx in non_empty:
                counts[idx] = 1
            remaining = total_keep - len(non_empty)
        else:
            for idx in non_empty[:total_keep]:
                counts[idx] = 1
            return counts

        allocatable = [max(0, length - counts[idx]) for idx, length in enumerate(view_lengths)]
        alloc_sum = sum(allocatable)
        if alloc_sum <= 0 or remaining <= 0:
            return counts

        fractional: List[Tuple[float, int]] = []
        for idx, capacity in enumerate(allocatable):
            if capacity <= 0:
                continue
            raw = remaining * (capacity / alloc_sum)
            take = min(capacity, int(math.floor(raw)))
            counts[idx] += take
            fractional.append((raw - take, idx))

        leftover = total_keep - sum(counts)
        fractional.sort(reverse=True)
        for _, idx in fractional:
            if leftover <= 0:
                break
            capacity = view_lengths[idx] - counts[idx]
            if capacity <= 0:
                continue
            counts[idx] += 1
            leftover -= 1

        for idx, length in enumerate(view_lengths):
            counts[idx] = min(counts[idx], length)

        return counts

    @staticmethod
    def _allocate_keep_counts_from_targets(
        view_lengths: List[int], desired_counts: List[float], total_keep: int
    ) -> List[int]:
        total_capacity = sum(view_lengths)
        total_keep = max(1, min(int(total_keep), total_capacity))
        counts = [0] * len(view_lengths)

        non_empty = [idx for idx, length in enumerate(view_lengths) if length > 0]
        if not non_empty:
            return counts

        # Keep at least one token per non-empty view when the total budget permits it.
        if total_keep >= len(non_empty):
            for idx in non_empty:
                counts[idx] = 1
            remaining = total_keep - len(non_empty)
        else:
            for idx in non_empty[:total_keep]:
                counts[idx] = 1
            return counts

        fractional: List[Tuple[float, int]] = []
        for idx in non_empty:
            capacity = max(0, view_lengths[idx] - counts[idx])
            if capacity <= 0:
                continue
            desired_extra = max(0.0, float(desired_counts[idx]) - counts[idx])
            take = min(capacity, int(math.floor(desired_extra)))
            counts[idx] += take
            fractional.append((desired_extra - take, idx))

        leftover = total_keep - sum(counts)
        fractional.sort(reverse=True)
        while leftover > 0:
            progressed = False
            for _, idx in fractional:
                capacity = view_lengths[idx] - counts[idx]
                if capacity <= 0:
                    continue
                counts[idx] += 1
                leftover -= 1
                progressed = True
                if leftover <= 0:
                    break
            if not progressed:
                break

        return counts

    @staticmethod
    def _build_full_indices(
        batch_size: int,
        seq_len: int,
        always_keep_positions: torch.Tensor,
        selected_view_positions: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        keep_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        if always_keep_positions.numel() > 0:
            keep_mask[:, always_keep_positions] = True
        for selected in selected_view_positions:
            if selected.numel() > 0:
                keep_mask.scatter_(1, selected, True)

        base = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return base[keep_mask].view(batch_size, -1)

    @staticmethod
    def _counts_to_keep_ratios(
        keep_count_schedule: List[List[int]],
        start_h: int,
        start_w: int,
    ) -> List[List[float]]:
        if not keep_count_schedule:
            return []

        prev_h = max(1, int(start_h))
        prev_w = max(1, int(start_w))
        ratios: List[List[float]] = []
        for keep_h, keep_w in keep_count_schedule:
            keep_h = int(max(0, keep_h))
            keep_w = int(max(0, keep_w))
            ratio_h = 1.0 if prev_h <= 0 else max(0.001, min(1.0, float(keep_h) / float(prev_h)))
            ratio_w = 1.0 if prev_w <= 0 else max(0.001, min(1.0, float(keep_w) / float(prev_w)))
            ratios.append([float(ratio_h), float(ratio_w)])
            prev_h = max(1, keep_h)
            prev_w = max(1, keep_w)
        return ratios

    @staticmethod
    def _expand_active_keep_schedule(
        pruning_loc: List[int],
        active_pruning_loc: List[int],
        active_schedule: List[List[int]],
    ) -> Tuple[List[List[int]], List[bool]]:
        if not pruning_loc or not active_pruning_loc or not active_schedule:
            return [], []

        schedule_index = {int(layer_idx): idx for idx, layer_idx in enumerate(active_pruning_loc)}
        full_schedule: List[List[int]] = []
        compaction_flags: List[bool] = []
        current_counts = list(map(int, active_schedule[0]))
        first_active = int(active_pruning_loc[0])
        for layer_idx in pruning_loc:
            layer_idx = int(layer_idx)
            if layer_idx in schedule_index:
                current_counts = list(map(int, active_schedule[schedule_index[layer_idx]]))
                compaction_flags.append(layer_idx >= first_active)
            else:
                compaction_flags.append(False)
            full_schedule.append(list(current_counts))
        return full_schedule, compaction_flags

    def _compute_manual_attention_probe(
        self,
        layer_idx: int,
        self_layer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        detailed_timing = bool(getattr(self.config, "detailed_timing", True))
        probe_t0 = time.perf_counter() if detailed_timing else None

        def finish_probe(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if detailed_timing and probe_t0 is not None:
                self.runtime_stats["manual_probe_seconds"] = float(
                    self.runtime_stats.get("manual_probe_seconds", 0.0) + (time.perf_counter() - probe_t0)
                )
            return result

        if apply_multimodal_rotary_pos_emb is None or repeat_kv is None:
            return finish_probe(None)

        self_attn = getattr(self_layer, "self_attn", None)
        position_embeddings = kwargs.get("position_embeddings", None)
        if self_attn is None or position_embeddings is None:
            return finish_probe(None)

        try:
            bsz, q_len, _ = hidden_states.size()
            # In Qwen generate(), the prefill path may still pass a cache object.
            # We only want the long-sequence prefill probe, not the short decode steps.
            if kwargs.get("past_key_value", None) is not None and q_len <= 96:
                return finish_probe(None)

            layout = self._resolve_layout_state(q_len, hidden_states.device)
            query_positions = layout["query_positions"]
            view_positions = layout["view_positions"]
            if len(view_positions) < 2 or query_positions.numel() <= 0:
                return finish_probe(None)

            c_h = int(view_positions[0].numel())
            c_w = int(view_positions[1].numel())
            if c_h <= 0 or c_w <= 0:
                return finish_probe(None)

            visual_positions = torch.cat(view_positions, dim=0).contiguous()

            cached_q_proj, cached_k_proj = self._pop_cached_proj_states(layer_idx)
            if cached_q_proj is not None and cached_k_proj is not None:
                query_states = cached_q_proj
                key_states = cached_k_proj
                self.runtime_stats["manual_probe_qk_cache_hits"] = int(
                    self.runtime_stats.get("manual_probe_qk_cache_hits", 0) + 1
                )
            else:
                query_states = self_attn.q_proj(hidden_states)
                key_states = self_attn.k_proj(hidden_states)
                self.runtime_stats["manual_probe_qk_cache_misses"] = int(
                    self.runtime_stats.get("manual_probe_qk_cache_misses", 0) + 1
                )

            query_states = query_states.view(bsz, q_len, -1, self_attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, -1, self_attn.head_dim).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self_attn.rope_scaling["mrope_section"],
            )

            key_states = repeat_kv(key_states, self_attn.num_key_value_groups)

            # Single-shot AutoPrune only needs text-query attention over the current visual tokens.
            # Restricting the probe to this submatrix avoids materializing the full q_len x q_len map.
            query_states = query_states.index_select(2, query_positions)
            key_states = key_states.index_select(2, visual_positions)
            attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self_attn.head_dim)

            if attn_logits.dtype == torch.float16:
                attn_logits = torch.where(torch.isinf(attn_logits), torch.zeros_like(attn_logits), attn_logits)

            attn_weights = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            self.runtime_stats["manual_attention_probe_layers"] = int(
                self.runtime_stats.get("manual_attention_probe_layers", 0) + 1
            )
            return finish_probe({
                "query_positions": query_positions,
                "view_positions": view_positions,
                "attn_head": attn_weights[:, :, :, :c_h],
                "attn_wrist": attn_weights[:, :, :, c_h : c_h + c_w],
            })
        except Exception:
            return finish_probe(None)

    def _make_new_forward(self, layer_idx: int):
        def new_forward(self_layer, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            patch_t0 = time.perf_counter()

            def finish(result):
                self.runtime_stats["patch_total_seconds"] = float(
                    self.runtime_stats.get("patch_total_seconds", 0.0) + (time.perf_counter() - patch_t0)
                )
                return result

            batch_size, seq_len, _ = hidden_states.shape

            if layer_idx == 0:
                self.runtime_stats["layer0_total_calls"] = int(self.runtime_stats.get("layer0_total_calls", 0) + 1)
                if seq_len <= 96:
                    self.runtime_stats["layer0_short_seq_calls"] = int(self.runtime_stats.get("layer0_short_seq_calls", 0) + 1)
                else:
                    self.runtime_stats["layer0_long_seq_calls"] = int(self.runtime_stats.get("layer0_long_seq_calls", 0) + 1)
                if kwargs.get("past_key_value", None) is None:
                    self.runtime_stats["layer0_past_none_calls"] = int(self.runtime_stats.get("layer0_past_none_calls", 0) + 1)
                else:
                    self.runtime_stats["layer0_past_present_calls"] = int(
                        self.runtime_stats.get("layer0_past_present_calls", 0) + 1
                    )

            if layer_idx == 0 and self.runtime_stats.get("prefill_seq_len") is None:
                self.runtime_stats["prefill_seq_len"] = int(seq_len)

            short_seq_skip = seq_len <= 96

            if layer_idx == 0 or self.global_indices is None:
                self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

            # Keep position_ids aligned with pruned hidden_states.
            if position_ids is None:
                position_ids = self.global_indices
            else:
                if position_ids.dim() == 2:
                    # [B, L]
                    if position_ids.shape[1] > seq_len:
                        gather_idx = self.global_indices
                        if gather_idx.shape[0] == 1 and batch_size > 1:
                            gather_idx = gather_idx.expand(batch_size, -1)
                        position_ids = position_ids.gather(1, gather_idx)
                elif position_ids.dim() == 3:
                    # Qwen2.5-VL commonly uses [3, B, L]
                    if position_ids.shape[2] > seq_len:
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
                    # Qwen2.5-VL rotary embeddings are often [3, B, L, D].
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
                        # Fallback layout [B, L, D]
                        gather_idx = self.global_indices
                        if gather_idx.shape[0] == 1 and batch_size > 1:
                            gather_idx = gather_idx.expand(batch_size, -1)
                        cos = cos.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, cos.size(-1)))
                        sin = sin.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, sin.size(-1)))
                        kwargs["position_embeddings"] = (cos, sin)
                except Exception:
                    # Keep original position_embeddings when gather adaptation fails.
                    pass

            # During decode / very short sequences, skip pruning but keep aligned ids/rotary tensors.
            if short_seq_skip:
                if self.config.debug and layer_idx == self.config.start_layer:
                    print(f"[AutoPruneDebug] skip at layer {layer_idx}: short_seq={seq_len}")
                outputs = self._run_profiled_original_forward(
                    layer_idx,
                    self_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    kwargs,
                    stat_prefix="regular_forward",
                )
                self._pop_cached_proj_states(layer_idx)
                return finish(outputs)

            # Pure ablation path: directly drop a fixed number of visual tokens at start layer,
            # then keep the shortened sequence for following layers.
            if self.config.static_drop_tokens_at_start > 0 and layer_idx == self.config.start_layer:
                outputs = self._run_profiled_original_forward(
                    layer_idx,
                    self_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    kwargs,
                    stat_prefix="regular_forward",
                )

                layout = self._resolve_layout_state(seq_len, hidden_states.device)
                view_positions = layout["view_positions"]
                always_keep_positions = layout["always_keep_positions"]
                curr_vis_len = int(sum(int(view.numel()) for view in view_positions))
                if curr_vis_len <= 0:
                    return finish(outputs)

                keep_vis = max(1, curr_vis_len - int(self.config.static_drop_tokens_at_start))
                keep_vis = min(keep_vis, curr_vis_len)
                keep_counts = self._allocate_keep_counts([int(view.numel()) for view in view_positions], keep_vis)
                selected_view_positions = []
                for view_indices, keep_count in zip(view_positions, keep_counts):
                    if keep_count <= 0 or view_indices.numel() == 0:
                        selected_view_positions.append(torch.empty((batch_size, 0), dtype=torch.long, device=hidden_states.device))
                        continue
                    selected_view_positions.append(
                        view_indices[:keep_count].unsqueeze(0).expand(batch_size, -1)
                    )

                full_indices = self._build_full_indices(
                    batch_size,
                    seq_len,
                    always_keep_positions,
                    selected_view_positions,
                    hidden_states.device,
                )

                current_output = outputs[0]
                gather_t0 = time.perf_counter()
                expanded_indices = full_indices.unsqueeze(-1).expand(-1, -1, current_output.size(-1))
                pruned_hidden = current_output.gather(1, expanded_indices).contiguous()

                if self.global_indices is not None:
                    current_globals = self.global_indices
                    if current_globals.shape[0] == 1 and batch_size > 1:
                        current_globals = current_globals.expand(batch_size, -1)
                    self.global_indices = current_globals.gather(1, full_indices)
                self.runtime_stats["gather_seconds"] = float(
                    self.runtime_stats.get("gather_seconds", 0.0) + (time.perf_counter() - gather_t0)
                )

                c_h = int(keep_counts[0]) if len(keep_counts) > 0 else 0
                c_w = int(keep_counts[1]) if len(keep_counts) > 1 else max(0, keep_vis - c_h)
                self.view_boundary_tracker["current_h"] = c_h
                self.view_boundary_tracker["current_w"] = c_w
                self.runtime_stats["last_tokens_before"] = int(curr_vis_len)
                self.runtime_stats["last_tokens_after"] = int(c_h + c_w)
                self.runtime_stats["pruned_layers"] = int(self.runtime_stats.get("pruned_layers", 0) + 1)

                if self.config.debug:
                    print(
                        "[AutoPruneDebug] static_drop_tokens_at_start="
                        f"{self.config.static_drop_tokens_at_start}, keep_vis={keep_vis}"
                    )

                # Preserve original return signature by replacing only hidden_states.
                if isinstance(outputs, tuple):
                    return finish((pruned_hidden,) + outputs[1:])
                return finish(outputs)

            original_output_attentions = kwargs.get("output_attentions", False)
            if self.noop_control_mode and layer_idx >= self.config.start_layer:
                outputs = self._run_profiled_original_forward(
                    layer_idx,
                    self_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    kwargs,
                    stat_prefix="regular_forward",
                )
                return finish(outputs)

            if layer_idx < self.config.start_layer:
                outputs = self._run_profiled_original_forward(
                    layer_idx,
                    self_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    kwargs,
                    stat_prefix="regular_forward",
                )
                return finish(outputs)

            # True no-prune control: if the requested budget already keeps all visual
            # tokens, bypass all pruning-score collection and stay on the regular path.
            if not self.noop_control_mode and layer_idx == self.config.start_layer:
                layout = self._resolve_layout_state(seq_len, hidden_states.device)
                view_positions = layout["view_positions"]
                current_vis_len = int(sum(int(view.numel()) for view in view_positions))
                if current_vis_len > 0 and self._is_no_prune_budget(current_vis_len):
                    self.noop_control_mode = True
                    self.keep_percent_schedule = []
                    self.keep_count_schedule = []
                    self.compaction_layer_flags = []
                    self.runtime_stats["schedule_len"] = 0
                    self.runtime_stats["total_budget"] = float(current_vis_len)
                    self.runtime_stats["last_tokens_before"] = int(current_vis_len)
                    self.runtime_stats["last_tokens_after"] = int(current_vis_len)
                    self.runtime_stats["final_visual_tokens"] = float(current_vis_len)
                    self.runtime_stats["flops_reduction"] = 0.0
                    if self.config.debug:
                        print("[AutoPruneDebug] no-prune budget detected before score collection")
                    outputs = self._run_profiled_original_forward(
                        layer_idx,
                        self_layer,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        kwargs,
                        stat_prefix="regular_forward",
                    )
                    return finish(outputs)

            need_dynamic_metrics = (
                layer_idx == self.config.start_layer
                and not self.keep_percent_schedule
            )
            need_progressive_scores = (
                layer_idx > self.config.start_layer
                and bool(self.keep_percent_schedule)
                and not self.config.monitor_only
                and not self.config.disable_token_prune
                and not self.config.disable_progressive_prune
            )
            if need_progressive_scores:
                sched_idx = layer_idx - self.config.start_layer
                if 0 <= sched_idx < len(self.keep_percent_schedule):
                    if self.compaction_layer_flags and not self.compaction_layer_flags[sched_idx]:
                        need_progressive_scores = False
                    else:
                        ratios = self.keep_percent_schedule[sched_idx]
                        if ratios[0] >= 1.0 and ratios[1] >= 1.0:
                            need_progressive_scores = False
            need_attn_scores = need_dynamic_metrics or need_progressive_scores
            probe_stride = max(1, int(getattr(self.config, "manual_attention_probe_stride", 1) or 1))
            anchor_layers = self.manual_probe_anchor_layers or set(self.pruning_loc)
            should_probe_layer = need_dynamic_metrics or (
                need_progressive_scores
                and layer_idx in anchor_layers
                and ((layer_idx - self.config.start_layer) % probe_stride == 0)
            )
            should_reuse_scores = need_progressive_scores and self.config.manual_attention_probe and not should_probe_layer
            skip_non_anchor_prune = bool(
                should_reuse_scores and getattr(self.config, "prune_only_on_anchor_layers", False)
            )
            use_manual_probe = self.config.manual_attention_probe and need_attn_scores and should_probe_layer

            forward_kwargs = dict(kwargs)
            forced_output_attentions = (
                need_attn_scores and not use_manual_probe and not should_reuse_scores and not skip_non_anchor_prune and not original_output_attentions
            )
            if need_attn_scores and not use_manual_probe and not should_reuse_scores and not skip_non_anchor_prune:
                forward_kwargs["output_attentions"] = True

            # If this layer neither needs fresh pruning scores nor performs a
            # physical compaction, keep it on the regular path. This avoids
            # paying extra pruning-branch bookkeeping on pass-through layers.
            if not need_attn_scores and not skip_non_anchor_prune:
                outputs = self._run_profiled_original_forward(
                    layer_idx,
                    self_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    forward_kwargs,
                    stat_prefix="regular_forward",
                )
                return finish(outputs)

            if skip_non_anchor_prune:
                outputs = self._run_profiled_original_forward(
                    layer_idx,
                    self_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    forward_kwargs,
                    stat_prefix="regular_forward",
                )
                return finish(outputs)

            outputs = self._run_profiled_original_forward(
                layer_idx,
                self_layer,
                hidden_states,
                attention_mask,
                position_ids,
                forward_kwargs,
            )

            def clean_outputs(current_outputs):
                if original_output_attentions or not forced_output_attentions:
                    return current_outputs
                return (current_outputs[0],) + current_outputs[2:]

            attn_weights = None
            manual_probe = None
            if need_attn_scores and not use_manual_probe and (original_output_attentions or forced_output_attentions) and len(outputs) > 1:
                attn_weights = outputs[1]
            elif need_attn_scores and use_manual_probe:
                manual_probe = self._compute_manual_attention_probe(
                    layer_idx, self_layer, hidden_states, attention_mask, kwargs
                )

            if need_attn_scores and attn_weights is None and manual_probe is None and not should_reuse_scores:
                if self.config.debug and layer_idx == self.config.start_layer:
                    print(f"[AutoPruneDebug] attn_weights is None at layer {layer_idx}")
                return finish(clean_outputs(outputs))

            if layer_idx == self.config.start_layer:
                if manual_probe is not None:
                    query_positions = manual_probe["query_positions"]
                    view_positions = manual_probe["view_positions"]
                    attn_head = manual_probe["attn_head"]
                    attn_wrist = manual_probe["attn_wrist"]
                else:
                    layout = self._resolve_layout_state(seq_len, hidden_states.device)
                    query_positions = layout["query_positions"]
                    view_positions = layout["view_positions"]
                if len(view_positions) < 2 or query_positions.numel() <= 0:
                    return finish(clean_outputs(outputs))

                c_h = int(view_positions[0].numel())
                c_w = int(view_positions[1].numel())
                if c_h <= 0 or c_w <= 0:
                    return finish(clean_outputs(outputs))

                current_vis_len = c_h + c_w
                self.view_boundary_tracker["current_h"] = c_h
                self.view_boundary_tracker["current_w"] = c_w

                if self._is_no_prune_budget(current_vis_len):
                    self.noop_control_mode = True
                    self.keep_percent_schedule = []
                    self.keep_count_schedule = []
                    self.runtime_stats["schedule_len"] = 0
                    self.runtime_stats["total_budget"] = float(current_vis_len)
                    self.runtime_stats["last_tokens_before"] = int(current_vis_len)
                    self.runtime_stats["last_tokens_after"] = int(current_vis_len)
                    self.runtime_stats["final_visual_tokens"] = float(current_vis_len)
                    self.runtime_stats["flops_reduction"] = 0.0
                    if self.config.debug:
                        print("[AutoPruneDebug] no-prune budget detected, bypass progressive path")
                    return finish(clean_outputs(outputs))

                if manual_probe is None:
                    attn_queries = attn_weights.index_select(2, query_positions)
                    attn_head = attn_queries.index_select(3, view_positions[0])
                    attn_wrist = attn_queries.index_select(3, view_positions[1])

                detailed_timing = bool(getattr(self.config, "detailed_timing", True))
                metric_t0 = time.perf_counter() if detailed_timing else None

                def get_metric(view_attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    norm = view_attn / (view_attn.sum(dim=-1, keepdim=True) + 1e-9)
                    mi_vec, kl_vec = calculate_mi_and_kl(norm[0])
                    denom = max(1, norm.shape[2])
                    return mi_vec.sum() / denom, kl_vec.sum() / denom

                mi_h, kl_h = get_metric(attn_head)
                mi_w, kl_w = get_metric(attn_wrist)
                nmi_h = _finite_scalar(mi_h / (kl_h + 1e-6), default=1e-6, min_value=1e-6)
                nmi_w = _finite_scalar(mi_w / (kl_w + 1e-6), default=1e-6, min_value=1e-6)
                if detailed_timing and metric_t0 is not None:
                    self.runtime_stats["metric_seconds"] = float(
                        self.runtime_stats.get("metric_seconds", 0.0) + (time.perf_counter() - metric_t0)
                    )

                if self.cached_rank is None:
                    rank_t0 = time.perf_counter() if detailed_timing else None
                    text_start = int(query_positions[0].item())
                    self.cached_rank = self._calculate_effective_rank(hidden_states, slice(text_start, seq_len))
                    if detailed_timing and rank_t0 is not None:
                        self.runtime_stats["rank_seconds"] = float(
                            self.runtime_stats.get("rank_seconds", 0.0) + (time.perf_counter() - rank_t0)
                        )

                schedule_t0 = time.perf_counter() if detailed_timing else None
                self.cached_rank = _finite_scalar(self.cached_rank, default=1.0, min_value=1.0)
                nmi_avg = (nmi_h + nmi_w) / 2.0
                complexity_signal = _finite_scalar(self.cached_rank / torch.clamp(nmi_avg, min=1e-6), default=float(self.config.mu), min_value=1e-6)
                complexity_score = float(
                    logistic_complexity_score(
                        complexity_signal,
                        getattr(self.config, "adaptive_layer_mu", self.config.mu),
                        getattr(self.config, "adaptive_layer_tau", self.config.tau),
                    ).item()
                )

                kl_tensor = torch.stack([
                    _finite_scalar(kl_h, default=0.0, min_value=0.0),
                    _finite_scalar(kl_w, default=0.0, min_value=0.0),
                ])
                budget_mode = str(getattr(self.config, "budget_mode", "absolute") or "absolute").lower()
                absolute_anchor_mode = str(getattr(self.config, "absolute_anchor_mode", "none") or "none").lower()
                configured_visual_budget_cap = int(self.config.target_token_num) if int(self.config.target_token_num) > 0 else None
                raw_total_budget = float(current_vis_len)
                budget_fallback_used = False
                budget_clamped_by_target = False
                absolute_anchor_budget = None
                absolute_controller_ratio = 1.0
                elbow_strengths: List[float] = []
                realized_coverages: List[float] = []

                if self.config.monitor_only:
                    total_budget = torch.tensor(float(current_vis_len), device=hidden_states.device)
                    keep_counts = [c_h, c_w]
                else:
                    if budget_mode == "elbow":
                        keep_counts, elbow_strengths = allocate_keep_counts_by_elbow_two_views(
                            attn_head,
                            attn_wrist,
                            self.config.min_keep_per_view,
                        )
                        raw_total_budget = float(sum(keep_counts))
                        total_keep = int(round(raw_total_budget))
                    else:
                        if absolute_anchor_mode == "elbow":
                            anchor_keep_counts, elbow_strengths = allocate_keep_counts_by_elbow_two_views(
                                attn_head,
                                attn_wrist,
                                self.config.min_keep_per_view,
                            )
                            absolute_anchor_budget = float(sum(anchor_keep_counts))

                        raw_total_budget, auto_lower_bound, auto_upper_bound, normalized_signal, reference_signal, absolute_anchor_budget, absolute_controller_ratio = decide_absolute_budget_v2(
                            self.config,
                            complexity_signal,
                            self.cached_rank,
                            [c_h, c_w],
                            current_vis_len,
                            anchor_budget=absolute_anchor_budget,
                        )
                        if not math.isfinite(raw_total_budget):
                            raw_total_budget = float(min(current_vis_len, max(1, self.config.target_token_num)))
                            budget_fallback_used = True
                        total_keep = int(round(raw_total_budget))

                        temp = max(float(self.config.allocation_temp), 1e-4)
                        alloc_signal = torch.log1p(torch.clamp(kl_tensor, min=0.0))
                        alloc_weights = torch.softmax(alloc_signal / temp, dim=0)
                        alloc_weights = torch.nan_to_num(alloc_weights, nan=0.0, posinf=0.0, neginf=0.0)
                        if float(alloc_weights.sum().item()) <= 0.0:
                            alloc_weights = torch.ones_like(alloc_weights) / max(1, alloc_weights.numel())
                        if configured_visual_budget_cap is not None:
                            clamped_keep = min(total_keep, configured_visual_budget_cap)
                            budget_clamped_by_target = clamped_keep != total_keep
                            total_keep = clamped_keep
                        total_keep = max(1, min(total_keep, current_vis_len))
                        keep_counts = allocate_keep_counts(
                            [c_h, c_w],
                            total_keep,
                            min_keep_per_view=self.config.min_keep_per_view,
                            weights=alloc_weights,
                        )
                        total_budget = torch.tensor(float(total_keep), device=hidden_states.device)

                    if budget_mode == "elbow":
                        if configured_visual_budget_cap is not None:
                            clamped_keep = min(total_keep, configured_visual_budget_cap)
                            budget_clamped_by_target = clamped_keep != total_keep
                            total_keep = clamped_keep
                        total_keep = max(1, min(total_keep, current_vis_len))
                        temp = max(float(self.config.allocation_temp), 1e-4)
                        alloc_signal = torch.log1p(torch.clamp(kl_tensor, min=0.0))
                        alloc_weights = torch.softmax(alloc_signal / temp, dim=0)
                        alloc_weights = torch.nan_to_num(alloc_weights, nan=0.0, posinf=0.0, neginf=0.0)
                        if float(alloc_weights.sum().item()) <= 0.0:
                            alloc_weights = torch.ones_like(alloc_weights) / max(1, alloc_weights.numel())
                        if sum(keep_counts) != total_keep:
                            keep_counts = allocate_keep_counts(
                                [c_h, c_w],
                                total_keep,
                                min_keep_per_view=self.config.min_keep_per_view,
                                weights=alloc_weights,
                            )
                        total_budget = torch.tensor(float(total_keep), device=hidden_states.device)

                total_budget = torch.clamp(
                    total_budget,
                    min=float(min(2, current_vis_len)),
                    max=float(current_vis_len),
                )

                self.runtime_stats["nmi_head"] = float(nmi_h.detach().cpu())
                self.runtime_stats["nmi_wrist"] = float(nmi_w.detach().cpu())
                self.runtime_stats["effective_rank"] = float(self.cached_rank.detach().cpu())
                self.runtime_stats["total_budget"] = float(total_budget.detach().cpu())
                self.runtime_stats["complexity_signal"] = float(complexity_signal.detach().cpu())
                self.runtime_stats["complexity_score"] = float(complexity_score)
                self.runtime_stats["budget_mode"] = budget_mode
                self.runtime_stats["raw_total_budget"] = float(raw_total_budget)
                self.runtime_stats["absolute_anchor_mode"] = absolute_anchor_mode
                self.runtime_stats["absolute_budget_scale"] = float(getattr(self.config, "absolute_budget_scale", 1.0))
                self.runtime_stats["absolute_anchor_budget"] = (
                    float(absolute_anchor_budget) if absolute_anchor_budget is not None else None
                )
                self.runtime_stats["absolute_controller_ratio"] = float(absolute_controller_ratio)
                self.runtime_stats["budget_fallback_used"] = int(bool(budget_fallback_used))
                self.runtime_stats["budget_clamped_by_target"] = int(bool(budget_clamped_by_target))
                if elbow_strengths:
                    self.runtime_stats["elbow_strengths"] = [float(v) for v in elbow_strengths]
                if realized_coverages:
                    self.runtime_stats["realized_view_coverages"] = [float(v) for v in realized_coverages]

                max_h = torch.tensor(float(max(c_h, 1)), device=hidden_states.device)
                max_w = torch.tensor(float(max(c_w, 1)), device=hidden_states.device)
                t_h = torch.tensor(float(keep_counts[0]), device=hidden_states.device)
                t_w = torch.tensor(float(keep_counts[1]), device=hidden_states.device)

                no_prune_control = keep_counts[0] >= c_h and keep_counts[1] >= c_w
                if no_prune_control:
                    if self.config.disable_progressive_prune:
                        self.keep_percent_schedule = [[1.0, 1.0]]
                        self.keep_count_schedule = [[c_h, c_w]]
                        self.compaction_layer_flags = [False]
                    else:
                        self.keep_percent_schedule = [
                            [1.0, 1.0] for _ in range(max(1, len(self.pruning_loc)))
                        ]
                        self.keep_count_schedule = [
                            [c_h, c_w] for _ in range(max(1, len(self.pruning_loc)))
                        ]
                        self.compaction_layer_flags = [False for _ in range(max(1, len(self.pruning_loc)))]
                elif self.config.disable_progressive_prune:
                    ratio_h = torch.clamp(t_h / max_h, min=0.001, max=1.0)
                    ratio_w = torch.clamp(t_w / max_w, min=0.001, max=1.0)
                    self.keep_percent_schedule = [[float(ratio_h.item()), float(ratio_w.item())]]
                    self.keep_count_schedule = [[int(keep_counts[0]), int(keep_counts[1])]]
                    self.compaction_layer_flags = [True]
                else:
                    num_sched_layers = max(1, len(self.pruning_loc))
                    x0_h = self._effective_x0_for_target(int(keep_counts[0]), c_h)
                    x0_w = self._effective_x0_for_target(int(keep_counts[1]), c_w)
                    s_h = self._generate_schedule_aligned(t_h * num_sched_layers, x0_h, self.config.k_shape)
                    s_w = self._generate_schedule_aligned(t_w * num_sched_layers, x0_w, self.config.k_shape)
                    self.runtime_stats["effective_x0_head"] = float(x0_h)
                    self.runtime_stats["effective_x0_wrist"] = float(x0_w)
                    self.manual_probe_anchor_layers = self._resolve_manual_probe_anchor_layers((x0_h + x0_w) / 2.0)
                    dense_h = self._simulate_dense_keep_counts(c_h, s_h, self.config.min_keep_per_view)
                    dense_w = self._simulate_dense_keep_counts(c_w, s_w, self.config.min_keep_per_view)
                    self.keep_percent_schedule = torch.stack([s_h, s_w], dim=1).tolist()
                    full_keep_count_schedule = [
                        [int(k_h_val), int(k_w_val)]
                        for k_h_val, k_w_val in zip(dense_h, dense_w)
                    ]

                    if self.config.piecewise_linear_keep_schedule:
                        max_shots = int(getattr(self.config, "physical_compaction_max_shots", 0) or 0)
                        if max_shots > 0:
                            if self.config.piecewise_linear_hard_final_cap:
                                keep_count_schedule, compaction_flags = self._build_sparse_compaction_keep_counts(
                                    dense_h,
                                    dense_w,
                                    c_h,
                                    c_w,
                                    int(keep_counts[0]),
                                    int(keep_counts[1]),
                                    max_shots,
                                )
                            else:
                                keep_count_schedule, compaction_flags = self._build_sparse_area_matched_keep_counts(
                                    dense_h,
                                    dense_w,
                                    c_h,
                                    c_w,
                                    max_shots,
                                )
                        else:
                            sched_h = self._build_piecewise_linear_keep_counts(dense_h, c_h, keep_counts[0])
                            sched_w = self._build_piecewise_linear_keep_counts(dense_w, c_w, keep_counts[1])
                            keep_count_schedule = [
                                [int(k_h_val), int(k_w_val)]
                                for k_h_val, k_w_val in zip(sched_h, sched_w)
                            ]
                            prev_h, prev_w = c_h, c_w
                            compaction_flags = []
                            for k_h_val, k_w_val in keep_count_schedule:
                                should_compact = (int(k_h_val) < int(prev_h)) or (int(k_w_val) < int(prev_w))
                                compaction_flags.append(bool(should_compact))
                                prev_h, prev_w = int(k_h_val), int(k_w_val)
                    else:
                        keep_count_schedule = [list(counts) for counts in full_keep_count_schedule]
                        prev_h, prev_w = c_h, c_w
                        compaction_flags = []
                        for k_h_val, k_w_val in keep_count_schedule:
                            should_compact = (int(k_h_val) < int(prev_h)) or (int(k_w_val) < int(prev_w))
                            compaction_flags.append(bool(should_compact))
                            prev_h, prev_w = int(k_h_val), int(k_w_val)

                    active_pruning_loc = [int(layer_idx) for layer_idx in self.pruning_loc]
                    active_schedule = [list(map(int, counts)) for counts in keep_count_schedule]
                    selected_pruning_mode = "progressive_full_dense"
                    if bool(getattr(self.config, "enable_adaptive_layer_policy", False)):
                        active_pruning_loc, active_schedule = compress_keep_count_schedule(self.pruning_loc, keep_count_schedule)
                        layer_policy_loc, adaptive_layer_score = resolve_adaptive_pruning_layer_count(
                            self.config,
                            active_pruning_loc,
                            complexity_signal,
                        )
                        if layer_policy_loc:
                            schedule_index = {
                                int(layer_idx): idx for idx, layer_idx in enumerate(active_pruning_loc)
                            }
                            active_pruning_loc = [int(layer_idx) for layer_idx in layer_policy_loc]
                            active_schedule = [
                                list(map(int, active_schedule[schedule_index[int(layer_idx)]]))
                                for layer_idx in active_pruning_loc
                            ]
                            self.runtime_stats["adaptive_layer_count_score"] = float(adaptive_layer_score)
                        if active_pruning_loc and bool(getattr(self.config, "enable_oneshot_fallback", False)) and should_use_oneshot_fallback(
                            self.config,
                            active_schedule,
                            [int(keep_counts[0]), int(keep_counts[1])],
                        ):
                            active_pruning_loc = [int(self.pruning_loc[0])]
                            active_schedule = [[int(keep_counts[0]), int(keep_counts[1])]]
                            selected_pruning_mode = "schedule_compressed_1layer"
                        elif active_pruning_loc:
                            selected_pruning_mode = "schedule_compressed_multilayer"

                    self.keep_count_schedule, self.compaction_layer_flags = self._expand_active_keep_schedule(
                        self.pruning_loc,
                        active_pruning_loc,
                        active_schedule,
                    )
                    self.keep_percent_schedule = self._counts_to_keep_ratios(self.keep_count_schedule, c_h, c_w)
                    self.manual_probe_anchor_layers = set(active_pruning_loc) if active_pruning_loc else {self.config.start_layer}
                    self.runtime_stats["selected_pruning_mode"] = selected_pruning_mode
                    self.runtime_stats["dense_keep_count_schedule"] = [
                        [int(k_h_val), int(k_w_val)] for k_h_val, k_w_val in full_keep_count_schedule
                    ]
                    self.runtime_stats["active_pruning_layer_indices"] = [int(layer_idx) for layer_idx in active_pruning_loc]
                    self.runtime_stats["dense_pruning_layer_indices"] = [int(layer_idx) for layer_idx in self.pruning_loc]
                    self.runtime_stats["compressed_schedule_length"] = int(len(active_schedule))
                self.runtime_stats["schedule_len"] = len(self.keep_percent_schedule)
                self.runtime_stats["planned_compaction_layers"] = int(sum(1 for flag in self.compaction_layer_flags if flag))
                if detailed_timing and schedule_t0 is not None:
                    self.runtime_stats["budget_schedule_seconds"] = float(
                        self.runtime_stats.get("budget_schedule_seconds", 0.0) + (time.perf_counter() - schedule_t0)
                    )

                if self.config.debug:
                    print(
                        "[AutoPruneDebug] "
                        f"prefill_seq_len={self.runtime_stats['prefill_seq_len']} "
                        f"budget={self.runtime_stats['total_budget']:.2f} "
                        f"nmi_h={self.runtime_stats['nmi_head']:.4f} "
                        f"nmi_w={self.runtime_stats['nmi_wrist']:.4f} "
                        f"rank={self.runtime_stats['effective_rank']:.2f}"
                    )

            if self.config.monitor_only or not self.keep_percent_schedule:
                if self.config.debug and layer_idx == self.config.start_layer:
                    print("[AutoPruneDebug] no schedule available after start-layer processing")
                return finish(clean_outputs(outputs))

            if self.config.disable_token_prune:
                if self.config.debug and layer_idx == self.config.start_layer:
                    print("[AutoPruneDebug] disable_token_prune=True, skip token selection/gather branch")
                return finish(clean_outputs(outputs))

            if skip_non_anchor_prune:
                return finish(clean_outputs(outputs))

            sched_idx = layer_idx - self.config.start_layer
            if sched_idx < 0 or sched_idx >= len(self.keep_percent_schedule):
                return finish(clean_outputs(outputs))

            if self.config.disable_progressive_prune and sched_idx > 0:
                if self.config.debug and layer_idx == self.config.start_layer + 1:
                    print("[AutoPruneDebug] disable_progressive_prune=True, skip pruning after start layer")
                return finish(clean_outputs(outputs))

            layout = self._resolve_layout_state(seq_len, hidden_states.device)
            query_positions = layout["query_positions"]
            view_positions = layout["view_positions"]
            always_keep_positions = layout["always_keep_positions"]
            if len(view_positions) < 2 or query_positions.numel() <= 0:
                return finish(clean_outputs(outputs))

            c_h = int(view_positions[0].numel())
            c_w = int(view_positions[1].numel())
            curr_vis_len = c_h + c_w
            if curr_vis_len <= 0:
                return finish(clean_outputs(outputs))

            if manual_probe is not None:
                score_h = manual_probe["attn_head"].mean(dim=1).mean(dim=1)
                score_w = manual_probe["attn_wrist"].mean(dim=1).mean(dim=1)
            else:
                if attn_weights is not None:
                    attn_queries = attn_weights.index_select(2, query_positions)
                    # Per-layer token selection always uses the current layer's attention scores.
                    score_h = attn_queries.index_select(3, view_positions[0]).mean(dim=1).mean(dim=1)
                    score_w = attn_queries.index_select(3, view_positions[1]).mean(dim=1).mean(dim=1)
                else:
                    reused_scores = self._reuse_cached_view_scores(batch_size, view_positions, hidden_states.device)
                    if reused_scores is None:
                        return finish(clean_outputs(outputs))
                    score_h, score_w = reused_scores

            if manual_probe is not None or attn_weights is not None:
                self._cache_view_scores(view_positions, score_h, score_w)

            if self.keep_count_schedule and 0 <= sched_idx < len(self.keep_count_schedule):
                target_h, target_w = self.keep_count_schedule[sched_idx]
                k_h = 0 if c_h <= 0 else max(min(int(target_h), c_h), min(self.config.min_keep_per_view, c_h))
                k_w = 0 if c_w <= 0 else max(min(int(target_w), c_w), min(self.config.min_keep_per_view, c_w))
            else:
                ratios = self.keep_percent_schedule[sched_idx]
                k_h = 0 if c_h <= 0 else max(int(c_h * ratios[0]), min(self.config.min_keep_per_view, c_h))
                k_w = 0 if c_w <= 0 else max(int(c_w * ratios[1]), min(self.config.min_keep_per_view, c_w))

            if k_h >= c_h and k_w >= c_w:
                self.runtime_stats["last_tokens_before"] = int(curr_vis_len)
                self.runtime_stats["last_tokens_after"] = int(curr_vis_len)
                self.runtime_stats["final_visual_tokens"] = float(curr_vis_len)
                self.runtime_stats["flops_reduction"] = 0.0
                return finish(clean_outputs(outputs))

            target_total = int(k_h + k_w)
            force_compaction = layer_idx in anchor_layers or sched_idx == (len(self.keep_percent_schedule) - 1)
            compaction_delta_threshold = int(getattr(self.config, "compaction_delta_threshold_tokens", 0) or 0)
            if (
                not force_compaction
                and compaction_delta_threshold > 0
                and (curr_vis_len - target_total) < compaction_delta_threshold
            ):
                self.runtime_stats["last_tokens_before"] = int(curr_vis_len)
                self.runtime_stats["last_tokens_after"] = int(curr_vis_len)
                self.runtime_stats["final_visual_tokens"] = float(curr_vis_len)
                self.runtime_stats["flops_reduction"] = _estimate_flops_reduction(
                    seq_len,
                    curr_vis_len,
                    curr_vis_len,
                )
                self.runtime_stats["compaction_skipped_layers"] = int(
                    self.runtime_stats.get("compaction_skipped_layers", 0) + 1
                )
                return finish(clean_outputs(outputs))

            self.runtime_stats["last_tokens_before"] = int(curr_vis_len)
            self.runtime_stats["last_tokens_after"] = int(target_total)

            topk_t0 = time.perf_counter()
            _, idx_h = torch.topk(score_h, k_h, dim=1)
            _, idx_w = torch.topk(score_w, k_w, dim=1)
            idx_h, _ = idx_h.sort(dim=1)
            idx_w, _ = idx_w.sort(dim=1)
            self.runtime_stats["topk_seconds"] = float(
                self.runtime_stats.get("topk_seconds", 0.0) + (time.perf_counter() - topk_t0)
            )

            detailed_timing = bool(getattr(self.config, "detailed_timing", True))
            bookkeeping_t0 = time.perf_counter() if detailed_timing else None
            dev = hidden_states.device
            view0_base = view_positions[0].unsqueeze(0).expand(batch_size, -1)
            view1_base = view_positions[1].unsqueeze(0).expand(batch_size, -1)
            selected_h = view0_base.gather(1, idx_h) if k_h > 0 else torch.empty((batch_size, 0), dtype=torch.long, device=dev)
            selected_w = view1_base.gather(1, idx_w) if k_w > 0 else torch.empty((batch_size, 0), dtype=torch.long, device=dev)
            full_indices = self._build_full_indices(
                batch_size,
                seq_len,
                always_keep_positions,
                [selected_h, selected_w],
                dev,
            )

            current_output = outputs[0]
            gather_t0 = time.perf_counter()
            expanded_indices = full_indices.unsqueeze(-1).expand(-1, -1, current_output.size(-1))
            pruned_hidden = current_output.gather(1, expanded_indices).contiguous()

            if self.global_indices is not None:
                current_globals = self.global_indices
                if current_globals.shape[0] == 1 and batch_size > 1:
                    current_globals = current_globals.expand(batch_size, -1)
                self.global_indices = current_globals.gather(1, full_indices)
            if detailed_timing and bookkeeping_t0 is not None:
                self.runtime_stats["compaction_bookkeeping_seconds"] = float(
                    self.runtime_stats.get("compaction_bookkeeping_seconds", 0.0) + (time.perf_counter() - bookkeeping_t0)
                )
            self.runtime_stats["gather_seconds"] = float(
                self.runtime_stats.get("gather_seconds", 0.0) + (time.perf_counter() - gather_t0)
            )

            self.view_boundary_tracker["current_h"] = k_h
            self.view_boundary_tracker["current_w"] = k_w
            self.runtime_stats["pruned_layers"] = int(self.runtime_stats.get("pruned_layers", 0) + 1)
            self.runtime_stats["final_visual_tokens"] = float(target_total)
            self.runtime_stats["flops_reduction"] = _estimate_flops_reduction(
                seq_len,
                curr_vis_len,
                target_total,
            )

            # Preserve original return signature (with/without attentions) after replacing hidden_states.
            return finish(clean_outputs((pruned_hidden,) + outputs[1:]))

        return new_forward

    def get_runtime_stats(self) -> Dict[str, Any]:
        return dict(self.runtime_stats)

    def get_flops_reduction(self) -> float:
        return float(self.runtime_stats.get("flops_reduction", 0.0) or 0.0)


def maybe_apply_nora_autoprune(model, config: Optional[NoraAutoPruneConfig]) -> Optional[NoraAutoPrunePatch]:
    if config is None or not getattr(config, "enabled", False):
        return None
    patch = NoraAutoPrunePatch(model, config)
    patch.apply()
    return patch


__all__ = [
    "NoraAutoPruneConfig",
    "NoraAutoPrunePatch",
    "maybe_apply_nora_autoprune",
]
