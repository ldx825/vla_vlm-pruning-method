import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .autoprune_mantis_common import (
    AdaptiveBudgetStrategy,
    MantisAutoPruneConfig,
    calculate_mi_and_kl,
    compute_complexity_signal,
    logistic_complexity_score,
)


def _finite_scalar(
    value: torch.Tensor,
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> torch.Tensor:
    scalar = torch.nan_to_num(value.float(), nan=default, posinf=default, neginf=default)
    if min_value is not None or max_value is not None:
        scalar = torch.clamp(
            scalar,
            min=min_value if min_value is not None else -float("inf"),
            max=max_value if max_value is not None else float("inf"),
        )
    return scalar


def calculate_effective_rank(hidden_states: torch.Tensor, query_positions: torch.Tensor) -> torch.Tensor:
    try:
        if query_positions.numel() == 0:
            return torch.tensor(1.0, device=hidden_states.device)
        text_features = hidden_states[0].index_select(0, query_positions).float()
        if text_features.numel() == 0:
            return torch.tensor(1.0, device=hidden_states.device)
        text_features = torch.nan_to_num(text_features, nan=0.0, posinf=0.0, neginf=0.0)
        sigmas = torch.linalg.svdvals(text_features)
        sigmas = torch.nan_to_num(sigmas, nan=0.0, posinf=0.0, neginf=0.0)
        sigma_sum = sigmas.sum()
        if sigma_sum < 1e-9:
            return torch.tensor(1.0, device=hidden_states.device)
        probs = sigmas / sigma_sum
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))
        return _finite_scalar(torch.exp(entropy), default=1.0, min_value=1.0)
    except Exception:
        return torch.tensor(1.0, device=hidden_states.device)


def _select_query_rows(attn_weights: torch.Tensor, query_positions: torch.Tensor) -> torch.Tensor:
    if query_positions.numel() <= 0:
        return attn_weights[:, :, :0, :]
    if attn_weights.shape[2] == int(query_positions.numel()):
        return attn_weights
    return attn_weights.index_select(2, query_positions)


def generate_schedule_aligned(
    pruning_loc: List[int],
    config: MantisAutoPruneConfig,
    target_total: torch.Tensor,
    start_count: int,
    device: torch.device,
) -> torch.Tensor:
    num_pruning_layers = max(1, len(pruning_loc))
    if start_count <= 0:
        return torch.ones(num_pruning_layers, device=device)

    layer_indices = torch.tensor(pruning_loc, device=device, dtype=torch.float32)
    raw_curve = 1.0 / (1.0 + torch.exp(config.k_shape * (layer_indices - float(config.x0))))
    scales = torch.linspace(0.01, 100.0, 2000, device=device)
    clamped = torch.clamp(scales.unsqueeze(1) * raw_curve.unsqueeze(0), max=1.0)
    totals = clamped.sum(dim=1) * int(start_count)
    best_idx = torch.argmin((totals - target_total).abs())
    best_curve = clamped[best_idx]
    prev_curve = torch.cat([torch.ones(1, device=device), best_curve[:-1]])
    return torch.clamp(best_curve / (prev_curve + 1e-9), min=0.001, max=1.0)


def select_progressive_pruning_layers(pruning_loc: List[int], target_layers: int) -> List[int]:
    if not pruning_loc:
        return []
    target_layers = max(1, min(int(target_layers), len(pruning_loc)))
    if target_layers >= len(pruning_loc):
        return [int(layer_idx) for layer_idx in pruning_loc]
    if target_layers == 1:
        return [int(pruning_loc[0])]

    selected: List[int] = []
    seen = set()
    last_index = len(pruning_loc) - 1
    for offset in range(target_layers):
        loc_idx = int(round(offset * last_index / max(1, target_layers - 1)))
        layer_idx = int(pruning_loc[loc_idx])
        if layer_idx in seen:
            continue
        selected.append(layer_idx)
        seen.add(layer_idx)

    if len(selected) < target_layers:
        for layer_idx in pruning_loc:
            if int(layer_idx) in seen:
                continue
            selected.append(int(layer_idx))
            seen.add(int(layer_idx))
            if len(selected) >= target_layers:
                break
    return sorted(selected)


def resolve_adaptive_pruning_layer_count(
    config: MantisAutoPruneConfig,
    pruning_loc: List[int],
    complexity_signal: torch.Tensor,
) -> Tuple[List[int], float]:
    if not pruning_loc:
        return [], 0.0

    filtered_choices = []
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
    target_layers = int(filtered_choices[bucket_index])
    return select_progressive_pruning_layers(pruning_loc, target_layers), complexity_score


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
    config: MantisAutoPruneConfig,
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
    budget_margin = max(2, int(round(target_total * float(getattr(config, "oneshot_fallback_budget_margin", 0.15)))))
    near_target_early = second_total <= target_total + budget_margin

    total_drop = max(1, first_total - target_total)
    frontloaded_drop = max(0, first_total - second_total)
    frontload_ratio = frontloaded_drop / float(total_drop)
    enough_frontloaded = frontload_ratio >= float(getattr(config, "oneshot_fallback_frontload_ratio", 0.5))

    return bool(near_target_early and enough_frontloaded)


def _repair_monotonic_counts(
    counts: List[int],
    *,
    start_count: int,
    floor_count: int,
) -> List[int]:
    if not counts:
        return counts

    repaired = [max(floor_count, min(start_count, int(value))) for value in counts]
    repaired[0] = min(repaired[0], int(start_count))
    for idx in range(1, len(repaired)):
        repaired[idx] = min(repaired[idx], repaired[idx - 1])
        repaired[idx] = max(repaired[idx], int(floor_count))
    return repaired


def _match_total_with_monotonic_budget(
    counts: List[int],
    *,
    target_total: int,
    start_count: int,
    floor_count: int,
) -> List[int]:
    counts = _repair_monotonic_counts(counts, start_count=start_count, floor_count=floor_count)
    current_total = int(sum(counts))
    target_total = int(max(len(counts) * floor_count, min(len(counts) * start_count, target_total)))

    if current_total == target_total:
        return counts

    if current_total > target_total:
        excess = current_total - target_total
        while excess > 0:
            changed = False
            for idx in range(len(counts)):
                lower = floor_count if idx == len(counts) - 1 else counts[idx + 1]
                if counts[idx] <= lower:
                    continue
                counts[idx] -= 1
                excess -= 1
                changed = True
                if excess == 0:
                    break
            if not changed:
                break
        return counts

    deficit = target_total - current_total
    while deficit > 0:
        changed = False
        for idx in range(len(counts) - 1, -1, -1):
            upper = start_count if idx == 0 else counts[idx - 1]
            if counts[idx] >= upper:
                continue
            counts[idx] += 1
            deficit -= 1
            changed = True
            if deficit == 0:
                break
        if not changed:
            break
    return counts


def simulate_dense_keep_counts(start_count: int, step_ratios: torch.Tensor, floor_count: int) -> List[int]:
    current = int(start_count)
    lower_bound = max(1, min(int(floor_count), int(start_count)))
    dense_counts: List[int] = []
    for ratio in step_ratios.tolist():
        next_count = int(current * float(ratio))
        next_count = max(lower_bound, next_count)
        next_count = min(next_count, current)
        dense_counts.append(int(next_count))
        current = int(next_count)
    return dense_counts


def allocate_keep_counts(
    view_lengths: List[int],
    total_keep: int,
    min_keep_per_view: int,
    weights: Optional[torch.Tensor] = None,
) -> List[int]:
    total_capacity = sum(int(length) for length in view_lengths)
    if total_capacity <= 0:
        return [0] * len(view_lengths)

    total_keep = max(1, min(int(total_keep), total_capacity))
    non_empty = [idx for idx, length in enumerate(view_lengths) if int(length) > 0]
    if not non_empty:
        return [0] * len(view_lengths)

    counts = [0] * len(view_lengths)

    if total_keep < len(non_empty):
        if weights is None:
            priority = list(non_empty)
        else:
            order = torch.argsort(weights, descending=True).tolist()
            priority = [idx for idx in order if idx in non_empty]
        for idx in priority[:total_keep]:
            counts[idx] = 1
        return counts

    base_floor = []
    for idx, length in enumerate(view_lengths):
        if idx not in non_empty:
            base_floor.append(0)
            continue
        base_floor.append(min(int(length), int(min_keep_per_view)))

    if sum(base_floor) > total_keep:
        base_floor = [1 if idx in non_empty else 0 for idx in range(len(view_lengths))]

    counts = list(base_floor)
    remaining = int(total_keep - sum(counts))
    if remaining <= 0:
        return counts

    if weights is None:
        weights = torch.tensor(
            [float(max(1, view_lengths[idx] - counts[idx])) for idx in range(len(view_lengths))],
            dtype=torch.float32,
        )
    weights = weights.to(dtype=torch.float32)
    if torch.sum(weights) <= 0:
        weights = torch.ones_like(weights)
    weights = weights / torch.sum(weights)

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


def allocate_keep_counts_by_coverage(
    attn_queries: torch.Tensor,
    view_positions: List[torch.Tensor],
    target_coverage: float,
    min_keep_per_view: int,
) -> Tuple[List[int], List[float]]:
    keep_counts: List[int] = []
    realized_coverages: List[float] = []
    target_coverage = float(max(0.0, min(0.999, target_coverage)))

    for view_indices in view_positions:
        view_len = int(view_indices.numel())
        if view_len <= 0:
            keep_counts.append(0)
            realized_coverages.append(0.0)
            continue

        attn_view = attn_queries.index_select(3, view_indices)
        score = attn_view.mean(dim=1).mean(dim=1)[0].float()
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score = torch.clamp(score, min=0.0)
        score_sum = float(score.sum().item())
        if score_sum <= 1e-9:
            keep_count = max(1, min(view_len, int(min_keep_per_view)))
            keep_counts.append(keep_count)
            realized_coverages.append(float(keep_count) / float(max(1, view_len)))
            continue

        sorted_score, _ = torch.sort(score, descending=True)
        normalized = sorted_score / max(score_sum, 1e-9)
        cdf = torch.cumsum(normalized, dim=0)
        keep_idx = int(torch.searchsorted(cdf, torch.tensor(target_coverage, device=cdf.device)).item())
        keep_count = min(view_len, keep_idx + 1)
        keep_count = max(1, keep_count)
        keep_count = max(min(view_len, int(min_keep_per_view)), keep_count)
        keep_counts.append(int(keep_count))
        realized_coverages.append(float(cdf[min(keep_count - 1, view_len - 1)].item()))

    return keep_counts, realized_coverages


def allocate_keep_counts_by_elbow(
    attn_queries: torch.Tensor,
    view_positions: List[torch.Tensor],
    min_keep_per_view: int,
) -> Tuple[List[int], List[float]]:
    keep_counts: List[int] = []
    elbow_strengths: List[float] = []

    for view_indices in view_positions:
        view_len = int(view_indices.numel())
        if view_len <= 0:
            keep_counts.append(0)
            elbow_strengths.append(0.0)
            continue

        attn_view = attn_queries.index_select(3, view_indices)
        score = attn_view.mean(dim=1).mean(dim=1)[0].float()
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
        keep_count = knee_idx + 1
        keep_count = max(1, keep_count)
        keep_count = max(min(view_len, int(min_keep_per_view)), keep_count)
        keep_counts.append(int(keep_count))
        elbow_strengths.append(float(gain[knee_idx].item()))

    return keep_counts, elbow_strengths


def decide_absolute_budget_v2(
    config: MantisAutoPruneConfig,
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
            int(math.ceil(float(effective_rank.item()) * float(max(1.0, config.warm_keep_factor)))),
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


def build_keep_count_schedule(
    config: MantisAutoPruneConfig,
    pruning_loc: List[int],
    view_lengths: List[int],
    target_keep_per_view: List[int],
    device: torch.device,
) -> List[List[int]]:
    num_pruning_layers = max(1, len(pruning_loc))
    if config.disable_progressive_prune:
        return [[int(target) for target in target_keep_per_view] for _ in range(num_pruning_layers)]

    dense_per_view: List[List[int]] = []
    for start_count, target_keep in zip(view_lengths, target_keep_per_view):
        if start_count <= 0:
            dense_per_view.append([0] * num_pruning_layers)
            continue
        target_keep = max(1, min(int(target_keep), int(start_count)))
        floor_count = max(1, min(int(config.min_keep_per_view), int(start_count)))
        if target_keep <= floor_count:
            dense_per_view.append([int(floor_count)] * num_pruning_layers)
            continue
        if target_keep >= start_count:
            dense_per_view.append([int(start_count)] * num_pruning_layers)
            continue
        if num_pruning_layers > 1 and float(config.warm_keep_factor) > 1.0:
            warm_keep = int(math.ceil(float(target_keep) * float(config.warm_keep_factor)))
            warm_keep = max(int(target_keep), min(int(start_count), warm_keep))
            remaining_layers = num_pruning_layers - 1
            total_target = int(target_keep * num_pruning_layers)
            remaining_total = int(total_target - warm_keep)
            min_remaining_total = int(remaining_layers * floor_count)
            max_remaining_total = int(remaining_layers * warm_keep)
            if warm_keep > target_keep and min_remaining_total <= remaining_total <= max_remaining_total:
                remaining_total_tensor = torch.tensor(float(remaining_total), device=device)
                remaining_ratios = generate_schedule_aligned(
                    pruning_loc[1:],
                    config,
                    remaining_total_tensor,
                    int(warm_keep),
                    device,
                )
                remaining_counts = simulate_dense_keep_counts(int(warm_keep), remaining_ratios, int(floor_count))
                remaining_counts = _match_total_with_monotonic_budget(
                    remaining_counts,
                    target_total=remaining_total,
                    start_count=int(warm_keep),
                    floor_count=int(floor_count),
                )
                dense_counts = [int(warm_keep)] + remaining_counts
                dense_per_view.append(
                    _repair_monotonic_counts(
                        dense_counts,
                        start_count=int(start_count),
                        floor_count=int(floor_count),
                    )
                )
                continue
        target_total = int(target_keep * num_pruning_layers)
        target_total_tensor = torch.tensor(float(target_total), device=device)
        step_ratios = generate_schedule_aligned(pruning_loc, config, target_total_tensor, int(start_count), device)
        dense_counts = simulate_dense_keep_counts(int(start_count), step_ratios, int(floor_count))
        dense_counts = _match_total_with_monotonic_budget(
            dense_counts,
            target_total=target_total,
            start_count=int(start_count),
            floor_count=int(floor_count),
        )
        dense_per_view.append(dense_counts)

    keep_schedule: List[List[int]] = []
    for layer_offset in range(num_pruning_layers):
        keep_schedule.append([int(dense_per_view[view_idx][layer_offset]) for view_idx in range(len(dense_per_view))])
    return keep_schedule


def compute_schedule_from_attn(
    config: MantisAutoPruneConfig,
    strategy: AdaptiveBudgetStrategy,
    pruning_loc: List[int],
    hidden_states: torch.Tensor,
    attn_weights: torch.Tensor,
    layout: Dict[str, Any],
    runtime_stats: Dict[str, Any],
) -> Dict[str, Any]:
    metric_t0 = time.perf_counter()
    query_positions = layout["query_positions"]
    view_positions = layout["view_positions"]

    if query_positions.numel() <= 0 or not view_positions:
        return {"keep_count_schedule": [], "active_pruning_loc": []}
    if any(view.numel() <= 0 for view in view_positions):
        return {"keep_count_schedule": [], "active_pruning_loc": []}

    attn_queries = _select_query_rows(attn_weights, query_positions)
    nmis: List[torch.Tensor] = []
    kls: List[torch.Tensor] = []

    for view_indices in view_positions:
        attn_view = attn_queries.index_select(3, view_indices)
        attn_view = torch.nan_to_num(attn_view.float(), nan=0.0, posinf=0.0, neginf=0.0)
        attn_view = attn_view / torch.clamp(attn_view.sum(dim=-1, keepdim=True), min=1e-9)
        mi_vec, kl_vec = calculate_mi_and_kl(attn_view[0])
        total_mi = _finite_scalar(mi_vec.sum() / max(1, attn_view.shape[2]), default=0.0, min_value=0.0)
        total_kl = _finite_scalar(kl_vec.sum(), default=0.0, min_value=0.0)
        nmis.append(_finite_scalar(total_mi / (total_kl + 1e-6), default=1e-6, min_value=1e-6))
        kls.append(total_kl)

    runtime_stats["metric_seconds"] = float(
        runtime_stats.get("metric_seconds", 0.0) + (time.perf_counter() - metric_t0)
    )

    rank_t0 = time.perf_counter()
    effective_rank = calculate_effective_rank(hidden_states, query_positions)
    runtime_stats["rank_seconds"] = float(
        runtime_stats.get("rank_seconds", 0.0) + (time.perf_counter() - rank_t0)
    )

    budget_t0 = time.perf_counter()
    nmi_tensor = torch.stack(nmis)
    kl_tensor = torch.stack(kls)
    effective_rank = _finite_scalar(effective_rank, default=1.0, min_value=1.0)
    invalid_metric_detected = not (
        torch.isfinite(nmi_tensor).all() and torch.isfinite(kl_tensor).all() and torch.isfinite(effective_rank).all()
    )
    nmi_tensor = torch.nan_to_num(nmi_tensor, nan=1e-6, posinf=1e6, neginf=1e-6)
    nmi_tensor = torch.clamp(nmi_tensor, min=1e-6)
    kl_tensor = torch.nan_to_num(kl_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    kl_tensor = torch.clamp(kl_tensor, min=0.0)

    complexity_signal = compute_complexity_signal(nmi_tensor, effective_rank, default=config.mu)
    complexity_score = float(
        logistic_complexity_score(
            complexity_signal,
            getattr(config, "adaptive_layer_mu", config.mu),
            getattr(config, "adaptive_layer_tau", config.tau),
        ).item()
    )

    view_lengths = [int(view.numel()) for view in view_positions]
    total_visual = int(sum(view_lengths))
    configured_visual_budget_cap = int(config.target_token_num) if int(config.target_token_num) > 0 else None
    budget_fallback_used = False
    budget_clamped_by_target = False
    budget_mode = str(getattr(config, "budget_mode", "absolute") or "absolute").lower()
    realized_coverages: List[float] = []
    absolute_anchor_mode = str(getattr(config, "absolute_anchor_mode", "none") or "none").lower()
    absolute_anchor_budget: Optional[float] = None
    absolute_controller_ratio = 1.0

    if budget_mode == "coverage":
        target_coverage = float(
            getattr(config, "coverage_min", 0.82)
            + (getattr(config, "coverage_max", 0.95) - getattr(config, "coverage_min", 0.82)) * complexity_score
        )
        target_coverage = max(0.0, min(0.999, target_coverage))
        target_keep_per_view, realized_coverages = allocate_keep_counts_by_coverage(
            attn_queries,
            view_positions,
            target_coverage=target_coverage,
            min_keep_per_view=config.min_keep_per_view,
        )
        raw_total_budget = float(sum(target_keep_per_view))
        total_keep = int(round(raw_total_budget))
        if configured_visual_budget_cap is not None:
            clamped_keep = min(total_keep, configured_visual_budget_cap)
            budget_clamped_by_target = clamped_keep != total_keep
            total_keep = clamped_keep
        total_keep = max(1, min(total_keep, total_visual))
        if sum(target_keep_per_view) != total_keep:
            temp = max(float(config.allocation_temp), 1e-4)
            alloc_signal = torch.log1p(torch.clamp(kl_tensor, min=0.0))
            alloc_weights = torch.softmax(alloc_signal / temp, dim=0)
            alloc_weights = torch.nan_to_num(alloc_weights, nan=0.0, posinf=0.0, neginf=0.0)
            if float(alloc_weights.sum().item()) <= 0.0:
                alloc_weights = torch.ones_like(alloc_weights) / max(1, alloc_weights.numel())
            target_keep_per_view = allocate_keep_counts(
                view_lengths,
                total_keep,
                min_keep_per_view=config.min_keep_per_view,
                weights=alloc_weights,
            )
            realized_coverages = []
    elif budget_mode == "elbow":
        target_keep_per_view, elbow_strengths = allocate_keep_counts_by_elbow(
            attn_queries,
            view_positions,
            min_keep_per_view=config.min_keep_per_view,
        )
        raw_total_budget = float(sum(target_keep_per_view))
        total_keep = int(round(raw_total_budget))
        if configured_visual_budget_cap is not None:
            clamped_keep = min(total_keep, configured_visual_budget_cap)
            budget_clamped_by_target = clamped_keep != total_keep
            total_keep = clamped_keep
        total_keep = max(1, min(total_keep, total_visual))
        if sum(target_keep_per_view) != total_keep:
            temp = max(float(config.allocation_temp), 1e-4)
            alloc_signal = torch.log1p(torch.clamp(kl_tensor, min=0.0))
            alloc_weights = torch.softmax(alloc_signal / temp, dim=0)
            alloc_weights = torch.nan_to_num(alloc_weights, nan=0.0, posinf=0.0, neginf=0.0)
            if float(alloc_weights.sum().item()) <= 0.0:
                alloc_weights = torch.ones_like(alloc_weights) / max(1, alloc_weights.numel())
            target_keep_per_view = allocate_keep_counts(
                view_lengths,
                total_keep,
                min_keep_per_view=config.min_keep_per_view,
                weights=alloc_weights,
            )
    else:
        elbow_strengths = []
        if absolute_anchor_mode == "elbow":
            absolute_anchor_keep_per_view, elbow_strengths = allocate_keep_counts_by_elbow(
                attn_queries,
                view_positions,
                min_keep_per_view=config.min_keep_per_view,
            )
            absolute_anchor_budget = float(sum(absolute_anchor_keep_per_view))
        raw_total_budget, auto_lower_bound, auto_upper_bound, normalized_signal, reference_signal, absolute_anchor_budget, absolute_controller_ratio = decide_absolute_budget_v2(
            config,
            complexity_signal,
            effective_rank,
            view_lengths,
            total_visual,
            anchor_budget=absolute_anchor_budget,
        )
        if not math.isfinite(raw_total_budget):
            raw_total_budget = float(min(total_visual, max(1, config.target_token_num)))
            budget_fallback_used = True
        total_keep = int(round(raw_total_budget))
        if configured_visual_budget_cap is not None:
            clamped_keep = min(total_keep, configured_visual_budget_cap)
            budget_clamped_by_target = clamped_keep != total_keep
            total_keep = clamped_keep
        total_keep = max(1, min(total_keep, total_visual))

        temp = max(float(config.allocation_temp), 1e-4)
        alloc_signal = torch.log1p(torch.clamp(kl_tensor, min=0.0))
        alloc_weights = torch.softmax(alloc_signal / temp, dim=0)
        alloc_weights = torch.nan_to_num(alloc_weights, nan=0.0, posinf=0.0, neginf=0.0)
        if float(alloc_weights.sum().item()) <= 0.0:
            alloc_weights = torch.ones_like(alloc_weights) / max(1, alloc_weights.numel())
        target_keep_per_view = allocate_keep_counts(
            view_lengths,
            total_keep,
            min_keep_per_view=config.min_keep_per_view,
            weights=alloc_weights,
        )

    full_keep_count_schedule = build_keep_count_schedule(
        config,
        pruning_loc,
        view_lengths,
        target_keep_per_view,
        hidden_states.device,
    )

    selected_pruning_mode = "progressive_full_dense"
    if config.enable_adaptive_layer_policy and not config.disable_progressive_prune and len(pruning_loc) > 1:
        active_pruning_loc, keep_count_schedule = compress_keep_count_schedule(pruning_loc, full_keep_count_schedule)
        if not active_pruning_loc:
            active_pruning_loc = [int(pruning_loc[0])]
            keep_count_schedule = [list(map(int, full_keep_count_schedule[0]))]
        if config.enable_oneshot_fallback and should_use_oneshot_fallback(config, keep_count_schedule, target_keep_per_view):
            active_pruning_loc = [int(pruning_loc[0])]
            keep_count_schedule = [[int(value) for value in target_keep_per_view]]
            selected_pruning_mode = "schedule_compressed_1layer"
        else:
            selected_pruning_mode = "schedule_compressed_multilayer"
    else:
        active_pruning_loc = [int(layer_idx) for layer_idx in pruning_loc]
        keep_count_schedule = [list(map(int, counts)) for counts in full_keep_count_schedule]
        if config.disable_progressive_prune:
            selected_pruning_mode = "oneshot_fixed"

    runtime_stats["budget_schedule_seconds"] = float(
        runtime_stats.get("budget_schedule_seconds", 0.0) + (time.perf_counter() - budget_t0)
    )
    runtime_stats["invalid_metric_detected"] = int(runtime_stats.get("invalid_metric_detected", 0) or invalid_metric_detected)
    runtime_stats["budget_fallback_used"] = int(runtime_stats.get("budget_fallback_used", 0) or budget_fallback_used)
    runtime_stats["budget_clamped_by_target"] = int(
        runtime_stats.get("budget_clamped_by_target", 0) or budget_clamped_by_target
    )
    runtime_stats["nmi_per_view"] = [float(value.item()) for value in nmi_tensor]
    runtime_stats["kl_per_view"] = [float(value.item()) for value in kl_tensor]
    runtime_stats["effective_rank"] = float(effective_rank.item())
    runtime_stats["complexity_signal"] = float(complexity_signal.item())
    runtime_stats["complexity_score"] = float(complexity_score)
    runtime_stats["configured_visual_budget_cap"] = configured_visual_budget_cap
    runtime_stats["budget_mode"] = budget_mode
    runtime_stats["raw_total_budget"] = float(raw_total_budget)
    runtime_stats["total_budget"] = int(total_keep)
    if budget_mode == "coverage":
        runtime_stats["target_coverage"] = float(target_coverage)
        runtime_stats["realized_view_coverages"] = [float(value) for value in realized_coverages]
    if budget_mode == "elbow":
        runtime_stats["elbow_strengths"] = [float(value) for value in elbow_strengths]
    if budget_mode == "absolute":
        runtime_stats["absolute_auto_bounds"] = int(bool(getattr(config, "absolute_auto_bounds", True)))
        runtime_stats["absolute_anchor_mode"] = absolute_anchor_mode
        runtime_stats["absolute_budget_scale"] = float(getattr(config, "absolute_budget_scale", 1.0))
        runtime_stats["auto_budget_lower_bound"] = int(auto_lower_bound)
        runtime_stats["auto_budget_upper_bound"] = int(auto_upper_bound)
        runtime_stats["normalized_complexity_signal"] = float(normalized_signal)
        runtime_stats["reference_complexity_signal"] = float(reference_signal)
        runtime_stats["absolute_anchor_budget"] = (
            float(absolute_anchor_budget) if absolute_anchor_budget is not None else None
        )
        runtime_stats["absolute_controller_ratio"] = float(absolute_controller_ratio)
        if absolute_anchor_mode == "elbow":
            runtime_stats["elbow_strengths"] = [float(value) for value in elbow_strengths]
    runtime_stats["adaptive_layer_policy_enabled"] = int(bool(config.enable_adaptive_layer_policy))
    runtime_stats["adaptive_layer_choices"] = [int(value) for value in getattr(config, "adaptive_layer_choices", ())]
    runtime_stats["adaptive_layer_count"] = int(len(active_pruning_loc))
    runtime_stats["dense_schedule_length"] = int(len(full_keep_count_schedule))
    runtime_stats["compressed_schedule_length"] = int(len(keep_count_schedule))
    runtime_stats["dense_pruning_layer_indices"] = [int(layer_idx) for layer_idx in pruning_loc]
    runtime_stats["dense_keep_count_schedule"] = [list(map(int, counts)) for counts in full_keep_count_schedule]
    runtime_stats["active_pruning_layer_indices"] = [int(layer_idx) for layer_idx in active_pruning_loc]
    runtime_stats["selected_pruning_mode"] = selected_pruning_mode
    runtime_stats["oneshot_fallback_used"] = int(selected_pruning_mode == "schedule_compressed_1layer")
    runtime_stats["target_keep_per_view"] = [int(value) for value in target_keep_per_view]
    runtime_stats["keep_count_schedule"] = [list(map(int, counts)) for counts in keep_count_schedule]
    return {
        "keep_count_schedule": keep_count_schedule,
        "active_pruning_loc": [int(layer_idx) for layer_idx in active_pruning_loc],
    }


def align_position_ids(
    global_indices: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
):
    if global_indices is None:
        return position_ids
    gather_idx = global_indices
    if gather_idx.shape[0] == 1 and batch_size > 1:
        gather_idx = gather_idx.expand(batch_size, -1)
    if position_ids is None:
        return gather_idx
    if position_ids.dim() == 2 and position_ids.shape[1] != seq_len:
        return position_ids.gather(1, gather_idx)
    return position_ids


def align_attention_mask(
    global_indices: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
):
    if attention_mask is None or global_indices is None:
        return attention_mask
    gather_idx = global_indices
    if gather_idx.shape[0] == 1 and batch_size > 1:
        gather_idx = gather_idx.expand(batch_size, -1)

    if attention_mask.dim() == 2 and attention_mask.shape[1] != seq_len:
        return attention_mask.gather(1, gather_idx)

    if attention_mask.dim() == 4 and (attention_mask.shape[-1] != seq_len or attention_mask.shape[-2] != seq_len):
        attn = attention_mask
        q_index = gather_idx.unsqueeze(1).unsqueeze(-1).expand(-1, attn.shape[1], gather_idx.shape[1], attn.shape[-1])
        attn = attn.gather(2, q_index)
        k_index = gather_idx.unsqueeze(1).unsqueeze(2).expand(-1, attn.shape[1], gather_idx.shape[1], gather_idx.shape[1])
        attn = attn.gather(3, k_index)
        return attn

    return attention_mask


def align_position_embeddings(
    global_indices: Optional[torch.Tensor],
    kwargs: Dict[str, Any],
    batch_size: int,
    seq_len: int,
) -> None:
    if global_indices is None:
        return
    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is None:
        return
    try:
        gather_idx = global_indices
        if gather_idx.shape[0] == 1 and batch_size > 1:
            gather_idx = gather_idx.expand(batch_size, -1)
        cos, sin = position_embeddings
        if cos.dim() == 3 and cos.shape[1] != seq_len:
            cos = cos.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, cos.size(-1)))
            sin = sin.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, sin.size(-1)))
            kwargs["position_embeddings"] = (cos, sin)
    except Exception:
        pass


def select_view_token_indices(
    attn_weights: torch.Tensor,
    query_positions: torch.Tensor,
    view_positions: List[torch.Tensor],
    target_keep_per_view: List[int],
    batch_size: int,
    device: torch.device,
) -> List[torch.Tensor]:
    attn_queries = _select_query_rows(attn_weights, query_positions)
    selected_view_positions: List[torch.Tensor] = []

    for view_idx, view_indices in enumerate(view_positions):
        if view_indices.numel() <= 0:
            selected_view_positions.append(torch.empty((batch_size, 0), dtype=torch.long, device=device))
            continue
        keep_count = int(target_keep_per_view[view_idx]) if view_idx < len(target_keep_per_view) else int(view_indices.numel())
        keep_count = max(1, min(keep_count, int(view_indices.numel())))
        score = attn_queries.index_select(3, view_indices).mean(dim=1).mean(dim=1)
        score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
        _, topk = torch.topk(score, keep_count, dim=1)
        base = view_indices.unsqueeze(0).expand(batch_size, -1)
        selected = base.gather(1, topk)
        selected_view_positions.append(torch.sort(selected, dim=1).values)

    return selected_view_positions


def build_full_indices(
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


__all__ = [
    "align_attention_mask",
    "align_position_embeddings",
    "align_position_ids",
    "build_full_indices",
    "build_keep_count_schedule",
    "calculate_effective_rank",
    "compute_schedule_from_attn",
    "select_view_token_indices",
]
