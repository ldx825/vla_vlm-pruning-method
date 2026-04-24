"""
Adaptive Budget Strategy Module (Information-Theoretic)

当前版本保留原始 absolute 动机：
    S = L / NMI_avg

但在预算映射阶段支持两种语义：
1. 传统的 [b_min, b_max] sigmoid 映射
2. absolute + anchor + scale
   - absolute: 用复杂度信号控制保守/激进程度
   - anchor: 用样本自身的注意力分布给出预算尺度
   - scale: 跨模型迁移时的单一粗调旋钮
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().reshape(-1)[0].item())
    try:
        return float(value)
    except Exception:
        return float(default)


class AdaptiveBudgetStrategy:
    def __init__(
        self,
        # 传统 sigmoid 映射参数
        b_min: int = 96,
        b_max: int = 156,
        mu: float = 4.0,
        tau: float = 0.5,
        # 原始 AutoPrune 曲线参数；patch 侧仍会复用
        k0: float = 0.4,
        x0: float = 4.0,
        gamma: float = 0.2,
        # 新预算语义
        budget_mode: str = "absolute",
        absolute_anchor_mode: str = "none",
        absolute_budget_scale: float = 1.0,
        absolute_auto_bounds: bool = True,
        min_keep_per_view: int = 24,
        warm_keep_factor: float = 2.5,
        # 兼容旧参数接口
        **kwargs,
    ):
        self.b_max = int(b_max)
        self.b_min = int(b_min)
        self.mu = float(mu)
        self.tau = float(tau)
        self.k0 = float(k0)
        self.x0 = float(x0)
        self.gamma = float(gamma)

        self.budget_mode = str(budget_mode or "absolute").lower()
        self.absolute_anchor_mode = str(absolute_anchor_mode or "none").lower()
        self.absolute_budget_scale = float(absolute_budget_scale)
        self.absolute_auto_bounds = bool(absolute_auto_bounds)
        self.min_keep_per_view = int(min_keep_per_view)
        self.warm_keep_factor = float(warm_keep_factor)

    def decide_budget(
        self,
        metric_head: torch.Tensor,
        metric_wrist: torch.Tensor,
        effective_rank: torch.Tensor,
        *,
        visual_token_cap: Optional[int] = None,
        num_views: int = 2,
        min_keep_per_view: Optional[int] = None,
        anchor_budget: Optional[float] = None,
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        根据信息密度理论决策总预算。

        返回值保持兼容：budget 是 Python float，info 里包含预算诊断字段。
        """

        metric_head_f = max(_to_float(metric_head, 0.0), 0.0)
        metric_wrist_f = max(_to_float(metric_wrist, 0.0), 0.0)
        rank_f = max(_to_float(effective_rank, 1.0), 1.0)
        total_visual = max(1, int(visual_token_cap or self.b_max or 1))
        non_empty_views = max(1, int(num_views))
        min_keep = int(self.min_keep_per_view if min_keep_per_view is None else min_keep_per_view)

        # 1. 计算原始复杂度信号
        nmi_avg = (metric_head_f + metric_wrist_f) / 2.0 + 1e-6
        signal = rank_f / nmi_avg
        normalized_signal = math.log1p(signal / max(1.0, float(total_visual)))
        reference_signal = math.log1p(rank_f)
        delta = (normalized_signal - reference_signal) / max(self.tau, 1e-6)
        delta = max(-20.0, min(20.0, delta))
        sigmoid_val = 1.0 / (1.0 + math.exp(-delta))

        # 2. 自动上下界
        auto_floor = max(non_empty_views, min_keep * non_empty_views)
        auto_ceiling = min(
            total_visual,
            max(
                auto_floor + 8,
                int(math.ceil(rank_f * max(1.0, self.warm_keep_factor))),
            ),
        )

        if self.absolute_auto_bounds:
            lower_bound = max(1, auto_floor)
            upper_bound = max(lower_bound, total_visual)
        else:
            lower_bound = max(1, int(self.b_min))
            upper_bound = max(lower_bound, min(total_visual, int(self.b_max)))

        # 3. absolute + anchor + scale
        controller_ratio = 1.0
        anchor_budget_value = None
        if (
            self.budget_mode == "absolute"
            and self.absolute_anchor_mode == "elbow"
            and anchor_budget is not None
            and math.isfinite(float(anchor_budget))
        ):
            anchor_budget_value = float(anchor_budget)
            controller_log_ratio = max(-0.5, min(0.5, 0.5 * delta))
            controller_ratio = math.exp(controller_log_ratio)
            raw_budget = anchor_budget_value * controller_ratio * max(0.05, self.absolute_budget_scale)
        else:
            # 兼容旧版 absolute 映射
            if not self.absolute_auto_bounds:
                lower_bound = max(1, int(self.b_min))
                upper_bound = max(lower_bound, min(total_visual, int(self.b_max)))
            else:
                upper_bound = max(lower_bound, auto_ceiling)
            raw_budget = lower_bound + (upper_bound - lower_bound) * sigmoid_val

        raw_budget = float(max(lower_bound, min(upper_bound, raw_budget)))
        info = {
            "strategy": "absolute_anchor_v1" if anchor_budget_value is not None else "info_density_V2",
            "budget_mode": self.budget_mode,
            "absolute_anchor_mode": self.absolute_anchor_mode,
            "raw_total_budget": float(raw_budget),
            "budget_lower_bound": int(lower_bound),
            "budget_upper_bound": int(upper_bound),
            "complexity_signal": float(signal),
            "normalized_signal": float(normalized_signal),
            "reference_signal": float(reference_signal),
            "controller_ratio": float(controller_ratio),
            "anchor_budget": anchor_budget_value,
        }
        return float(raw_budget), info
