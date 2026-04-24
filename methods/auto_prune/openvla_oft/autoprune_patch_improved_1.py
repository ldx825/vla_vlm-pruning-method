"""
改进版的AutoPrune Patch (Corrected & Audited)
基于AutoPrune论文的可靠实现，修正了旧版逻辑错误，并适配OpenVLA多视角结构。
"""

import torch
import torch.nn as nn
import math
import types
import time
import sys
import numpy as np
from torch.profiler import record_function
from typing import List, Tuple
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

def calculate_mi_and_kl(attn: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算互信息(MI)和KL散度(KL)。
    attn: [..., Num_Text, Num_Vision] (Matches any batch dim)
    
    Returns:
        MI: Tensor [...,] 互信息
        KL: Tensor [...,] KL 散度
    """
    # Determine shapes dynamically
    num_text = attn.shape[-2]
    num_vision = attn.shape[-1]
    
    # Text token 的先验概率 (假设均匀)
    p_x = 1.0 / num_text 
    
    # 联合概率 P(X, Y) = P(Y|X) * P(X)
    # attn_map 是 P(Y|X) (Vision given Text)
    p_xy = attn * p_x 
    
    # Vision token 的边缘概率 P(Y) = Sum_X P(X, Y)
    # Sum along Text dimension (-2) => [..., V]
    p_y = p_xy.sum(dim=-2) 
    
    # 1. 计算 MI
    # Ratio = (attn * p_x) / (p_x * p_y) = attn / p_y
    # Expand p_y for broadcasting: [..., 1, V]
    p_y_broadcast = p_y.unsqueeze(-2) + eps
    ratio = attn / p_y_broadcast
    
    mi_element = p_xy * torch.log2(torch.clamp(ratio, eps, None))
    # Sum over both Text and Vision dimensions
    MI = mi_element.sum(dim=(-2, -1))
    
    # 2. 计算 KL (P(Y) || Uniform)
    # KL = Sum P(y) * log( P(y) / U )
    u = 1.0 / num_vision
    kl_element = p_y * torch.log2(torch.clamp(p_y / u, eps, None))
    # Sum over Vision dimension
    KL = kl_element.sum(dim=-1)
    
    return MI, KL

def calculate_dynamic_k(global_metric: float, k0: float = 0.4, gamma: float = 0.2) -> float:
    """
    根据论文逻辑计算动态 k 值。
    Information High (Metric High) -> k Low -> Retention High
    """
    # AutoPrune 论文公式: k = k0 - gamma * I (Metric)
    # 限制 k 不小于 0.05 以防止数值问题
    return max(k0 - (global_metric * gamma), 0.05)

def generate_keep_schedule_for_view(
    num_layers: int,
    start_layer: int,
    target_count: int,
    x0: float, 
    k: float,
    view_tokens: int
) -> List[float]:
    """
    为单个视角生成每层的剪枝比例 (Ratio Schedule)。
    返回的是每一层相对于*上一层*应该保留的比例 (Keep Ratio)。
    """
    if view_tokens <= target_count:
        return [1.0] * (num_layers - start_layer)

    # 1. 计算每一层的"累积目标保留率" (Cumulative Retention)
    # 基于 Logistic 函数: y = 1 / (1 + exp(k(l - x0)))
    # 我们调整 x0 或 simple scaling 使得最后一层达到 target_count / view_tokens
    
    target_final_ratio = target_count / view_tokens
    layers_indices = list(range(start_layer, num_layers))
    
    # 简单的 Logistic 曲线
    raw_values = []
    for l in layers_indices:
        # 归一化 layer index 也许更稳定，但这里沿用论文的 x0 绝对值概念
        val = 1.0 / (1.0 + np.exp(k * (l - x0)))
        raw_values.append(val)
    
    # 2. 调整曲线使得终点对齐目标
    # 这里的 raw_values[i] 代表第 i 个剪枝层之后，剩余 Token 占原始 Token 的比例
    current_final_ratio = raw_values[-1]
    
    # 如果曲线结束得太高，我们需要向下平移/缩放
    scaled_cumulative_ratios = []
    
    if current_final_ratio < 1e-6:
        # 避免除零
        scale = 1.0
    else:
        # 简单的线性缩放对齐
        scale = target_final_ratio / current_final_ratio
        
    # 应用缩放并限制在 [target_final_ratio, 1.0] 之间
    for v in raw_values:
        scaled = v * scale
        # 保证单调递减且不低于目标
        scaled = max(scaled, target_final_ratio)
        scaled = min(scaled, 1.0)
        scaled_cumulative_ratios.append(scaled)
        
    # 3. 转换为每一步的保留率 (Step Ratio)
    # Keep_Step[i] = Cumulative[i] / Cumulative[i-1]
    step_ratios = []
    prev_cum = 1.0 # 初始状态是 100%
    
    for cum in scaled_cumulative_ratios:
        step = cum / prev_cum
        # 钳位
        step = max(0.01, min(1.0, step))
        step_ratios.append(step)
        prev_cum = cum
        
    return step_ratios

from adaptive_budget_strategy import AdaptiveBudgetStrategy


def allocate_keep_counts_by_elbow_two_views(
    attn_h: torch.Tensor,
    attn_w: torch.Tensor,
    min_keep_per_view: int,
) -> Tuple[int, List[float]]:
    keep_counts = []
    elbow_strengths = []

    for attn_view in (attn_h, attn_w):
        view_len = int(attn_view.shape[-1])
        if view_len <= 0:
            keep_counts.append(0)
            elbow_strengths.append(0.0)
            continue

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
        keep_count = max(1, knee_idx + 1)
        keep_count = max(min(view_len, int(min_keep_per_view)), keep_count)
        keep_counts.append(int(keep_count))
        elbow_strengths.append(float(gain[knee_idx].item()))

    return sum(keep_counts), elbow_strengths


def _allocate_total_keep_by_score_mass(
    total_keep: int,
    lengths: List[int],
    score_masses: List[float],
    min_keep: int,
) -> List[int]:
    non_empty = [idx for idx, length in enumerate(lengths) if int(length) > 0]
    if not non_empty:
        return [0 for _ in lengths]

    total_cap = sum(int(lengths[idx]) for idx in non_empty)
    total_keep = max(len(non_empty), min(int(total_keep), total_cap))

    floor = [0 if idx not in non_empty else min(int(lengths[idx]), int(min_keep)) for idx in range(len(lengths))]
    if sum(floor) > total_keep:
        floor = [1 if idx in non_empty else 0 for idx in range(len(lengths))]

    counts = floor[:]
    remaining = total_keep - sum(counts)
    if remaining <= 0:
        return counts

    caps = [max(0, int(length) - count) for length, count in zip(lengths, counts)]
    weighted = []
    for idx, cap in enumerate(caps):
        if cap <= 0:
            weighted.append(0.0)
            continue
        score = float(score_masses[idx]) if idx < len(score_masses) else 0.0
        weighted.append(max(score, 1e-6))

    total_weight = sum(weighted)
    if total_weight <= 0:
        total_caps = sum(caps)
        raw = [remaining * (cap / total_caps) if total_caps > 0 else 0.0 for cap in caps]
    else:
        raw = [remaining * (weight / total_weight) for weight in weighted]

    extra = [min(caps[idx], int(val)) for idx, val in enumerate(raw)]
    counts = [count + add for count, add in zip(counts, extra)]
    leftover = total_keep - sum(counts)
    if leftover > 0:
        order = sorted(range(len(lengths)), key=lambda idx: raw[idx] - extra[idx], reverse=True)
        for idx in order:
            if leftover <= 0:
                break
            if counts[idx] >= int(lengths[idx]):
                continue
            counts[idx] += 1
            leftover -= 1
    return counts


def _normalize_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _visionzip_select_indices(
    dominant_scores: torch.Tensor,
    keep_tokens: int,
    important_ratio: float,
) -> torch.Tensor:
    B, N = dominant_scores.shape
    if keep_tokens >= N:
        return torch.arange(N, device=dominant_scores.device).unsqueeze(0).expand(B, -1)

    dominant_num = max(1, int(round(keep_tokens * important_ratio)))
    dominant_num = min(dominant_num, keep_tokens)
    contextual_num = max(0, keep_tokens - dominant_num)

    top_idx = dominant_scores.topk(dominant_num, dim=1).indices
    top_idx, _ = top_idx.sort(dim=1)
    if contextual_num <= 0 or dominant_num >= N:
        return top_idx

    all_idx = torch.arange(N, device=dominant_scores.device).unsqueeze(0).expand(B, -1)
    mask = torch.ones(B, N, dtype=torch.bool, device=dominant_scores.device)
    mask.scatter_(1, top_idx, False)
    residual_idx = all_idx[mask].view(B, N - dominant_num)
    if residual_idx.shape[1] <= contextual_num:
        final_idx = torch.cat([top_idx, residual_idx], dim=1)
        final_idx, _ = final_idx.sort(dim=1)
        return final_idx

    step = max(1, residual_idx.shape[1] // contextual_num)
    target_pos = torch.arange(0, residual_idx.shape[1], step, device=dominant_scores.device)[:contextual_num]
    target_idx = residual_idx[:, target_pos]
    final_idx = torch.cat([top_idx, target_idx], dim=1)
    final_idx, _ = final_idx.sort(dim=1)
    return final_idx


def _vispruner_select_indices(
    tokens: torch.Tensor,
    attn_scores: torch.Tensor,
    keep_tokens: int,
    important_ratio: float,
) -> torch.Tensor:
    B, N, _ = tokens.shape
    if keep_tokens >= N:
        return torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)

    important_num = max(1, int(round(keep_tokens * important_ratio)))
    important_num = min(important_num, keep_tokens)
    diverse_num = max(0, keep_tokens - important_num)

    token_indices = attn_scores.argsort(dim=-1, descending=True)
    important_idx = token_indices[:, :important_num]
    residual_idx = token_indices[:, important_num:]
    if diverse_num <= 0 or residual_idx.shape[1] == 0:
        important_idx, _ = important_idx.sort(dim=1)
        return important_idx

    normalized = _normalize_tokens(tokens)
    while residual_idx.shape[1] > diverse_num:
        R = residual_idx.shape[1]
        prune_r = min(8, R - diverse_num)
        if prune_r <= 0 or R < 2:
            break
        residual_tokens = normalized.gather(1, residual_idx.unsqueeze(-1).expand(-1, -1, normalized.size(-1)))
        even = residual_tokens[:, ::2, :]
        odd = residual_tokens[:, 1::2, :]
        pairs = min(even.shape[1], odd.shape[1])
        if pairs == 0:
            break
        even = even[:, :pairs, :]
        odd = odd[:, :pairs, :]
        sim = (even * odd).sum(dim=-1)
        distinct_indices = sim.argsort(dim=-1, descending=True)[:, prune_r:]
        kept_even_idx = residual_idx[:, ::2][:, :pairs].gather(1, distinct_indices)
        residual_idx = torch.cat([kept_even_idx, residual_idx[:, 1::2]], dim=1)
        if residual_idx.shape[1] <= diverse_num:
            break
        residual_idx = residual_idx[:, : max(diverse_num, residual_idx.shape[1] - prune_r)]

    final_idx = torch.cat([important_idx, residual_idx[:, :diverse_num]], dim=1)
    final_idx, _ = final_idx.sort(dim=1)
    return final_idx

class AutoPrunePatchImproved:
    """
    改进版AutoPrune Patch (Corrected & Audited)
    """
    def __init__(
        self,
        model,
        target_token_num: int = 256, 
        visual_tokens: int = 512,   
        x0: float = 4,   #12.0        
        k0: float = 0.4,            
        gamma: float = 0.22,        
        start_layer: int = 1,       
        num_views: int = 2,  #2      
        allocation_temp: float = 0.4, # 温度系数，系数越小竞争越大越极端
        # Backward compatibility
        text_sample_ratio: float = 0.25,
        safe_ratio: float = 0.2,
        min_keep_per_view: int = 24,
        budget_mode: str = "absolute",
        absolute_anchor_mode: str = "none",
        absolute_budget_scale: float = 1.0,
        absolute_auto_bounds: bool = True,
        warm_keep_factor: float = 2.5,
        compare_method: str = "",
        compare_important_ratio: float = 0.0,
        # 新增: 注入策略对象
        budget_strategy = None,
        **kwargs,
    ):
        self.model = model
        self.target = target_token_num # 64
        self.v_num = visual_tokens # 512
        self.num_views = num_views # 2
        self.tokens_per_view = visual_tokens // num_views # 256
        
        self.start_layer = start_layer
        self.allocation_temp = allocation_temp
        self.min_keep_per_view = int(min_keep_per_view)
        self.compare_method = str(compare_method or "").lower()
        self.compare_important_ratio = float(compare_important_ratio)
        
        # --- AUTO-DETECT VISUAL TOKENS (Fix for Multi-View/OFT) ---
        # 很多时候用户传入的 visual_tokens (e.g. 512) 是错的，或者不匹配当前模型 (e.g. 576, 1152)
        # 我们尝试直接从 OpenVLA 模型中读取真实配置
        try:
            if hasattr(model, "vision_backbone"):
                # Prismatic / OpenVLA standard attributes
                vb = model.vision_backbone
                patch_count = vb.get_num_patches() if hasattr(vb, "get_num_patches") else None
                img_count = vb.get_num_images_in_input() if hasattr(vb, "get_num_images_in_input") else 1
                
                if patch_count is not None:
                    detected_v_num = patch_count * img_count
                    if detected_v_num != visual_tokens:
                        print(f"⚠️  AutoPrune Warning: User provided visual_tokens={visual_tokens}, but model suggests {detected_v_num} ({patch_count}x{img_count}).")
                        print(f"   -> Trusting USER input ({visual_tokens}) over Model config ({detected_v_num}) for Multi-View scenarios.")
                        # self.v_num = detected_v_num
                        # self.tokens_per_view = detected_v_num // num_views # Update per-view count too
        except Exception as e:
            print(f"⚠️  AutoPrune: Failed to autodect visual tokens ({e}). Using provided: {visual_tokens}")

        
        if budget_strategy is None:
            self.strategy = AdaptiveBudgetStrategy(
                b_min=max(self.min_keep_per_view * self.num_views, 1),
                b_max=self.v_num,
                mu=18.0,
                tau=5.0,
                k0=k0,
                x0=x0,
                gamma=gamma,
                budget_mode=budget_mode,
                absolute_anchor_mode=absolute_anchor_mode,
                absolute_budget_scale=absolute_budget_scale,
                absolute_auto_bounds=absolute_auto_bounds,
                min_keep_per_view=self.min_keep_per_view,
                warm_keep_factor=warm_keep_factor,
            )
        else:
            self.strategy = budget_strategy
            
        # Backward compat args storage
        self.text_sample_ratio = text_sample_ratio
        self.safe_ratio = safe_ratio
        
        # ... layers init ...
        if hasattr(model, "language_model"):
             self.llm = model.language_model
             if hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
                 self.layers = self.llm.model.layers
             elif hasattr(self.llm, "layers"):
                 self.layers = self.llm.layers
             else:
                 self.layers = []
        elif hasattr(model, "layers"):
            self.layers = model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        else:
            self.layers = [] 
            
        self.num_layers = len(self.layers)
        self.pruning_loc = list(range(start_layer, self.num_layers))
        
        self.keep_percent_schedule = [] 
        self.view_boundary_tracker = {}
        self.original_forwards = {}
        self.global_indices = None # [Global Indices Tracker based on Old Implementation]
        
        # SVD Cache Optimization
        self.cached_rank = None

        # Statistics tracking
        self.latest_stats = {"original": 0, "final": 0, "layer_tokens": {}, "vis_tokens": {}, "layer_times": {}}

        # Monitor Only Mode: If target is absurdly high, we assume monitoring.
        self.monitor_mode = (self.target > 5000)

    def _compare_enabled(self):
        return self.compare_method in {"visionzip", "fastervlm", "vispruner"}

    def _resolved_compare_important_ratio(self):
        if self.compare_important_ratio > 0:
            return float(self.compare_important_ratio)
        if self.compare_method == "visionzip":
            return 0.84
        if self.compare_method in {"fastervlm", "vispruner"}:
            return 0.5
        return 0.75


    def _record_layer_stats(self, layer_idx, start_time):
        dt = time.perf_counter() - start_time
        if not hasattr(self, "latest_stats"): return
        
        if "layer_times" not in self.latest_stats:
            self.latest_stats["layer_times"] = {}
        self.latest_stats["layer_times"][layer_idx] = dt
        
        # --- PER-LAYER DEBUG LOGGING REMOVED FOR LATENCY ---
        # Only keeping stats update for external logging if needed.


    def get_flops_reduction(self):
        """Calculate approximate FLOPs reduction based on token counts."""
        if not self.latest_stats or "layer_tokens" not in self.latest_stats or not self.latest_stats["layer_tokens"]:
            return 0.0
        
        total_original_tokens = self.latest_stats["original"] * self.num_layers
        total_actual_tokens = sum(self.latest_stats["layer_tokens"].values())
        
        # If we haven't recorded all layers yet (e.g. during first pass/generation), estimate
        # Note: During generation seq_len is 1, so this metric is mostly useful during prefill or training
        if total_original_tokens == 0:
            return 0.0
            
        return 1.0 - (total_actual_tokens / total_original_tokens)

    def reset_cache(self):
        """ Reset cached effective rank at the start of a new episode. """
        self.cached_rank = None
        # print("🔄 AutoPrune: Rank Cache Reset")

    def apply(self):
        if not self.layers:
            print("❌ Error: Could not find model layers to patch.")
            return
            
        print(f"✅ AutoPrune (Multi-View) Applied.")
        print(f"   Target: {self.target}, Views: {self.num_views}, Start Layer: {self.start_layer}")
        
        for i, layer in enumerate(self.layers):
            self.original_forwards[i] = layer.forward
            layer.forward = types.MethodType(self._make_new_forward(i), layer)

        # Patch PrismaticForConditionalGeneration method if applicable
        if hasattr(self.model, "_regression_or_discrete_prediction"):
             print("  Patching _regression_or_discrete_prediction for Prismatic Model")
             self.model._regression_or_discrete_prediction = types.MethodType(self._patched_regression_or_discrete_prediction, self.model)

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
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = model_self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
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

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        
        # PATCH: Handle pruned sequence length
        # We assume action tokens are at the end of the sequence.
        action_len = ACTION_DIM * NUM_ACTIONS_CHUNK
        actions_hidden_states = last_hidden_states[:, -action_len:, :]

        # Handle different prediction methods
        if action_head is not None:
            # L1 regression prediction
            normalized_actions = action_head.predict_action(actions_hidden_states)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # Discrete token-based prediction
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    -action_len:,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = model_self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model_self.bin_centers.shape[0] - 1)
            normalized_actions = model_self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states

    def _calculate_effective_rank(self, hidden_states, text_indices):
        """ 计算 Text Embedding 的 Nu-effective Rank (L) """
        try:
            # 1. 提取 Text Feature (Force FP32 to avoid instability)
            # shape: [N_text, D=4096]
            # 假设 Batch Size = 1 取 [0]，如果变长 Batch 需要循环处理
            # 简单起见，我们只取第一个样本算 Rank 作为参考
            text_features = hidden_states[0, text_indices, :].float()
            
            # 2. 直接计算奇异值 (SVD)
            # torch.linalg.svdvals 比完整的 svd 更快，因为它不计算 singular vectors
            # 它在数值上通常比 Gram Matrix 方法 (X*X_T) 更稳定，尤其是在 float32 下
            sigmas = torch.linalg.svdvals(text_features)
            
            # 3. 归一化为概率分布 (Probability Distribution)
            # Effective Rank 定义基于奇异值的归一化分布
            # p_k = sigma_k / sum(sigma_i)
            sigma_sum = sigmas.sum()
            if sigma_sum < 1e-9: return 1.0 # Degenerate case (全0输入)
            
            probs = sigmas / sigma_sum
            
            # 4. 计算 Shannon Entropy
            # Entropy H = - sum(p * log(p))
            # 加 epsilon 防止 log(0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12))
            
            # 5. Effective Rank = exp(Entropy)
            rank = torch.exp(entropy)
            return rank
            
        except Exception as e:
            # Fallback (兜底逻辑)
            # print(f"Rank Calc Failed: {e}")
            return torch.tensor(1.0, device=hidden_states.device)

    def _generate_schedule_aligned(self, target, x0, k):
        """ 
        生成强制对齐到 Target 的曲线 (Using GPU Tensor Parallel Search) 
        Replaces slow Python Binary Search with fast GPU tensor operations.
        """
        # target_token_num (e.g. 256) is treated as the Average Budget per layer in the pruning range.
        layers = range(self.start_layer, self.num_layers)
        num_pruning_layers = len(layers)
        
        # 1. Calculate Global Token Budget for this View across all pruning layers
        total_possible = self.tokens_per_view * num_pruning_layers
        
        # Handle scalar comparison if target is Tensor
        # We use strict check. Assuming target is on same device.
        # But if total_possible is float, we convert to Tensor for safe comparison
        if isinstance(target, torch.Tensor):
            if target >= total_possible:
                return torch.ones(num_pruning_layers, device=target.device)
        elif target >= total_possible:
            return torch.ones(num_pruning_layers, device=self.model.device)

        target_total_budget = target 
        
        # 2. Generate Raw Logistic Curve on GPU
        # Formula: 1 / (1 + exp(k * (l - x0)))
        device = self.model.device
        
        # Create layer indices tensor [start_layer, start_layer+1, ...]
        layer_indices = torch.arange(self.start_layer, self.num_layers, device=device, dtype=torch.float32)
        
        # Compute raw curve: 1 / (1 + exp(k * (l - x0)))
        # k and x0 turn into tensors for broadcasting if needed, but here represent scalar params
        raw_curve_gpu = 1.0 / (1.0 + torch.exp(k * (layer_indices - x0)))
        
        # 3. GPU Parallel Grid Search for Scale Factor
        # Instead of sequential binary search, we broadcast 2000 candidates and pick the best.
        # Range 0.0 to 100.0 is usually sufficient (scale=1.0 keeps logistic, scale>1 lifts it up)
        num_candidates = 2000
        scales = torch.linspace(0.01, 100.0, num_candidates, device=device)
        
        # Broadcast: [Num_Candidates, 1] * [1, Num_Layers] -> [Num_Candidates, Num_Layers]
        # scaled_curves[i, j] is the cumulative keep ratio for scale i at layer j
        scaled_curves = scales.unsqueeze(1) * raw_curve_gpu.unsqueeze(0)
        
        # Clamp to 1.0 (cannot keep more than 100%)
        # Note: We simulate the float behavior here. "int" truncation is ignored for speed (negligible error).
        clamped_curves = torch.clamp(scaled_curves, max=1.0)
        
        # Calculate Total Tokens for each candidate scale
        # Sum(KeepRatio * InitialTokens)
        candidate_totals = clamped_curves.sum(dim=1) * self.tokens_per_view
        
        # Find best scale (closest to target budget)
        diffs = (candidate_totals - target_total_budget).abs()
        best_idx = torch.argmin(diffs)
        # best_scale = scales[best_idx] # No .item() to avoid Sync!
        
        # 4. Generate Final Scaled Curve (on CPU for list processing, or keep on GPU?)
        # For compatibility with downstream list processing, we convert to list.
        # But we can calculate step ratios on GPU too! Let's stay on GPU for speed.
        best_curve = clamped_curves[best_idx] # [Num_Layers]
        
        # 5. Convert to Step-wise Ratios (Conditional Probability)
        # Keep_Step[i] = Cumulative[i] / Cumulative[i-1]
        
        # Shifted curve for "previous" values: [1.0, curve[0], curve[1], ...]
        # We need to prepend 1.0 (100% kept before first layer)
        prev_curve = torch.cat([torch.tensor([1.0], device=device), best_curve[:-1]])
        
        # Avoid division by zero
        # If prev < 1e-9, step = 1.0 (since we have 0 tokens, we keep 100% of 0)
        # But for stability, we just clamp denominator
        steps = best_curve / (prev_curve + 1e-9)
        
        # Clamp steps to [0.001, 1.0] for stability
        steps = torch.clamp(steps, min=0.001, max=1.0)
        
        # Return Tensor directly (avoid .tolist() sync)
        return steps

    def _visualize_mask(self, idx_h, idx_w, step_count):
        """
        Visualize the pruning mask for Head and Wrist views.
        Saves a heatmap image to experiments/visualizations/.
        """
        try:
            import matplotlib
            matplotlib.use('Agg') # Force headless backend
            import matplotlib.pyplot as plt
            import os
        except Exception as e:
            print(f"Visualization Import Error: {e}")
            return

        save_dir = "experiments/visualizations"
        os.makedirs(save_dir, exist_ok=True)
        
        # Assume square grid for visualization
        grid_dim = int(math.sqrt(self.tokens_per_view))
        
        # Prepare grids
        mask_h = np.zeros((grid_dim, grid_dim))
        mask_w = np.zeros((grid_dim, grid_dim))
        
        # Tensor to CPU numpy
        if torch.is_tensor(idx_h):
             idx_h_np = idx_h[0].detach().cpu().numpy()
        else:
             idx_h_np = idx_h
             
        if torch.is_tensor(idx_w):
             idx_w_np = idx_w[0].detach().cpu().numpy()
        else:
             idx_w_np = idx_w

        # Fill mask (1 = kept, 0 = pruned)
        for idx in idx_h_np:
            if idx < self.tokens_per_view:
                r, c = divmod(idx, grid_dim)
                if r < grid_dim and c < grid_dim:
                    mask_h[r, c] = 1
        
        for idx in idx_w_np:
            if idx < self.tokens_per_view:
                r, c = divmod(idx, grid_dim)
                if r < grid_dim and c < grid_dim:
                    mask_w[r, c] = 1
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(mask_h, cmap='binary_r', vmin=0, vmax=1)
        axes[0].set_title(f"Head View (Kept: {len(idx_h_np)})")
        axes[0].axis('off')
        
        axes[1].imshow(mask_w, cmap='binary_r', vmin=0, vmax=1)
        axes[1].set_title(f"Wrist View (Kept: {len(idx_w_np)})")
        axes[1].axis('off')
        
        plt.suptitle(f"AutoPrune Mask - Step {step_count}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mask_step_{step_count:05d}.png")
        plt.close()

    def _make_new_forward(self, layer_idx):
        def new_forward(self_layer, hidden_states, attention_mask=None, position_ids=None, **kwargs):
            t0 = time.perf_counter()
            # --- 0. Fix Input Mismatch (Due to Pruning in Previous Layers) ---
            # Model loop passes original (long) mask/pos_ids, but hidden_states might be shorter.
            B, seq_len, _ = hidden_states.shape

            # Initialize Global Indices at Layer 0 (Start of generation/forward)
            if layer_idx == 0:
                self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            
            # Correction: Some models (like HF Llama) pass None for position_ids and let layer create it.
            # But if we pruned, we MUST provide position_ids.
            if self.global_indices is None:
                 # Fallback if layer 0 was missed (unlikely) or something reset
                 self.global_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

            # --- Stats Tracking ---
            if layer_idx == 0:
                # Reset stats for new forward pass (e.g. new token generation step)
                self.latest_stats = {
                    "original": seq_len,
                    "final": seq_len,
                    "layer_tokens": {},
                    "vis_tokens": {},
                    "active_pruning_layers": [],
                    "keep_count_schedule": [],
                    "final_visual_tokens": 0,
                }
            
            # Record token count for this layer
            if hasattr(self, "latest_stats"):
                self.latest_stats["layer_tokens"][layer_idx] = seq_len
                # Update final count continuously (last layer will be the final count)
                self.latest_stats["final"] = seq_len

            # Adjust position_ids
            # [CRITICAL / OLD LOGIC]: Rely on global_indices to filter original position_ids
            # If position_ids is None, we should create it based on global_indices to ensure continuity
            if position_ids is None:
                # Create default 0..N then slice
                # Caution: Llama usually creates position_ids based on attention_mask if None.
                # But here we want to force our specific indices.
                position_ids = self.global_indices
            else:
                # If position_ids is provided, check if it's "Full" or "Already Pruned" (unlikely)
                if position_ids.shape[1] > seq_len:
                    # Input is Full, Slice it
                    position_ids = position_ids[:, self.global_indices[0]] # batch大于1也没事
                elif position_ids.shape[1] == seq_len:
                    # Input matches current length. 
                    # CHECK: Does it match our global indices?
                    # If this is Layernorm/Embedding layer output, passed pos_ids might be None or Full.
                    # If it matches seq_len, it might mean the loop passed updated ids (rare) or we didn't prune yet.
                    # To be safe, if we have global_indices tracking, we generally prefer it unless we are sure.
                    # But if lengths match, maybe we just use it.
                    pass 
                
            # Adjust attention_mask
            # [CRITICAL FIX]: If we pruned, the original mask cannot be blindly sliced.
            # Slicing [:K, :K] applies the mask of the first K tokens (Head View) to the kept tokens (Head+Wrist+Text).
            # This causes Text to inherit Head's masking properties, breaking the model.
            # Setting to None forces Llama to generate a standard Causal Mask for the new length.
            # This behavior matches the working "Old Script".
            if attention_mask is not None and attention_mask.shape[-1] != seq_len:
                attention_mask = None # attention_mask llama是会自己生成吗？

            # Adjust precomputed rotary position embeddings (Transformers>=4.44 passes these in kwargs)
            # If we prune hidden states but keep full-length cos/sin, apply_rotary_pos_emb will crash.
            if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
                try:
                    cos, sin = kwargs["position_embeddings"]
                    if hasattr(cos, "shape") and cos.shape[1] != seq_len:
                        gather_idx = self.global_indices
                        if gather_idx.shape[0] == 1 and B > 1:
                            gather_idx = gather_idx.expand(B, -1)
                        cos = cos.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, cos.size(-1)))
                        sin = sin.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, sin.size(-1)))
                        kwargs["position_embeddings"] = (cos, sin)
                except Exception:
                    # Fallback path: let model compute internal rotary embeddings from position_ids.
                    kwargs["position_embeddings"] = None


            # Only request attentions on layers that actually use them for pruning logic.
            original_output_attentions = bool(kwargs.get('output_attentions', False))
            original_use_cache = bool(kwargs.get('use_cache', False))
            need_pruning_attn = layer_idx >= self.start_layer
            requested_attentions = bool(original_output_attentions or need_pruning_attn)
            kwargs['output_attentions'] = requested_attentions
            
            # --- 1. Run Original Forward ---
            t_fwd_start = time.perf_counter()
            outputs = self.original_forwards[layer_idx](
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                **kwargs
            )
            
            # Helper to strip attentions if they weren't requested
            # output format with output_attentions=True is: (hidden_states, attentions, [past_key_values])
            def get_clean_outputs(curr_outputs, hidden_override=None):
                hidden_state = curr_outputs[0] if hidden_override is None else hidden_override
                if original_output_attentions:
                    if hidden_override is None:
                        return curr_outputs
                    return (hidden_state,) + curr_outputs[1:]
                if not requested_attentions:
                    if hidden_override is None:
                        return curr_outputs
                    return (hidden_state,) + curr_outputs[1:]
                # Strip index 1 (attentions) but preserve cache when present.
                if original_use_cache and len(curr_outputs) >= 3:
                    return (hidden_state, curr_outputs[2])
                return (hidden_state,)
            
            t_fwd_end = time.perf_counter()
            fwd_time_ms = (t_fwd_end - t_fwd_start) * 1000.0

            # 检查是否应该剪枝
            B, seq_len, _ = hidden_states.shape
            
            # 如果序列太短 (还是Prompt阶段?) 或者还没到起始层，跳过
            # CORRECTED: Was seq_len <= self.v_num + 10, which blocked pruning after first few layers reduced tokens!
            if layer_idx < self.start_layer or seq_len <= 88: 
                # +10 safety buffer for minimal text
                if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
                return get_clean_outputs(outputs)
                
            attn_weights = outputs[1] # [B, H, Seq, Seq] outputs[0].shape torch.Size([1, 605, 4096])  outputs[1].shape  torch.Size([1, 32, 605, 605])
            if attn_weights is None:
                if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
                return get_clean_outputs(outputs)
                
            # --- Layout Detection ---
            # OpenVLA结构: [Vision, Text] (假设无BOS或BOS在Vision前且被包含在slice0中)
            # 我们假设 Vision 是 Inputs 的前 current_vis_len 个 token
            
            # 确定 Vision 区间
            # OFFSET CORRECTION: OpenVLA/Llama usually starts with BOS token at index 0.
            # So Vision tokens are at [1, 1 + current_vis_len]
            
            vis_start = 1
            
            if layer_idx == self.start_layer:
                # 初始状态重置 w/ BOS check
                if seq_len > self.v_num + 1: 
                   # 假设 Text 在后
                   text_len = seq_len - (self.v_num + 1)
                   vis_indices = slice(1, 1 + self.v_num) 
                   if text_len < 1: 
                       if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
                       return get_clean_outputs(outputs)
                   text_indices = slice(1 + self.v_num, seq_len)
                else:
                    if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
                    return get_clean_outputs(outputs) # 只有 Vision + BOS? 或者长度不够，不剪枝
                    
                self.view_boundary_tracker['current_h'] = self.tokens_per_view
                self.view_boundary_tracker['current_w'] = self.tokens_per_view
            
            else:
                 # 后续层，使用追踪的长度
                 if 'current_h' not in self.view_boundary_tracker:
                      if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
                      return get_clean_outputs(outputs)
                 
                 c_h = self.view_boundary_tracker['current_h']
                 c_w = self.view_boundary_tracker['current_w']
                 current_vis_len = c_h + c_w
                 
                 # Check lengths considering BOS
                 if seq_len <= current_vis_len + 1:
                     if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
                     return get_clean_outputs(outputs)

                 vis_indices = slice(1, 1 + current_vis_len)
                 text_indices = slice(1 + current_vis_len, seq_len)

            # --- 2. Calculate Schedule (Only at Start Layer) ---
            if layer_idx == self.start_layer:
                t_sync_start = time.perf_counter()
                # Old debug sync removed; it inflates measured latency without changing semantics.
                t_l2_start = time.perf_counter()
                
                # [DEBUG] Measure GPU Async Wait Time
                # This is the "Debt" we pay for previous layers' asynchronous execution.
                # If this is high (>1ms), it means the GPU was still busy calculating previous layers/operations 
                # when we tried to synchronize here.
                # t_fwd_debt = (t_l2_start - t_sync_start) * 1000
                # print(f"[L2 SYNC DETECT] GPU Async Wait Debt: {t_fwd_debt:.2f}ms")
                
                # 提取 Attention Matrix: Text(Query)(y轴) -> Vision(Key)(x轴)
                # dims: [B, H, Num_Text, Num_Visual]
                attn_sub = attn_weights[:, :, text_indices, vis_indices]
                
                # Split View Attention
                # Vision Layout: [Head_Block, Wrist_Block]
                mid_point = self.tokens_per_view
                attn_head = attn_sub[:, :, :, :mid_point] # 好细节，头和腕的token分开剪
                attn_wrist = attn_sub[:, :, :, mid_point:]
                
                # torch.cuda.synchronize()
                # t_l2_attn = time.perf_counter() # Split Finish
                
                # 计算 MI (Vectorized across Heads)
                def get_metric_for_view(attn_fraction):
                    # attn_fraction: [B, H, T, V] torch.Size([1, 32, 92, 256])
                    
                    # NORMALIZE: Ensure rows sum to 1 (P(V|T))
                    s = attn_fraction.sum(dim=-1, keepdim=True) # torch.Size([1, 32, 92, 1])
                    attn_fraction = attn_fraction / (s + 1e-9) # torch.Size([1, 32, 92, 256])

                    # Vectorized Metric Calculation
                    # attn_fraction[0]: [H, T, V]
                    mi_vec, kl_vec = calculate_mi_and_kl(attn_fraction[0])
                    
                    total_mi = mi_vec.sum()
                    total_kl = kl_vec.sum()
                    
                    # Average over Text dim (Already done effectively in MI/KL sum?)
                    # Wait, old code divided by num_text:
                    # total_mi /= num_text
                    
                    # Let's check old logic:
                    # MI = Sum P(x,y) * log...
                    # If P(x) = 1/T, then MI is naturally scaled.
                    # BUT old code explicitly did: `total_mi /= num_text`
                    # Why? MI is usually bits. Summing MI of heads is fine.
                    # Why divide by num_text?
                    # Ah, in old code:
                    # `mi_element.sum()` gave pure MI.
                    # `total_mi /= num_text` implies we want MI per text token? No, MI is a scalar.
                    # Maybe it was just scaling?
                    # Let's preserve the exact scaling behavior.
                    
                    num_text = attn_fraction.shape[2]
                    total_mi = total_mi / num_text
                    total_kl = total_kl / num_text
                    
                    return total_mi, total_kl

                with record_function("vla_nmi_calc"):
                    mi_h, kl_h = get_metric_for_view(attn_head)
                    mi_w, kl_w = get_metric_for_view(attn_wrist)
                    
                    # torch.cuda.synchronize()
                    # t_l2_mi = time.perf_counter() # MI Finish
                    
                    # NMI = MI / (KL + epsilon)
                    nmi_h = mi_h / (kl_h + 1e-6)
                    nmi_w = mi_w / (kl_w + 1e-6)
                
                # === NEW: Calculate Effective Rank (L) ===
                # Instruction is invariant in an episode, so we cache the rank to save SVD time.
                if self.cached_rank is None:
                     # t_svd_start = time.perf_counter()
                     with record_function("vla_svd_rank"):
                         L_rank = self._calculate_effective_rank(hidden_states, text_indices)
                     self.cached_rank = L_rank
                     # print(f"[Profile] SVD Time: {(time.perf_counter() - t_svd_start)*1000:.2f}ms")
                else:
                     L_rank = self.cached_rank
                
                # torch.cuda.synchronize()
                # t_l2_svd = time.perf_counter() # SVD Finish
                
                # --- 调用策略模块决定预算 (GPU Tensor Resident) ---
                anchor_budget = None
                if getattr(self.strategy, "absolute_anchor_mode", "none") == "elbow":
                    anchor_budget, elbow_strengths = allocate_keep_counts_by_elbow_two_views(
                        attn_head,
                        attn_wrist,
                        self.min_keep_per_view,
                    )
                else:
                    elbow_strengths = []

                if self._compare_enabled():
                    score_h_mass = float(attn_head.mean(dim=(1, 2)).sum().item())
                    score_w_mass = float(attn_wrist.mean(dim=(1, 2)).sum().item())
                    keep_counts = _allocate_total_keep_by_score_mass(
                        min(int(self.target), self.v_num),
                        [self.tokens_per_view, self.tokens_per_view],
                        [score_h_mass, score_w_mass],
                        self.min_keep_per_view,
                    )
                    self.compare_keep_counts = tuple(int(x) for x in keep_counts)
                    self.keep_percent_schedule = []
                    self.latest_stats["raw_total_budget"] = float(sum(keep_counts))
                    self.latest_stats["view_keep_counts"] = [int(x) for x in keep_counts]
                    self.latest_stats["keep_count_schedule"] = [list(map(int, keep_counts))]
                    self.latest_stats["active_pruning_layers"] = [int(self.start_layer)]
                    self.latest_stats["final_visual_tokens"] = int(sum(keep_counts))
                    self.latest_stats["view_score_masses"] = [float(score_h_mass), float(score_w_mass)]
                    self.latest_stats["complexity_signal"] = 0.0
                    self.latest_stats["elbow_strengths"] = []
                else:
                    with record_function("vla_cpu_sync_alloc"):
                        b_total, info = self.strategy.decide_budget(
                            metric_head=nmi_h,
                            metric_wrist=nmi_w,
                            effective_rank=L_rank,
                            visual_token_cap=c_h + c_w if layer_idx != self.start_layer else self.v_num,
                            num_views=self.num_views,
                            min_keep_per_view=self.min_keep_per_view,
                            anchor_budget=anchor_budget,
                        )
                    # Monitor Mode Override:
                    # We calculated the budget above for logging purposes, but now we FORCE full retention.
                    if self.monitor_mode:
                        b_total = float(self.tokens_per_view * self.num_views)

                    # --- AutoPrune 内部负责分配 Total Budget (GPU Tensor Logic) ---
                    with record_function("vla_cpu_alloc_sched"):
                        tau = self.allocation_temp
                        
                        # 使用 torch 计算 softmax (GPU)
                        # Stack scalars to 1D tensor [2]
                        logits = torch.stack([kl_h, kl_w]) / tau
                        probs = torch.nn.functional.softmax(logits, dim=0)
                        
                        w_head = probs[0]
                        
                        total_budget_tensor = torch.tensor(float(b_total), device=hidden_states.device, dtype=torch.float32)
                        t_h = torch.round(total_budget_tensor * w_head)
                        t_w = total_budget_tensor - t_h
                        
                        tokens_limit = torch.tensor(self.tokens_per_view, device=hidden_states.device, dtype=torch.float32)
                        
                        surplus_h = torch.relu(t_h - tokens_limit)
                        t_h = t_h - surplus_h
                        t_w = t_w + surplus_h
                        
                        surplus_w = torch.relu(t_w - tokens_limit)
                        t_w = t_w - surplus_w
                        t_h = t_h + surplus_w
                        t_h = torch.min(t_h, tokens_limit)

                        num_sched_layers = len(self.pruning_loc)
                        x_shape = 12.0 
                        k_shape = 0.3
                        
                        s_h_ratios = self._generate_schedule_aligned(t_h * num_sched_layers, x_shape, k_shape)
                        s_w_ratios = self._generate_schedule_aligned(t_w * num_sched_layers, x_shape, k_shape)
                    
                    self.latest_stats.update(info)
                    self.latest_stats["elbow_strengths"] = elbow_strengths
                    
                    schedule_tensor = torch.stack([s_h_ratios, s_w_ratios], dim=1)
                    self.keep_percent_schedule = schedule_tensor.tolist()

            # --- 3. Execute Pruning ---

            # Current boundary
            c_h = self.view_boundary_tracker['current_h']
            c_w = self.view_boundary_tracker['current_w']
            
            # Attention Score based on Mean
            curr_vis_len = c_h + c_w
            
            # CORRECTED SLICING: 
            # Visual tokens are at indices 1 to 1+curr_vis_len (exclusive of end).
            # slice(0, curr) would include BOS (idx 0) and exclude last visual token!
            
            # [Latency Optimization] Removed debug synchronization
            # torch.cuda.synchronize()
            # t_p1 = time.perf_counter()
            
            vis_slice = slice(1, 1 + curr_vis_len) 
            text_slice = slice(1 + curr_vis_len, seq_len)
            
            # [B, H, T, V]
            rel_attn = attn_weights[:, :, text_slice, vis_slice]
            
            # Score: Mean(H) -> Mean(T) -> [B, V]
            # .mean(1) [B, T, V] -> .mean(1) [B, V]
            imp_scores = rel_attn.mean(dim=1).mean(dim=1)
            
            # Split Scores
            score_h = imp_scores[:, :c_h]
            score_w = imp_scores[:, c_h:]

            if self._compare_enabled():
                if layer_idx != self.start_layer:
                    if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
                    return get_clean_outputs(outputs)
                k_h, k_w = self.compare_keep_counts
                k_h = min(int(k_h), c_h)
                k_w = min(int(k_w), c_w)
            else:
                if not self.keep_percent_schedule or self.monitor_mode:
                    if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
                    return get_clean_outputs(outputs)

                sched_idx = layer_idx - self.start_layer
                if sched_idx < 0 or sched_idx >= len(self.keep_percent_schedule):
                    if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
                    return get_clean_outputs(outputs)

                ratios = self.keep_percent_schedule[sched_idx]
                # Determine keep count with Hard Floor protection (User Request)
                k_h = max(int(c_h * ratios[0]), self.min_keep_per_view)
                k_w = max(int(c_w * ratios[1]), self.min_keep_per_view)
            
            # [CRITICAL OPTIMIZATION]: Short-circuit if NO pruning is needed.
            # If we keep everything, we SHOULD NOT touch the tensors.
            # This ensures "Full Budget" == "Bitwise Identical to Original Model".
            if k_h == c_h and k_w == c_w:
                # Update Tracker Stats (Values unchanged)
                self.view_boundary_tracker['current_h'] = k_h
                self.view_boundary_tracker['current_w'] = k_w
                # Still need to record stats!
                if hasattr(self, "latest_stats"):
                   self.latest_stats['alloc_h'] = k_h
                   self.latest_stats['alloc_w'] = k_w
                   if "vis_tokens" not in self.latest_stats: self.latest_stats["vis_tokens"] = {}
                   self.latest_stats["vis_tokens"][layer_idx] = k_h + k_w
                   self.latest_stats["final_visual_tokens"] = int(k_h + k_w)
                
                if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
                return get_clean_outputs(outputs)

            if self._compare_enabled():
                important_ratio = self._resolved_compare_important_ratio()
                view_tokens_h = outputs[0][:, 1 : 1 + c_h, :]
                view_tokens_w = outputs[0][:, 1 + c_h : 1 + c_h + c_w, :]
                if self.compare_method == "visionzip":
                    idx_h = _visionzip_select_indices(score_h, k_h, important_ratio)
                    idx_w = _visionzip_select_indices(score_w, k_w, important_ratio)
                else:
                    idx_h = _vispruner_select_indices(view_tokens_h, score_h, k_h, important_ratio)
                    idx_w = _vispruner_select_indices(view_tokens_w, score_w, k_w, important_ratio)
            else:
                _, idx_h = torch.topk(score_h, k_h, dim=1)
                _, idx_w = torch.topk(score_w, k_w, dim=1)
                
                idx_h, _ = idx_h.sort(dim=1)
                idx_w, _ = idx_w.sort(dim=1)

            # --- Unified Gather Refactor (Robustness Fix) ---
            # Instead of manually slicing and concatenating tensors (which risks losing sync),
            # we construct a single "Full Gather Index" tensor and apply it to everything.
            
            # 1. BOS [0]
            # Device check
            dev = hidden_states.device
            idx_bos = torch.zeros((B, 1), dtype=torch.long, device=dev)
            
            # 2. Head [1 + idx_h]
            idx_h_shifted = idx_h + 1
            
            # 3. Wrist [1 + c_h + idx_w]
            idx_w_shifted = idx_w + (1 + c_h)
            
            # 4. Text [1 + c_h + c_w ... End]
            text_start_idx = 1 + c_h + c_w
            text_len = seq_len - text_start_idx
            if text_len > 0:
                # Arange (1, T) -> Expand (B, T)
                idx_text = torch.arange(text_start_idx, seq_len, device=dev).unsqueeze(0).expand(B, -1)
                full_indices = torch.cat([idx_bos, idx_h_shifted, idx_w_shifted, idx_text], dim=1)
            else:
                full_indices = torch.cat([idx_bos, idx_h_shifted, idx_w_shifted], dim=1)

            # Apple Gather
            # full_indices: (B, N_new)
            # [CRITICAL LOGIC FIX]: We must gather from outputs[0] (Layer Output), NOT hidden_states (Layer Input)!
            # Gathering from hidden_states effectively bypasses the layer calculation (MLP/Attn),
            # causing the model to lose depth and features.
            current_layer_output = outputs[0]
            expanded_indices = full_indices.unsqueeze(-1).expand(-1, -1, current_layer_output.size(-1))
            pruned_hidden = current_layer_output.gather(1, expanded_indices)

            # Apply Gather to Global Indices (for Position ID Tracking)
            # self.global_indices: (1, N_old) usually, could be (B, N_old)
            if self.global_indices is not None:
                # Ensure batch broadcast if needed
                if self.global_indices.shape[0] == 1 and B > 1:
                     current_globals = self.global_indices.expand(B, -1)
                else:
                     current_globals = self.global_indices
                
                # Gather
                self.global_indices = current_globals.gather(1, full_indices)
            
            # Update Tracker Stats
            self.view_boundary_tracker['current_h'] = k_h
            self.view_boundary_tracker['current_w'] = k_w
            
            # --- VISUALIZATION HOOK (Optional) ---
            if hasattr(self, "stats") and (self.stats.get("total_steps", 0) % 10 == 0):
                 self._visualize_mask(idx_h, idx_w, self.stats["total_steps"])
            if hasattr(self, "stats"):
                 self.stats["total_steps"] = self.stats.get("total_steps", 0) + 1
            # Update Stats for Logging (Correctly reflect current kept tokens, not just target)
            if hasattr(self, "latest_stats"):
                self.latest_stats['alloc_h'] = k_h
                self.latest_stats['alloc_w'] = k_w
                # Record Visual Tokens for this layer
                if "vis_tokens" not in self.latest_stats: self.latest_stats["vis_tokens"] = {}
                self.latest_stats["vis_tokens"][layer_idx] = k_h + k_w
                self.latest_stats["final_visual_tokens"] = int(k_h + k_w)
                if self._compare_enabled():
                    self.latest_stats["active_pruning_layers"] = [int(self.start_layer)]
                    self.latest_stats["keep_count_schedule"] = [[int(k_h), int(k_w)]]
            
            if hasattr(self, "_record_layer_stats"): self._record_layer_stats(layer_idx, t0)
            
            return get_clean_outputs(outputs, hidden_override=pruned_hidden)

        return new_forward

def compare_implementations():
    """
    比较原版和改进版的实现
    """
    print("=" * 80)
    print("AutoPrune Patch Fixed & Validated")
    print("=" * 80)
    print("Key Fixes Applied:")
    print("1. MI Check: Sum over heads, Normalize by Text Length (Following Official Impl)")
    print("2. Score Check: Mean over Text & Heads (Not Sum - preventing length bias)")
    print("3. Layout: Verified OpenVLA [Vision, Text] assumption")
    print("4. Multi-View: Separate NMI calculation for Head/Wrist")

if __name__ == "__main__":
    compare_implementations()
