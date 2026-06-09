import time
import types
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class VisionZipConfig:
    method: str = "visionzip"
    visual_token_num: int = 128
    num_images_in_input: int = 2
    dominant_ratio: float = 0.7


class VisionZipPatch:
    """Paper-aligned VisionZip baseline for OpenVLA-OFT.

    VisionZip runs before LLM prefill:
      1. extract selected-layer visual tokens, attention, and key metrics;
      2. globally select K dominant patches using received attention;
      3. globally merge all remaining patches into M contextual tokens;
      4. prepend one fused global CLS token;
      5. build exactly 1 + K + M projected visual embeddings for the LLM.

    OpenVLA-OFT receives multiple images and uses fused DINOv2/SigLIP towers.
    For baseline fairness, all patch tokens from every input image are treated
    as one sequence. No per-view budget allocation is performed.
    """

    def __init__(self, model: torch.nn.Module, config: VisionZipConfig):
        self.model = model
        self.config = config
        self.method = str(config.method or "").lower()
        if self.method != "visionzip":
            raise ValueError(f"visionzip_patch.py only implements VisionZip, got method={config.method!r}")

        self.original_process_vision_features = None
        self.original_build_multimodal_attention = None
        self.original_regression_or_discrete_prediction = None
        self.original_run_diffusion_prediction = None
        self.num_layers = self._find_num_llm_layers(model)
        self.latest_stats: Dict[str, Any] = {}
        self._last_original_visual_tokens = 0
        self._last_final_visual_tokens = 0
        self._reported_real_pruning = False

    def apply(self):
        if not hasattr(self.model, "_process_vision_features"):
            raise AttributeError("OpenVLA model does not expose _process_vision_features")
        if not hasattr(self.model, "_build_multimodal_attention"):
            raise AttributeError("OpenVLA model does not expose _build_multimodal_attention")

        self.original_process_vision_features = self.model._process_vision_features
        self.original_build_multimodal_attention = self.model._build_multimodal_attention
        self.model._process_vision_features = types.MethodType(self._patched_process_vision_features, self.model)
        self.model._build_multimodal_attention = types.MethodType(self._patched_build_multimodal_attention, self.model)

        if hasattr(self.model, "_regression_or_discrete_prediction"):
            self.original_regression_or_discrete_prediction = self.model._regression_or_discrete_prediction
            self.model._regression_or_discrete_prediction = types.MethodType(
                self._patched_regression_or_discrete_prediction, self.model
            )
        if hasattr(self.model, "_run_diffusion_prediction"):
            self.original_run_diffusion_prediction = self.model._run_diffusion_prediction
            self.model._run_diffusion_prediction = types.MethodType(self._patched_run_diffusion_prediction, self.model)

        print("VisionZip patch applied.")
        print(
            "   Placement: visual encoder attention/key metrics -> projector -> "
            "global dominant/contextual zip -> LLM"
        )
        print(f"   Exact final visual token budget (1 + K + M): {int(self.config.visual_token_num)}")

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
        # VisionZip changes the real visual prefix length before LLM prefill.
        # Action extraction must therefore use the compacted tensor length.
        actual_num_patches = int(projected_patch_embeddings.shape[1])
        return self.original_regression_or_discrete_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            actual_num_patches,
            NUM_PROMPT_TOKENS,
            action_head,
        )

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
        actual_num_patches = int(projected_patch_embeddings.shape[1]) + 1
        return self.original_run_diffusion_prediction(
            input_embeddings,
            all_actions_mask,
            noise,
            action_head,
            projected_patch_embeddings,
            labels,
            attention_mask,
            actual_num_patches,
            NUM_PROMPT_TOKENS,
            noisy_action_projector,
        )

    def get_flops_reduction(self) -> float:
        layer_tokens = self.latest_stats.get("layer_tokens", {}) if self.latest_stats else {}
        original = float(self.latest_stats.get("original", 0) or 0) if self.latest_stats else 0.0
        if not layer_tokens or original <= 0 or self.num_layers <= 0:
            return float(self.latest_stats.get("flops_reduction", 0.0) or 0.0) if self.latest_stats else 0.0
        dense_total = original * self.num_layers
        actual_total = float(sum(layer_tokens.values()))
        return max(0.0, min(1.0, 1.0 - actual_total / dense_total))

    def _patched_process_vision_features(
        self,
        model_self,
        pixel_values,
        language_embeddings=None,
        use_film=False,
    ):
        t0 = time.perf_counter()
        if use_film:
            projected = self.original_process_vision_features(pixel_values, language_embeddings, use_film)
            self._record_passthrough_stats(projected.shape[1], time.perf_counter() - t0)
            return projected

        score_t0 = time.perf_counter()
        patch_features, importance, key_metric, _ = self._visionzip_inputs(
            model_self.vision_backbone, pixel_values
        )
        score_latency = time.perf_counter() - score_t0

        original_tokens = int(patch_features.shape[1])
        final_tokens = max(1, min(int(self.config.visual_token_num), original_tokens))
        if final_tokens >= original_tokens:
            projected = model_self.projector(patch_features)
            self._record_passthrough_stats(original_tokens, time.perf_counter() - t0)
            return projected
        dominant_num, contextual_num = self._allocate_budget(final_tokens, original_tokens)

        # NORA's validated integration compresses the visual embeddings that
        # are consumed by the language model. OpenVLA's projector is nonlinear,
        # so averaging raw pre-projector features can create out-of-distribution
        # projector inputs and corrupt the action policy.
        projector_t0 = time.perf_counter()
        full_projected_features = model_self.projector(patch_features)
        projector_latency = time.perf_counter() - projector_t0

        select_t0 = time.perf_counter()
        if dominant_num > 0:
            _, dominant_idx = torch.topk(importance, k=dominant_num, dim=1, largest=True, sorted=False)
            # Official VisionZip masked_select preserves source sequence order.
            dominant_idx, _ = torch.sort(dominant_idx, dim=1)
            dominant_gather = dominant_idx.unsqueeze(-1).expand(
                -1, -1, full_projected_features.size(-1)
            )
            dominant_tokens = full_projected_features.gather(1, dominant_gather)
        else:
            dominant_idx = torch.empty(
                patch_features.shape[0], 0, dtype=torch.long, device=patch_features.device
            )
            dominant_tokens = full_projected_features[:, :0, :]
        select_latency = time.perf_counter() - select_t0

        merge_t0 = time.perf_counter()
        contextual_tokens, contextual_positions, global_positions = self._contextual_merge(
            patch_features=full_projected_features,
            key_metric=key_metric,
            dominant_idx=dominant_idx,
            contextual_num=contextual_num,
        )
        # NORA's validated VisionZip adaptation preserves the representative
        # source positions of dominant/contextual/global tokens. OpenVLA is
        # spatially sensitive, so sort the compact sequence by those positions
        # instead of emitting global + dominant + contextual as three blocks.
        global_token = full_projected_features.mean(dim=1, keepdim=True)
        zipped_features = torch.cat([dominant_tokens, contextual_tokens, global_token], dim=1)
        representative_positions = torch.cat(
            [dominant_idx, contextual_positions, global_positions], dim=1
        )
        order = torch.argsort(representative_positions, dim=1)
        zipped_features = zipped_features.gather(
            1, order.unsqueeze(-1).expand(-1, -1, zipped_features.shape[-1])
        )
        if zipped_features.shape[1] != final_tokens:
            raise RuntimeError(
                f"VisionZip budget mismatch: expected {final_tokens}, got {zipped_features.shape[1]} "
                f"(K={dominant_num}, M={contextual_num})"
            )
        merge_latency = time.perf_counter() - merge_t0

        total_latency = time.perf_counter() - t0
        pruning_overhead = select_latency + merge_latency
        reduction_ratio = 1.0 - float(final_tokens) / float(original_tokens)
        self._last_original_visual_tokens = original_tokens
        self._last_final_visual_tokens = final_tokens

        if not self._reported_real_pruning:
            print(
                "VisionZip real pruning verified: "
                f"raw visual tokens {original_tokens} -> {final_tokens} "
                f"(CLS=1, K={dominant_num}, M={contextual_num}) before LLM prefill; "
                f"projector input/output length={original_tokens}, "
                f"LLM visual prefix length={int(zipped_features.shape[1])}"
            )
            self._reported_real_pruning = True

        self.latest_stats = {
            "original_visual_tokens": original_tokens,
            "final_visual_tokens": final_tokens,
            "visionzip_dominant_tokens": dominant_num,
            "visionzip_contextual_tokens": contextual_num,
            "visionzip_reduction_ratio_R": reduction_ratio,
            # Keep the shared tracker compatible with existing comparison logs.
            "fastervlm_reduction_ratio_R": reduction_ratio,
            "keep_count_schedule": [[final_tokens]],
            "active_pruning_layers": [0],
            "view_keep_counts": [final_tokens],
            "visionzip_score_latency_ms": score_latency * 1000.0,
            "visionzip_select_latency_ms": select_latency * 1000.0,
            "visionzip_merge_latency_ms": merge_latency * 1000.0,
            "visionzip_projector_latency_ms": projector_latency * 1000.0,
            "visionzip_total_vision_path_latency_ms": total_latency * 1000.0,
            "compare_pruning_overhead_latency_ms": pruning_overhead * 1000.0,
            "fastv_score_latency_ms": score_latency * 1000.0,
            "fastv_select_latency_ms": select_latency * 1000.0,
            "fastv_gather_latency_ms": merge_latency * 1000.0,
            "fastv_overhead_latency_ms": pruning_overhead * 1000.0,
            "vis_tokens": {0: final_tokens},
        }
        return zipped_features

    def _patched_build_multimodal_attention(self, model_self, input_embeddings, projected_patch_embeddings, attention_mask):
        t0 = time.perf_counter()
        # VisionZip happens before sequence construction. The original OpenVLA
        # helper therefore creates a fresh attention mask at the compacted
        # length. position_ids remain None, so the LLM builds positions/RoPE
        # directly for this real compacted sequence; no dense positions survive.
        multimodal_embeddings, multimodal_attention_mask = self.original_build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        build_latency = time.perf_counter() - t0

        final_seq_len = int(multimodal_embeddings.shape[1])
        removed_visual = max(0, self._last_original_visual_tokens - self._last_final_visual_tokens)
        original_seq_len = final_seq_len + removed_visual
        original_decoder_token_sum = original_seq_len * self.num_layers
        actual_decoder_token_sum = final_seq_len * self.num_layers
        self.latest_stats.update(
            {
                "original": original_seq_len,
                "final": final_seq_len,
                "layer_tokens": {i: final_seq_len for i in range(self.num_layers)},
                "decoder_layers_recorded": self.num_layers,
                "decoder_layers_missing": 0,
                "original_decoder_token_sum": original_decoder_token_sum,
                "actual_decoder_token_sum": actual_decoder_token_sum,
                "saved_decoder_token_sum": original_decoder_token_sum - actual_decoder_token_sum,
                "multimodal_build_latency_ms": build_latency * 1000.0,
                "flops_reduction": max(
                    0.0,
                    min(1.0, 1.0 - float(final_seq_len) / float(original_seq_len))
                    if original_seq_len > 0
                    else 0.0,
                ),
            }
        )
        return multimodal_embeddings, multimodal_attention_mask

    def _allocate_budget(self, final_tokens: int, original_tokens: int) -> Tuple[int, int]:
        if final_tokens <= 1:
            return 0, 0
        content_budget = final_tokens - 1
        if content_budget == 1:
            return 1, 0

        dominant = int(round(content_budget * float(self.config.dominant_ratio)))
        dominant = max(1, min(dominant, content_budget - 1, original_tokens))
        contextual = content_budget - dominant
        remaining = original_tokens - dominant
        if contextual > remaining:
            contextual = max(0, remaining)
            dominant = content_budget - contextual
        return dominant, contextual

    @staticmethod
    def _contextual_merge(patch_features, key_metric, dominant_idx, contextual_num: int):
        batch_size, token_count, hidden_size = patch_features.shape

        remaining_mask = torch.ones(batch_size, token_count, dtype=torch.bool, device=patch_features.device)
        remaining_mask.scatter_(1, dominant_idx, False)
        remaining_count = token_count - dominant_idx.shape[1]
        all_positions = torch.arange(token_count, device=patch_features.device).unsqueeze(0).expand(batch_size, -1)
        remaining_positions = all_positions[remaining_mask].view(batch_size, remaining_count)
        global_positions = remaining_positions[:, :1]
        if contextual_num <= 0:
            return patch_features[:, :0, :], remaining_positions[:, :0], global_positions

        remaining_features = patch_features[remaining_mask.unsqueeze(-1).expand_as(patch_features)].view(
            batch_size, remaining_count, hidden_size
        )
        metric_size = key_metric.shape[-1]
        remaining_metric = key_metric[remaining_mask.unsqueeze(-1).expand_as(key_metric)].view(
            batch_size, remaining_count, metric_size
        )
        remaining_metric = F.normalize(remaining_metric.float(), dim=-1).to(dtype=patch_features.dtype)

        # Official VisionZip uniform split:
        # step = max(1, remaining // M), then take the first M stride positions.
        step = max(1, remaining_count // contextual_num)
        target_idx = torch.arange(0, remaining_count, step, device=patch_features.device)[:contextual_num]
        if target_idx.numel() != contextual_num:
            raise RuntimeError(
                f"VisionZip contextual target mismatch: requested {contextual_num}, got {target_idx.numel()}"
            )
        all_idx = torch.arange(remaining_count, device=patch_features.device)
        merge_mask = ~torch.isin(all_idx, target_idx)

        target_metric = remaining_metric[:, target_idx, :]
        target_hidden = remaining_features[:, target_idx, :]
        target_positions = remaining_positions[:, target_idx]
        merge_metric = remaining_metric[:, merge_mask, :]
        merge_hidden = remaining_features[:, merge_mask, :]
        if merge_metric.shape[1] == 0:
            return target_hidden, target_positions, global_positions

        similarity = torch.bmm(merge_metric, target_metric.transpose(1, 2))
        assign_idx = similarity.argmax(dim=2)
        assign_one_hot = torch.zeros(
            batch_size,
            merge_metric.shape[1],
            contextual_num,
            dtype=patch_features.dtype,
            device=patch_features.device,
        )
        assign_one_hot.scatter_(2, assign_idx.unsqueeze(-1), 1)
        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), merge_hidden) / counts
        return target_hidden + aggregated_hidden, target_positions, global_positions

    def _visionzip_inputs(self, vision_backbone, pixel_values):
        if vision_backbone.get_num_images_in_input() == 1:
            if not vision_backbone.use_fused_vision_backbone:
                patches, scores, metric, cls = self._single_tower_visionzip(vision_backbone.featurizer, pixel_values)
                return patches, scores, metric, cls
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            return self._fused_tower_visionzip(
                vision_backbone.featurizer, img, vision_backbone.fused_featurizer, img_fused
            )

        images = torch.split(pixel_values, [6] * vision_backbone.get_num_images_in_input(), dim=1)
        all_patches, all_scores, all_metrics, all_cls = [], [], [], []
        for image in images:
            img_regular, img_fused = torch.split(image, [3, 3], dim=1)
            patches, scores, metric, cls = self._fused_tower_visionzip(
                vision_backbone.featurizer, img_regular, vision_backbone.fused_featurizer, img_fused
            )
            all_patches.append(patches)
            all_scores.append(scores)
            all_metrics.append(metric)
            all_cls.append(cls)
        # One global sequence and one global CLS token across every view.
        return (
            torch.cat(all_patches, dim=1),
            torch.cat(all_scores, dim=1),
            torch.cat(all_metrics, dim=1),
            torch.stack(all_cls, dim=1).mean(dim=1),
        )

    def _fused_tower_visionzip(self, featurizer, img, fused_featurizer, img_fused):
        patches, scores, metric, cls = self._single_tower_visionzip(featurizer, img)
        patches_fused, scores_fused, metric_fused, cls_fused = self._single_tower_visionzip(
            fused_featurizer, img_fused
        )
        if patches.shape[:2] != patches_fused.shape[:2]:
            raise RuntimeError("VisionZip requires fused towers to share the same patch grid")

        scores = self._combine_scores(scores, scores_fused)
        metric = torch.cat([metric, metric_fused], dim=-1)
        if cls is None:
            cls = patches.mean(dim=1, keepdim=True)
        if cls_fused is None:
            cls_fused = patches_fused.mean(dim=1, keepdim=True)
        fused_cls = torch.cat([cls, cls_fused], dim=-1)
        return torch.cat([patches, patches_fused], dim=-1), scores, metric, fused_cls

    def _single_tower_visionzip(self, featurizer, img):
        target_idx = max(0, len(featurizer.blocks) - 2)
        x = self._vit_tokens_after_embed(featurizer, img)
        prefix_tokens = int(getattr(featurizer, "num_prefix_tokens", 0))

        patch_features: Optional[torch.Tensor] = None
        importance: Optional[torch.Tensor] = None
        metric: Optional[torch.Tensor] = None
        cls_feature: Optional[torch.Tensor] = None
        for idx, block in enumerate(featurizer.blocks):
            normed = block.norm1(x)
            if idx == target_idx:
                importance, metric = self._attention_importance_and_key_metric(block.attn, normed, prefix_tokens)
            x = block(x)
            if idx == target_idx:
                patch_features = x[:, prefix_tokens:, :]
                cls_feature = x[:, :1, :] if prefix_tokens > 0 else None
                break

        if patch_features is None or importance is None or metric is None:
            raise RuntimeError("VisionZip could not extract selected-layer visual features/attention/keys")
        return (
            patch_features,
            importance.to(dtype=patch_features.dtype),
            metric.to(dtype=patch_features.dtype),
            cls_feature,
        )

    @staticmethod
    def _combine_scores(first, second):
        if first is None:
            return second
        if second is None:
            return first
        first = first / first.sum(dim=1, keepdim=True).clamp_min(1e-6)
        second = second / second.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return 0.5 * (first + second)

    @staticmethod
    def _attention_importance_and_key_metric(attn_module, normed_tokens, prefix_tokens: int):
        batch_size, seq_len, _ = normed_tokens.shape
        qkv = attn_module.qkv(normed_tokens)
        qkv = qkv.reshape(batch_size, seq_len, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        q = attn_module.q_norm(q) if hasattr(attn_module, "q_norm") else q
        k = attn_module.k_norm(k) if hasattr(attn_module, "k_norm") else k
        scale = getattr(attn_module, "scale", q.shape[-1] ** -0.5)
        attention = ((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1)

        if prefix_tokens > 0:
            # Paper Algorithm 1: CLS-to-patch attention, aggregated across heads.
            importance = attention[:, :, 0, prefix_tokens:].sum(dim=1)
        else:
            # Paper appendix: without CLS, rank by attention received from all tokens.
            importance = attention[:, :, :, prefix_tokens:].mean(dim=(1, 2))

        # Paper Algorithm 2 / official code uses selected-layer key values as
        # the semantic merge metric. Flatten heads while preserving each patch.
        patch_keys = k[:, :, prefix_tokens:, :].transpose(1, 2).reshape(batch_size, seq_len - prefix_tokens, -1)
        return importance, patch_keys

    @staticmethod
    def _vit_tokens_after_embed(featurizer, img):
        x = featurizer.patch_embed(img)
        if hasattr(featurizer, "_pos_embed"):
            x = featurizer._pos_embed(x)
        if hasattr(featurizer, "patch_drop"):
            x = featurizer.patch_drop(x)
        if hasattr(featurizer, "norm_pre"):
            x = featurizer.norm_pre(x)
        return x

    def _record_passthrough_stats(self, visual_tokens: int, latency_s: float) -> None:
        self._last_original_visual_tokens = int(visual_tokens)
        self._last_final_visual_tokens = int(visual_tokens)
        self.latest_stats = {
            "original_visual_tokens": int(visual_tokens),
            "final_visual_tokens": int(visual_tokens),
            "compare_pruning_overhead_latency_ms": latency_s * 1000.0,
            "keep_count_schedule": [[int(visual_tokens)]],
            "active_pruning_layers": [],
            "vis_tokens": {0: int(visual_tokens)},
        }

    @staticmethod
    def _find_num_llm_layers(model) -> int:
        def visit(module):
            if hasattr(module, "layers"):
                try:
                    return len(module.layers)
                except Exception:
                    pass
            for attr in ("language_model", "model", "backbone", "llm", "transformer"):
                if hasattr(module, attr):
                    found = visit(getattr(module, attr))
                    if found:
                        return found
            return 0

        return int(visit(model) or 32)
