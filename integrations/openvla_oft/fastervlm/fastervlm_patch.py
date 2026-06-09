import time
import types
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class ComparePruningConfig:
    method: str = "fastervlm"
    visual_token_num: int = 128
    num_images_in_input: int = 2
    important_ratio: float = 0.0
    layer_list: Tuple[int, ...] = (8, 16, 24)
    image_token_ratio_list: Tuple[float, ...] = (0.75, 0.5, 0.25)
    fastv_pruning_layer: int = 2
    sparsevlm_pruning_start_layer: int = 1
    sparsevlm_lambda: float = 1.0
    sparsevlm_recycle_top_ratio: float = 0.5
    sparsevlm_cluster_ratio: float = 0.25
    sparsevlm_rank_tolerance: float = 1e-4
    sparsevlm_knn_k: int = 8
    min_keep_per_view: int = 1
    enable_attention_probe: bool = True
    patch_scope: str = "full"


class ComparePruningPatch:
    """FasterVLM baseline for OpenVLA-OFT.

    The paper ranks image patches with the visual encoder's [CLS]->patch attention
    and removes the lowest R percent before LLM prefill. As in the validated NORA
    integration, selection is applied to the final projected visual embeddings
    consumed by the LLM.

    OpenVLA-OFT uses two fused vision towers per image and can receive multiple
    images.  For baseline fairness we intentionally do not allocate budgets per
    view; all visual patch tokens are pooled into one sequence and a single global
    top-K is selected.
    """

    def __init__(self, model: torch.nn.Module, config: ComparePruningConfig):
        self.model = model
        self.config = config
        self.method = str(config.method or "").lower()
        if self.method not in {"fastervlm", "vispruner"}:
            raise ValueError(
                f"fastervlm_patch.py only implements FasterVLM/VisPruner, got method={config.method!r}"
            )

        self.original_process_vision_features = None
        self.original_build_multimodal_attention = None
        self.original_regression_or_discrete_prediction = None
        self.original_run_diffusion_prediction = None
        self.num_layers = self._find_num_llm_layers(model)
        self.latest_stats: Dict[str, Any] = {}
        self._last_original_visual_tokens = 0
        self._last_final_visual_tokens = 0
        self._last_reduction_ratio = 0.0
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

        print("✅ FasterVLM patch applied.")
        print(
            "   Placement: visual encoder [CLS] attention -> projector -> "
            "global top-K visual embedding pruning -> LLM"
        )
        print(f"   Final visual token budget: {int(self.config.visual_token_num)}")

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
        # OpenVLA computes NUM_PATCHES from the dense vision-backbone config.
        # FasterVLM has already reduced that sequence, so use the actual tensor
        # length or action-token extraction will start at the wrong position.
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
        # The diffusion path appends one timestep embedding inside the original
        # method, so its effective patch-prefix length is current tokens + 1.
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
        """Match AutoPrune's logging proxy: average sequence-token reduction over LLM layers."""
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
            # OpenVLA-OFT LIBERO eval does not use FiLM.  Keep a conservative
            # fallback rather than silently applying a mismatched visual path.
            projected = self.original_process_vision_features(pixel_values, language_embeddings, use_film)
            self._record_passthrough_stats(projected.shape[1], time.perf_counter() - t0)
            return projected

        score_t0 = time.perf_counter()
        patch_features, cls_scores = self._vision_features_and_cls_scores(model_self.vision_backbone, pixel_values)
        score_latency = time.perf_counter() - score_t0

        original_tokens = int(patch_features.shape[1])
        if cls_scores is None:
            raise RuntimeError("FasterVLM requires a vision tower with a CLS token for scoring")
        keep_tokens = max(1, min(int(self.config.visual_token_num), original_tokens))
        reduction_ratio = 1.0 - (float(keep_tokens) / float(original_tokens))
        if keep_tokens >= original_tokens:
            # Exact no-pruning invariant: use OpenVLA's original visual path
            # rather than reconstructing it from attention-probe intermediates.
            projected = self.original_process_vision_features(pixel_values, language_embeddings, use_film)
            self._record_passthrough_stats(original_tokens, time.perf_counter() - t0)
            return projected

        # NORA applies FasterVLM to the final visual embeddings consumed by the
        # LLM. Project every patch once, then gather the selected token rows.
        projector_t0 = time.perf_counter()
        full_projected_features = model_self.projector(patch_features)
        projector_latency = time.perf_counter() - projector_t0

        select_t0 = time.perf_counter()
        # FasterVLM Eq. (5): threshold by the computation budget R.  Passing a
        # final keep count K is equivalent to R = 1 - K/N, then retaining the
        # top K tokens with a_i >= tau.  Sorting restores original token order.
        _, topk_idx = torch.topk(cls_scores, k=keep_tokens, dim=1, largest=True, sorted=False)
        keep_idx, _ = torch.sort(topk_idx, dim=1)
        select_latency = time.perf_counter() - select_t0

        gather_t0 = time.perf_counter()
        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, full_projected_features.size(-1))
        projected = full_projected_features.gather(dim=1, index=gather_idx)
        gather_latency = time.perf_counter() - gather_t0

        total_latency = time.perf_counter() - t0
        pruning_overhead = select_latency + gather_latency
        self._last_original_visual_tokens = original_tokens
        self._last_final_visual_tokens = keep_tokens
        self._last_reduction_ratio = reduction_ratio
        if not self._reported_real_pruning:
            print(
                "✅ FasterVLM real pruning verified: "
                f"raw visual tokens {original_tokens} -> {keep_tokens} "
                f"(R={reduction_ratio:.6f}) before LLM prefill; "
                f"projector input/output length={original_tokens}, "
                f"LLM visual prefix length={int(projected.shape[1])}"
            )
            self._reported_real_pruning = True
        self.latest_stats = {
            "original_visual_tokens": original_tokens,
            "final_visual_tokens": keep_tokens,
            "fastervlm_reduction_ratio_R": reduction_ratio,
            "keep_count_schedule": [[keep_tokens]],
            "active_pruning_layers": [0],
            "view_keep_counts": [keep_tokens],
            "fastervlm_score_latency_ms": score_latency * 1000.0,
            "fastervlm_select_latency_ms": select_latency * 1000.0,
            "fastervlm_gather_latency_ms": gather_latency * 1000.0,
            "fastervlm_projector_latency_ms": projector_latency * 1000.0,
            "fastervlm_vision_score_latency_ms": score_latency * 1000.0,
            "fastervlm_total_vision_path_latency_ms": total_latency * 1000.0,
            "compare_pruning_overhead_latency_ms": pruning_overhead * 1000.0,
            # Reuse existing log columns that were originally named for FastV.
            "fastv_score_latency_ms": score_latency * 1000.0,
            "fastv_select_latency_ms": select_latency * 1000.0,
            "fastv_gather_latency_ms": gather_latency * 1000.0,
            "fastv_overhead_latency_ms": pruning_overhead * 1000.0,
            "vis_tokens": {0: keep_tokens},
        }
        return projected

    def _patched_build_multimodal_attention(self, model_self, input_embeddings, projected_patch_embeddings, attention_mask):
        t0 = time.perf_counter()
        # Because FasterVLM prunes before sequence construction, OpenVLA builds
        # the attention mask directly at the final K-token length.  There is no
        # stale full-length mask to slice.  Likewise, the LLM receives
        # position_ids=None and constructs position ids / rotary embeddings for
        # this already-compacted sequence, so position embeddings stay aligned.
        multimodal_embeddings, multimodal_attention_mask = self.original_build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        build_latency = time.perf_counter() - t0

        final_seq_len = int(multimodal_embeddings.shape[1])
        removed_visual = max(0, int(self._last_original_visual_tokens) - int(self._last_final_visual_tokens))
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
                    min(1.0, 1.0 - (float(final_seq_len) / float(original_seq_len)))
                    if original_seq_len > 0
                    else 0.0,
                ),
            }
        )
        return multimodal_embeddings, multimodal_attention_mask

    def _record_passthrough_stats(self, visual_tokens: int, latency_s: float) -> None:
        self._last_original_visual_tokens = int(visual_tokens)
        self._last_final_visual_tokens = int(visual_tokens)
        self.latest_stats = {
            "original_visual_tokens": int(visual_tokens),
            "final_visual_tokens": int(visual_tokens),
            "fastervlm_reduction_ratio_R": 0.0,
            "compare_pruning_overhead_latency_ms": latency_s * 1000.0,
            "keep_count_schedule": [[int(visual_tokens)]],
            "active_pruning_layers": [],
            "vis_tokens": {0: int(visual_tokens)},
        }

    def _vision_features_and_cls_scores(self, vision_backbone, pixel_values):
        if vision_backbone.get_num_images_in_input() == 1:
            if not vision_backbone.use_fused_vision_backbone:
                return self._single_tower_features_and_scores(vision_backbone.featurizer, pixel_values)

            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, scores = self._fused_tower_features_and_scores(
                vision_backbone.featurizer, img, vision_backbone.fused_featurizer, img_fused
            )
            return patches, scores

        images = torch.split(pixel_values, [6] * vision_backbone.get_num_images_in_input(), dim=1)
        all_patches = []
        all_scores = []
        for image in images:
            img_regular, img_fused = torch.split(image, [3, 3], dim=1)
            patches, scores = self._fused_tower_features_and_scores(
                vision_backbone.featurizer, img_regular, vision_backbone.fused_featurizer, img_fused
            )
            all_patches.append(patches)
            all_scores.append(scores)
        return torch.cat(all_patches, dim=1), torch.cat(all_scores, dim=1)

    def _fused_tower_features_and_scores(self, featurizer, img, fused_featurizer, img_fused):
        patches, scores = self._single_tower_features_and_scores(featurizer, img)
        patches_fused, scores_fused = self._single_tower_features_and_scores(fused_featurizer, img_fused)
        # OpenVLA's DINOv2 tower has [CLS] (+ register tokens), while its SigLIP
        # tower has no CLS token.  The two towers share the same patch grid, so
        # use DINOv2's paper-aligned CLS score for the fused patch representation.
        # If both towers expose CLS tokens, average their normalized attentions.
        if scores is None:
            scores = scores_fused
        elif scores_fused is not None and scores.shape == scores_fused.shape:
            scores = 0.5 * (scores + scores_fused)
        if scores is None:
            scores = torch.ones(patches.shape[:2], device=patches.device, dtype=patches.dtype)
        return torch.cat([patches, patches_fused], dim=2), scores

    def _single_tower_features_and_scores(self, featurizer, img):
        # OpenVLA projects second-to-last-layer patch features, while FasterVLM
        # ranks them with the visual encoder output layer's CLS->patch
        # attention. Keep those two layer choices separate.
        feature_idx = max(0, len(featurizer.blocks) - 2)
        score_idx = max(0, len(featurizer.blocks) - 1)
        x = self._vit_tokens_after_embed(featurizer, img)
        prefix_tokens = int(getattr(featurizer, "num_prefix_tokens", 0))

        cls_scores: Optional[torch.Tensor] = None
        patch_features: Optional[torch.Tensor] = None
        for idx, block in enumerate(featurizer.blocks):
            # FasterVLM Eq. (4): q_cls K_patch^T / sqrt(d) from the visual encoder.
            # TIMM attention applies norm1 before qkv, so use the same normalized
            # hidden states that the block's attention module consumes.
            normed = block.norm1(x)
            if idx == score_idx and prefix_tokens > 0:
                cls_scores = self._cls_to_patch_attention(block.attn, normed, prefix_tokens)
            x = block(x)
            if idx == feature_idx:
                patch_features = x[:, prefix_tokens:, :]
            if idx >= score_idx:
                break

        if patch_features is None:
            # Fallback to the already-patched OpenVLA featurizer.  This should not
            # happen for ViT towers, but keeps failures explicit and non-destructive.
            patch_features = featurizer(img)
        if cls_scores is not None:
            cls_scores = cls_scores.to(dtype=patch_features.dtype)
        return patch_features, cls_scores

    def _vit_tokens_after_embed(self, featurizer, img):
        x = featurizer.patch_embed(img)
        if hasattr(featurizer, "_pos_embed"):
            x = featurizer._pos_embed(x)
        if hasattr(featurizer, "patch_drop"):
            x = featurizer.patch_drop(x)
        if hasattr(featurizer, "norm_pre"):
            x = featurizer.norm_pre(x)
        return x

    @staticmethod
    def _cls_to_patch_attention(attn_module, normed_tokens, prefix_tokens: int):
        bsz, seq_len, _ = normed_tokens.shape
        qkv = attn_module.qkv(normed_tokens)
        qkv = qkv.reshape(bsz, seq_len, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        q = attn_module.q_norm(q) if hasattr(attn_module, "q_norm") else q
        k = attn_module.k_norm(k) if hasattr(attn_module, "k_norm") else k
        scale = getattr(attn_module, "scale", q.shape[-1] ** -0.5)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        # Average heads, then discard all prefix/class tokens from the target side.
        return attn[:, :, 0, prefix_tokens:].mean(dim=1)

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
