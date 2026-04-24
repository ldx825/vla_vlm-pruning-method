# NORA Paper Baseline Pruning

This adds NORA/Qwen2.5-VL versions of:

- `fastervlm`
- `visionzip`
- `pyramiddrop`
- `fastv`
- `sparsevlm`

All methods treat image tokens as one unified visual token pool. They do not use AutoPrune's per-view allocation.

Example:

```bash
python methods/auto_prune/nora/experiments/libero/run_libero_eval_nora.py \
  --task_suite_name libero_spatial \
  --enable_baseline_pruning true \
  --baseline_pruning_method fastervlm \
  --baseline_target_token_num 128
```

Methods:

- FasterVLM compresses visual tokens before LLM prefill. The requested final keep count is converted internally to the paper's reduction ratio `R = 1 - keep / total`.
- VisionZip compresses before LLM prefill. `K` and `M` are split as 70% dominant tokens and 30% contextual tokens.
- PyramidDrop prunes during prefill at decoder stage boundaries. It solves `lambda` so the average remaining visual tokens across pruning stages matches `baseline_target_token_num`.
- FastV runs the first `baseline_start_layer` layers normally, then prunes after layer K. The score is attention from all non-visual tokens to visual tokens, including NORA action tokens.
- SparseVLM starts at `baseline_start_layer` and prunes progressively in decoder layers. Text raters are selected only from non-action instruction/text tokens, then visual tokens are scored, adaptively deleted with rank/redundancy, partially rescued, clustered, and merged.

FastV example:

```bash
python methods/auto_prune/nora/experiments/libero/run_libero_eval_nora.py \
  --task_suite_name libero_spatial \
  --enable_baseline_pruning true \
  --baseline_pruning_method fastv \
  --baseline_target_token_num 128 \
  --baseline_start_layer 2
```

SparseVLM example:

```bash
python methods/auto_prune/nora/experiments/libero/run_libero_eval_nora.py \
  --task_suite_name libero_spatial \
  --enable_baseline_pruning true \
  --baseline_pruning_method sparsevlm \
  --baseline_target_token_num 128 \
  --baseline_start_layer 2 \
  --baseline_sparsevlm_lambda 1.0
```

Stats are exposed through `model.last_inference_profile` using the same keys consumed by the existing LIBERO eval:

- `last_tokens_before`
- `last_tokens_after`
- `topk_seconds`
- `gather_seconds`
- `flops_reduction`
- `final_visual_tokens`
