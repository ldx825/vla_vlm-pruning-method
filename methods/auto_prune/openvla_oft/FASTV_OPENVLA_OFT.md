# FastV for OpenVLA-OFT

This directory now contains a standalone FastV baseline for LIBERO evaluation:

- `fastv_patch.py`
- `run_libero_eval.py` integration through `--use_fastv_pruning`

Example:

```bash
python methods/auto_prune/openvla_oft/run_libero_eval.py \
  --pretrained_checkpoint /path/to/openvla-oft/checkpoint \
  --task_suite_name libero_spatial \
  --use_fastv_pruning true \
  --fastv_keep_visual_tokens 128 \
  --fastv_visual_tokens 512 \
  --fastv_pruning_layer 2
```

Implementation notes:

- `fastv_pruning_layer` is FastV's `K` and is 1-based. The patch runs the first `K` layers normally, then prunes immediately after layer `K`.
- `fastv_keep_visual_tokens` is the final visual-token budget. The paper's ratio is derived internally as `R = 1 - keep / visual_tokens`.
- All visual tokens are treated as one unified pool, including tokens from multiple camera views. There is no per-view budget split.
- Visual-token importance is averaged from attention over heads and over all non-visual query tokens in the current sequence, including OpenVLA action tokens.
- The patch physically gathers the decoder hidden states and updates `position_ids` / precomputed RoPE `position_embeddings` through the same global-index map, so later layers run on the shorter sequence.
- Runtime stats use the same `latest_stats` keys consumed by the existing AutoPrune LIBERO evaluator, including `final_visual_tokens`, `flops_reduction`, `fastv_score_latency_ms`, `fastv_select_latency_ms`, and `fastv_gather_latency_ms`.
