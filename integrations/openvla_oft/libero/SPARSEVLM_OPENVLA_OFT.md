# SparseVLM for OpenVLA-OFT

This directory now contains a standalone SparseVLM baseline for LIBERO evaluation:

- `sparsevlm_patch.py`
- `run_libero_eval.py` integration through `--use_sparsevlm_pruning`

Example:

```bash
python methods/auto_prune/openvla_oft/run_libero_eval.py \
  --pretrained_checkpoint /path/to/openvla-oft/checkpoint \
  --task_suite_name libero_spatial \
  --use_sparsevlm_pruning true \
  --sparsevlm_target_visual_tokens 128 \
  --sparsevlm_visual_tokens 512 \
  --sparsevlm_pruning_start_layer 1
```

Implementation notes:

- All visual tokens are treated as one unified pool: `[BOS] + all visual tokens from all views + text/action tokens`.
- Text raters are selected from non-visual tokens with the SparseVLM released-code rule: visual-to-text affinity, softmax over text, average over visual tokens, then keep text tokens above the mean.
- Each decoder layer after `sparsevlm_pruning_start_layer` can sparsify the current visual pool.
- Visual-token scores are the selected raters' attention to visual tokens, averaged over heads and raters.
- The adaptive deletion count uses visual effective rank and redundancy, then applies a budget projection so the sequence can reach `sparsevlm_target_visual_tokens`.
- Deleted tokens are not simply discarded: a configurable subset is rescued, clustered with density-peak clustering, merged, and inserted back into the visual segment.
- The patch physically shortens `hidden_states` and updates the global index map used for `position_ids` and precomputed RoPE `position_embeddings`.
- Action tokens are extracted relative to the end of the OpenVLA sequence after pruning, avoiding stale `NUM_PATCHES` offsets.
- Runtime stats use the same `latest_stats` keys consumed by the existing AutoPrune LIBERO evaluator, including `final_visual_tokens`, `flops_reduction`, `sparsevlm_*_latency_ms`, and token-sum fields.
