# Mantis mllava Baseline Pruning

This package adds Mantis/mllava-compatible pruning baselines:

- `fastervlm`: uses vision-encoder `[CLS] -> patch` attention and keeps the top `target_token_num` visual tokens before LLM prefill.
- `visionzip`: keeps dominant tokens by `[CLS]` attention and turns the rest into contextual tokens with a 70/30 dominant/context split.
- `pyramiddrop`: prunes during LLM prefill at stage boundaries; it solves for `lambda` so the average remaining visual tokens across pruning stages matches `target_token_num`.
- `fastv`: runs the first `K` decoder layers densely, scores every visual token from attention paid by all other current tokens, then really shortens the prefill sequence to `target_token_num` visual tokens.
- `sparsevlm`: progressively selects image-related text raters, scores visual tokens layer by layer, adaptively deletes redundant visual tokens, recycles part of the deleted set into merged contextual tokens, and projects the final budget toward `target_token_num`.

All methods treat all visual tokens as one unified pool. They intentionally do not use AutoPrune's per-view allocation policy, which keeps the BLINK multi-view baseline comparison clean.

```python
from methods.baseline_pruning.mantis import (
    MantisBaselinePruneConfig,
    maybe_apply_mantis_baseline_pruning,
)

patch = maybe_apply_mantis_baseline_pruning(
    model,
    MantisBaselinePruneConfig(
        enabled=True,
        method="fastv",  # "fastervlm", "visionzip", "pyramiddrop", or "sparsevlm"
        target_token_num=128,
        fastv_pruning_layer=2,
    ),
)
```

Environment builder:

```python
from methods.baseline_pruning.mantis import build_mantis_baseline_config_from_env
patch = maybe_apply_mantis_baseline_pruning(model, build_mantis_baseline_config_from_env())
```

Useful environment variables:

- `MANTIS_BASELINE_PRUNE_ENABLED=1`
- `MANTIS_BASELINE_PRUNE_METHOD=fastervlm|visionzip|pyramiddrop|fastv|sparsevlm`
- `MANTIS_BASELINE_PRUNE_TARGET_TOKEN_NUM=128`
- `MANTIS_BASELINE_PRUNE_FASTV_PRUNING_LAYER=2`
- `MANTIS_BASELINE_PRUNE_PYRAMIDDROP_START_LAYER=2`
- `MANTIS_BASELINE_PRUNE_PYRAMIDDROP_MAX_PRUNING_LAYERS=6`
- `MANTIS_BASELINE_PRUNE_SPARSEVLM_START_LAYER=2`
- `MANTIS_BASELINE_PRUNE_SPARSEVLM_MAX_PRUNING_LAYERS=6`
- `MANTIS_BASELINE_PRUNE_SPARSEVLM_RECYCLE_TOP_RATIO=0.5`
- `MANTIS_BASELINE_PRUNE_SPARSEVLM_CLUSTER_RATIO=0.25`

The patch is also exposed as `model._autoprune_patch`, so existing evaluation code that reads AutoPrune-style stats can collect:

- `last_tokens_before`
- `last_tokens_after`
- `method_overhead_seconds`
- `flops_reduction`
- `prefill_seq_len`
- `pruned_layers`
