# Mantis mllava Baseline Pruning

This package adds three Mantis/mllava-compatible pruning baselines:

- `fastervlm`: uses vision-encoder `[CLS] -> patch` attention and keeps the top `target_token_num` visual tokens before LLM prefill.
- `visionzip`: keeps dominant tokens by `[CLS]` attention and turns the rest into contextual tokens with a 70/30 dominant/context split.
- `pyramiddrop`: prunes during LLM prefill at stage boundaries; it solves for `lambda` so the average remaining visual tokens across pruning stages matches `target_token_num`.

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
        method="fastervlm",  # "visionzip" or "pyramiddrop"
        target_token_num=128,
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
- `MANTIS_BASELINE_PRUNE_METHOD=fastervlm|visionzip|pyramiddrop`
- `MANTIS_BASELINE_PRUNE_TARGET_TOKEN_NUM=128`
- `MANTIS_BASELINE_PRUNE_PYRAMIDDROP_START_LAYER=2`
- `MANTIS_BASELINE_PRUNE_PYRAMIDDROP_MAX_PRUNING_LAYERS=6`

The patch is also exposed as `model._autoprune_patch`, so existing evaluation code that reads AutoPrune-style stats can collect:

- `last_tokens_before`
- `last_tokens_after`
- `method_overhead_seconds`
- `flops_reduction`
- `prefill_seq_len`
- `pruned_layers`
