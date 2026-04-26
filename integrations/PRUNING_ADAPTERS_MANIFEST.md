# Pruning Adapter Manifest

This directory archives model-specific pruning integration code copied from the working experiment trees.

## Mantis

Target:

```text
integrations/mantis/mllava_pruning_adapters/
```

Source:

```text
/mnt/jarvis/migration/home/jarvis/Mantis/Mantis/mantis/models/mllava/
```

Coverage:

- AutoPrune shared Mantis patch/helpers
- VisionZip Mantis adapter
- FasterVLM Mantis adapter
- FastV Mantis adapter
- PyramidDrop Mantis adapter
- SparseVLM Mantis adapter

## NORA

Targets:

```text
integrations/nora/libero/
integrations/nora/scripts/
```

Sources:

```text
/mnt/jarvis/migration/home/jarvis/auto_nora/experiments/libero/
/mnt/jarvis/migration/home/jarvis/auto_nora/scripts/
```

Coverage:

- `compare_pruning_patch_nora.py`: comparison pruning patch for VisionZip, FasterVLM, PyramidDrop, FastV, and SparseVLM
- `autoprune_nora.py`: AutoPrune integration
- LIBERO/NORA eval helpers and launch/debug scripts

## OpenVLA-OFT

Target:

```text
integrations/openvla_oft/libero/
```

Source:

```text
/home/liudongxu/vla_vlm-pruning-method/methods/auto_prune/openvla_oft/
```

Coverage:

- `autoprune_patch_improved_1.py`
- `fastv_patch.py`
- `sparsevlm_patch.py`
- `adaptive_budget_strategy.py`
- OpenVLA-OFT LIBERO eval entry and notes for FastV/SparseVLM
