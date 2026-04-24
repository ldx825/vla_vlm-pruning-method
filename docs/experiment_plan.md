# Experiment Plan

Use this document to track pruning integration and evaluation progress.

## Integration Table

| Model | Method | Code path | Status | Notes |
|---|---|---|---|---|
| OpenVLA-OFT | FastV | `integrations/openvla_oft/fastv/` | Todo |  |
| OpenVLA-OFT | SparseVLM | `integrations/openvla_oft/sparsevlm/` | Todo |  |
| OpenVLA-OFT | FasterVLM | `integrations/openvla_oft/fastervlm/` | Todo |  |
| OpenVLA-OFT | VisionZip | `integrations/openvla_oft/visionzip/` | Todo |  |
| OpenVLA-OFT | PyramidDrop | `integrations/openvla_oft/pyramiddrop/` | Todo |  |
| OpenVLA-OFT | Auto Prune | `methods/auto_prune/openvla_oft/` | In progress | Imported from local workspace |
| NORA | Auto Prune | `methods/auto_prune/nora/` | In progress | Imported from local workspace |
| Mantis | Auto Prune | `methods/auto_prune/mantis/` | In progress | Imported from local workspace |

## Result Tracking

Recommended result fields:

| Date | Model | Method | Dataset/Benchmark | Budget | Accuracy/Success | Latency | Commit | Notes |
|---|---|---|---|---|---|---|---|---|

Keep raw logs and videos outside Git. Store only summarized tables, analysis notes, and small reproducibility metadata in this repository.
