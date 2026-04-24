# Source Snapshot Manifest

This file records the upstream repositories imported into this private workspace.

| Component | Path | Upstream URL | Snapshot |
|---|---|---|---|
| OpenVLA-OFT | `models/openvla-oft/` | https://github.com/moojink/openvla-oft | `e4287e9` |
| NORA | `models/nora/` | https://github.com/declare-lab/nora | `6b18c23` |
| Mantis | `models/mantis/` | https://github.com/TIGER-AI-Lab/Mantis | `f3a3192` |
| FastV | `methods/fastv/` | https://github.com/pkunlp-icler/FastV | `d165972` |
| SparseVLM | `methods/sparsevlm/` | https://github.com/Gumpest/SparseVLMs | `87fe431` |
| FasterVLM | `methods/fastervlm/` | https://github.com/Theia-4869/FasterVLM | `f5d5f12` |
| VisionZip | `methods/visionzip/` | https://github.com/dvlab-research/VisionZip | `8f86b55` |
| PyramidDrop | `methods/pyramiddrop/` | https://github.com/Cooperx521/PyramidDrop | `6444f30` |

## Local Auto Prune Sources

| Target | Imported from | Destination |
|---|---|---|
| OpenVLA-OFT | `/home/liudongxu/Auto_openvla-oft/` | `methods/auto_prune/openvla_oft/` |
| NORA | `/home/liudongxu/auto_nora/` | `methods/auto_prune/nora/` |
| Mantis | `/home/liudongxu/Mantis/mantis/models/mllava/` | `methods/auto_prune/mantis/` |
| Reports | `/home/liudongxu/*AUTOPRUNE*.md`, `/home/liudongxu/Mantis/MANTIS_AUTOPRUNE_GROUP_REPORT_20260331.md` | `methods/auto_prune/reports/` |

Large local assets were not imported: checkpoints, model weights, datasets, logs, rollout videos, and Python cache files.
