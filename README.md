# VLA/VLM Pruning Methods

Private research repository for organizing pruning experiments on OpenVLA-OFT, NORA, and Mantis.

The repository collects three model codebases, five open-source pruning baselines, and the current Auto Prune implementations used in this workspace. Large runtime assets such as checkpoints, datasets, logs, and rollout videos are intentionally excluded from Git.

## Repository Layout

```text
models/
  openvla-oft/        # OpenVLA-OFT source snapshot
  nora/               # NORA source snapshot
  mantis/             # Mantis source snapshot

methods/
  fastv/              # FastV baseline
  sparsevlm/          # SparseVLM baseline
  fastervlm/          # FasterVLM baseline
  visionzip/          # VisionZip baseline
  pyramiddrop/        # PyramidDrop baseline
  auto_prune/         # Local Auto Prune implementations and reports

integrations/
  openvla_oft/        # Model-specific integration notes and future patches
  nora/
  mantis/

docs/
  sources.md          # Upstream repository URLs and captured commit ids
  experiment_plan.md  # Experiment organization template

scripts/              # Shared launch/evaluation scripts to be added
results/              # Lightweight result summaries only
```

## Models

| Model | Path | Purpose |
|---|---|---|
| OpenVLA-OFT | `models/openvla-oft/` | VLA policy model for LIBERO-style robotics evaluation |
| NORA | `models/nora/` | VLA/VLM model codebase for pruning integration |
| Mantis | `models/mantis/` | Multimodal model codebase for VLM pruning evaluation |

## Pruning Methods

| Method | Path | Status |
|---|---|---|
| FastV | `methods/fastv/` | Source snapshot imported |
| SparseVLM | `methods/sparsevlm/` | Source snapshot imported |
| FasterVLM | `methods/fastervlm/` | Source snapshot imported |
| VisionZip | `methods/visionzip/` | Source snapshot imported |
| PyramidDrop | `methods/pyramiddrop/` | Source snapshot imported |
| Auto Prune | `methods/auto_prune/` | Local implementations imported |

## Auto Prune Code

Current Auto Prune files are organized by target model:

```text
methods/auto_prune/openvla_oft/
methods/auto_prune/nora/
methods/auto_prune/mantis/
methods/auto_prune/reports/
```

The OpenVLA-OFT import keeps the pruning patch, adaptive budget strategy, LIBERO evaluation entrypoint, and the lightweight `AutoPrune/` source tree. The NORA and Mantis imports keep the model-specific pruning scripts and current technical notes.

## Experiment Matrix

| Model | FastV | SparseVLM | FasterVLM | VisionZip | PyramidDrop | Auto Prune |
|---|---|---|---|---|---|---|
| OpenVLA-OFT | Planned | Planned | Planned | Planned | Planned | In progress |
| NORA | Planned | Planned | Planned | Planned | Planned | In progress |
| Mantis | Planned | Planned | Planned | Planned | Planned | In progress |

## Git Notes

This is a private GitHub repository owned by `ldx825`.

Remote:

```bash
git@github.com:ldx825/vla_vlm-pruning-method.git
```

Recommended workflow:

```bash
git status
git add <files>
git commit -m "Describe the experiment change"
git push
```

Do not commit model weights, datasets, generated logs, or rollout videos. The `.gitignore` file excludes the common large runtime artifacts.
