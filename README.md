# VLA/VLM Pruning Methods

这是一个私有研究仓库，用来统一管理 VLA/VLM 视觉 token 剪枝方法的源码、模型适配层、LIBERO 评测入口和实验记录。当前重点是把 AutoPrune、FastV、SparseVLM、FasterVLM、VisionZip、PyramidDrop 等方法接入 OpenVLA-OFT、NORA 和 Mantis，并在多视角机器人/VLM 场景下比较 success rate、latency 和 FLOPs reduction。

## 项目架构

```text
models/
  openvla-oft/        # OpenVLA-OFT 源码快照，LIBERO VLA 评测主模型
  nora/               # NORA 源码快照
  mantis/             # Mantis 源码快照

methods/
  fastv/              # FastV 上游源码快照
  sparsevlm/          # SparseVLM 上游源码快照
  fastervlm/          # FasterVLM 上游源码快照
  visionzip/          # VisionZip 上游源码快照
  pyramiddrop/        # PyramidDrop 上游源码快照
  auto_prune/         # 早期/镜像版 AutoPrune 与 OpenVLA-OFT/NORA/Mantis 实验入口
  baseline_pruning/   # Mantis baseline pruning 的共享实现

integrations/
  openvla_oft/libero/ # OpenVLA-OFT + LIBERO 的主要接入目录
  nora/libero/        # NORA + LIBERO 接入、profile 和评测入口
  nora/scripts/       # NORA 调试与批量实验脚本
  mantis/             # Mantis mLLaVA pruning adapters

docs/
  sources.md          # 上游仓库 URL 和导入 commit 信息
  experiment_plan.md  # 实验矩阵与组织模板

paper/                # 论文 PDF 和方法参考材料
results/              # 轻量结果摘要；大日志和视频不应提交
scripts/              # 通用脚本
```

> 注意：`methods/auto_prune/openvla_oft/` 和 `integrations/openvla_oft/libero/` 当前保留了一份兼容镜像。`run_libero_eval.py` 两边一致，FastV 修正版也已同步，避免从任一路径启动时导入旧实现。

## OpenVLA-OFT / LIBERO

主要入口在 `integrations/openvla_oft/libero/`：


| 文件                            | 作用                                                                                           |
| ------------------------------- | ---------------------------------------------------------------------------------------------- |
| `run_libero_eval.py`            | OpenVLA-OFT 的 LIBERO 评测入口，收集 success rate、latency、FLOPs 统计                         |
| `autoprune_patch_improved_1.py` | 当前正确接入方式的 AutoPrune patch，FastV/SparseVLM 参考其 layer wrapper 和 mask/position 处理 |
| `adaptive_budget_strategy.py`   | AutoPrune 动态预算策略                                                                         |
| `fastv_patch_improved_1.py`     | 修正版 FastV，一次性在第 K 层后剪枝                                                            |
| `fastv_patch.py`                | 兼容入口，转发到`fastv_patch_improved_1.py`                                                    |
| `sparsevlm_patch.py`            | SparseVLM standalone baseline                                                                  |
| `FASTV_OPENVLA_OFT.md`          | FastV OpenVLA-OFT 使用说明                                                                     |
| `SPARSEVLM_OPENVLA_OFT.md`      | SparseVLM OpenVLA-OFT 使用说明                                                                 |

FastV 当前实现遵循论文流程：

1. 前 `K` 层正常 dense forward。
2. 在第 `K` 层统计视觉 token 收到的 attention score。
3. 对所有视角的视觉 token 建立一个统一池，按分数保留 `fastv_keep_visual_tokens` 个 token。
4. 后续 decoder layer 只在剪短后的序列上继续前向。

OpenVLA-OFT 的 LIBERO 推理不是自回归 KV-cache generation，因此 FastV 不处理 KV-cache，只做 hidden states、position ids、position embeddings 和 attention mask 的序列对齐。latency 和 FLOPs 字段与 AutoPrune 统计口径保持一致。

## NORA / LIBERO

主要入口在 `integrations/nora/libero/`：


| 文件                                | 作用                                                                       |
| ----------------------------------- | -------------------------------------------------------------------------- |
| `run_libero_eval_nora.py`           | NORA LIBERO 评测入口                                                       |
| `nora_utils.py`                     | NORA 推理、模型包装与 pruning hook                                         |
| `autoprune_nora.py`                 | NORA AutoPrune 接入                                                        |
| `compare_pruning_patch_nora.py`     | NORA baseline pruning：FastV、SparseVLM、FasterVLM、VisionZip、PyramidDrop |
| `profile_nora_hotspots.py`          | NORA 性能热点分析                                                          |
| `libero_utils.py`, `robot_utils.py` | LIBERO 环境辅助函数                                                        |

常用 launch/debug 脚本在 `integrations/nora/scripts/`，覆盖 FastV、SparseVLM、FasterVLM、VisionZip、PyramidDrop 和 AutoPrune 的若干 LIBERO 任务。

## Mantis / mLLaVA

Mantis 相关适配主要在两个位置：

```text
integrations/mantis/mllava_pruning_adapters/
methods/baseline_pruning/mantis/
```

`integrations/mantis/mllava_pruning_adapters/` 提供 AutoPrune、FastV、SparseVLM、FasterVLM、VisionZip、PyramidDrop 的 mLLaVA/Mantis patch 与 pruning 组件。`methods/baseline_pruning/mantis/` 提供 baseline pruning 的共享算法、公共工具和 README。

## 方法状态


| 方法        | OpenVLA-OFT/LIBERO | NORA/LIBERO              | Mantis/mLLaVA  |
| ----------- | ------------------ | ------------------------ | -------------- |
| AutoPrune   | Active             | Active                   | Active         |
| FastV       | Active             | Active via compare patch | Active adapter |
| SparseVLM   | Active             | Active via compare patch | Active adapter |
| FasterVLM   | Compare baseline   | Active via compare patch | Active adapter |
| VisionZip   | Compare baseline   | Active via compare patch | Active adapter |
| PyramidDrop | Compare baseline   | Active via compare patch | Active adapter |
