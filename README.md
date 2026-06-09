# VLA/VLM Pruning Methods

这是一个私有研究仓库，用来统一管理 VLA/VLM 视觉 token 剪枝方法的源码、模型适配层、LIBERO 评测入口和实验记录。当前重点是把 AutoPrune、FastV、SparseVLM、FasterVLM、VisionZip、PyramidDrop、VLA-Pruner、VLA-ADP 等方法接入 OpenVLA-OFT、NORA 和 Mantis，并在多视角机器人/VLM 场景下比较 success rate、latency 和 FLOPs reduction。

## 当前重点

- OpenVLA-OFT + LIBERO：评测 LIBERO-Spatial/Object/Goal/90/Long 上的 successful-rate、latency(ms/step)、FLOPs 缩减率。
- NORA + LIBERO：维护 dual-view NORA-LONG/NORA LIBERO 的 AutoPrune、baseline pruning、VLA-Pruner、VLA-ADP 接入。
- AutoPrune：多视角 VLA 版本，带动态预算、分视角分配和 AutoPrune 风格统计接口。
- FastV：OpenVLA-OFT 修正版使用统一视觉 token 池，不按视角分组；参数使用 `keep_visual_tokens` 替代论文里的剪枝比例 `R`。
- VLA-Pruner：已新增 OpenVLA-OFT 和 NORA 适配，结合语义 attention score 与跨步 action-attention temporal score。
- VLA-ADP：已新增 OpenVLA-OFT 和 NORA 适配，使用 pre-LLM Q/K 得分、按视角预算分配，并可基于轨迹变化动态决定是否剪枝。
- SparseVLM 和其他 baseline：保持与 AutoPrune 相同的统计字段，便于公平比较。

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
  openvla_oft/scripts/# OpenVLA-OFT 单实验、debug 和表格队列脚本
  openvla_oft/fastervlm/     # FasterVLM 的独立 OpenVLA-OFT patch 与单实验脚本
  openvla_oft/visionzip/     # VisionZip 的独立 OpenVLA-OFT patch 与单实验脚本
  openvla_oft/pyramiddrop/   # PyramidDrop 的独立 OpenVLA-OFT patch 与单实验脚本
  openvla_oft/baseline_libero/ # 三个 baseline 共用、已接好统计接口的 LIBERO 评测入口
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
| `vlapruner_oft_patch.py`        | VLA-Pruner 的 OpenVLA-OFT patch，统一视觉池 + temporal action attention                         |
| `vlaadp_oft_patch.py`           | VLA-ADP 的 OpenVLA-OFT patch，pre-LLM Q/K 评分 + 轨迹门控 + 按视角预算                           |
| `FASTV_OPENVLA_OFT.md`          | FastV OpenVLA-OFT 使用说明                                                                     |
| `SPARSEVLM_OPENVLA_OFT.md`      | SparseVLM OpenVLA-OFT 使用说明                                                                 |

FastV 当前实现遵循论文流程：

1. 前 `K` 层正常 dense forward。
2. 在第 `K` 层统计视觉 token 收到的 attention score。
3. 对所有视角的视觉 token 建立一个统一池，按分数保留 `fastv_keep_visual_tokens` 个 token。
4. 后续 decoder layer 只在剪短后的序列上继续前向。

OpenVLA-OFT 的 LIBERO 推理不是自回归 KV-cache generation，因此 FastV 不处理 KV-cache，只做 hidden states、position ids、position embeddings 和 attention mask 的序列对齐。latency 和 FLOPs 字段与 AutoPrune 统计口径保持一致。

### FasterVLM / VisionZip / PyramidDrop

三个 baseline 的稳定 OpenVLA-OFT 适配按方法独立归档：

| 目录 | 核心实现 | 剪枝位置与预算语义 |
| --- | --- | --- |
| `integrations/openvla_oft/fastervlm/` | `fastervlm_patch.py` | 视觉编码器最终层 CLS-to-patch attention 全局排序；在 LLM prefill 前真实保留指定视觉 token 数 |
| `integrations/openvla_oft/visionzip/` | `visionzip_patch.py` | 在 projected visual embeddings 上构造 dominant/contextual/global tokens；最终 token 数按约 70%/30% 分配 |
| `integrations/openvla_oft/pyramiddrop/` | `pyramiddrop_patch.py` | 在 decoder stage 末逐步剪枝；传入参数表示所有 decoder 层的平均视觉 token 数，并自动求解 λ |
| `integrations/openvla_oft/baseline_libero/` | `run_libero_eval.py` | 三个 baseline 共用的 LIBERO 入口，统一记录 success rate、latency 和 FLOPs reduction |

这些 baseline 遵循相同实验约定：

- 所有相机视角组成一个统一视觉 token 池，不进行按视角预算分配。
- 执行真实序列剪枝，后续 LLM 层实际接收更短的 hidden states，而不是仅使用 attention mask 做假剪枝。
- 保留 token 原始顺序，并同步处理 attention mask、position IDs、position embeddings 和 OpenVLA-OFT 动作 token 偏移。
- proprio token、diffusion timestep token、文本 token 与动作 token 不会被当作视觉 token 删除。
- PyramidDrop 只使用动作占位符之前的最后一个指令 token 作为 query，不会误用动作 token。
- FLOPs reduction 使用与 AutoPrune 一致的逐层实际 token 数代理统计。

对应单实验脚本分别位于每个方法目录：

脚本从当前仓库加载 patch 和 `baseline_libero/run_libero_eval.py`，但默认复用
`/home/liudongxu/Auto_openvla-oft` 中已有的权重、LIBERO 环境与日志目录；可通过
`OPENVLA_OFT_RUNTIME_ROOT` 指向其他现有运行资源目录。

```bash
# FasterVLM：KEEP_VISUAL_TOKENS 是最终保留视觉 token 数
cd integrations/openvla_oft/fastervlm
GPU=0 TASK_SUITE_NAME=libero_goal TASK_ID=4 KEEP_VISUAL_TOKENS=448 NUM_TRIALS=20 \
  bash run_fastervlm_oft_single.sh

# VisionZip：KEEP_VISUAL_TOKENS 是 K + M + 1 的最终视觉 token 数
cd ../visionzip
GPU=0 TASK_SUITE_NAME=libero_goal TASK_ID=4 KEEP_VISUAL_TOKENS=448 NUM_TRIALS=20 \
  bash run_visionzip_oft_single.sh

# PyramidDrop：AVERAGE_VISUAL_TOKENS 是 decoder 各层平均视觉 token 数
cd ../pyramiddrop
GPU=0 TASK_SUITE_NAME=libero_goal TASK_ID=4 AVERAGE_VISUAL_TOKENS=448 NUM_TRIALS=20 \
  bash run_pyramiddrop_oft_single.sh
```

OpenVLA-OFT 新增脚本在 `integrations/openvla_oft/scripts/`：

| 脚本 | 作用 |
| --- | --- |
| `run_vlapruner_oft_single.sh` | 单个 LIBERO suite/task/keep 设置的 VLA-Pruner-OFT 实验 |
| `run_vlaadp_oft_single.sh` | 单个 LIBERO suite/task/keep 设置的 VLA-ADP-OFT 实验 |
| `run_table_vlapruner_oft_queue.sh` | VLA-Pruner-OFT 表格实验队列 |
| `run_table_vlaadp_oft_queue.sh` | VLA-ADP-OFT 表格实验队列 |
| `debug_vlapruner_oft_spatial_task0_gpu7.sh` | VLA-Pruner-OFT debugpy 调试脚本 |
| `debug_vlaadp_oft_long_task0.sh` | VLA-ADP-OFT debugpy 调试脚本 |

常用参数示例：

```bash
# VLA-Pruner-OFT
--use_vlapruner_oft True
--vlapruner_keep_visual_tokens 224
--vlapruner_pruning_layer 3
--vlapruner_temporal_window 3
--vlapruner_temporal_gamma 0.8
--vlapruner_warmup_steps 3

# VLA-ADP-OFT
--use_vlaadp_oft True
--vlaadp_keep_visual_tokens 224
--vlaadp_qk_layer 0
--vlaadp_main_view_weight 0.4
--vlaadp_wrist_view_weight 0.6
--vlaadp_use_dynamic_visual_strategy True
--vlaadp_cold_start_windows 2
--vlaadp_limit_max_consecutive_pruned 3
```

## NORA / LIBERO

主要入口在 `integrations/nora/libero/`：


| 文件                                | 作用                                                                       |
| ----------------------------------- | -------------------------------------------------------------------------- |
| `run_libero_eval_nora.py`           | NORA LIBERO 评测入口                                                       |
| `nora_utils.py`                     | NORA 推理、模型包装与 pruning hook                                         |
| `autoprune_nora.py`                 | NORA AutoPrune 接入                                                        |
| `compare_pruning_patch_nora.py`     | NORA baseline pruning：FastV、SparseVLM、FasterVLM、VisionZip、PyramidDrop |
| `vlapruner_nora.py`                 | VLA-Pruner 的 NORA/Qwen2.5-VL 适配                                         |
| `vlaadp_nora.py`                    | VLA-ADP 的 NORA/Qwen2.5-VL 适配                                            |
| `profile_nora_hotspots.py`          | NORA 性能热点分析                                                          |
| `libero_utils.py`, `robot_utils.py` | LIBERO 环境辅助函数                                                        |

常用 launch/debug 脚本在 `integrations/nora/scripts/`，覆盖 FastV、SparseVLM、FasterVLM、VisionZip、PyramidDrop、AutoPrune、VLA-Pruner 和 VLA-ADP 的若干 LIBERO 任务。新增脚本包括：

| 脚本 | 作用 |
| --- | --- |
| `run_vlapruner_nora_libero.sh` | NORA-LIBERO VLA-Pruner 单实验入口 |
| `run_vlaadp_nora_libero.sh` | NORA-LIBERO VLA-ADP 单实验入口 |
| `run_table_vlapruner_nora_queue.sh` | VLA-Pruner-NORA 表格实验队列 |
| `run_table_vlaadp_nora_queue.sh` | VLA-ADP-NORA 表格实验队列 |
| `debug_vlapruner_nora_long_task0.sh` | VLA-Pruner-NORA Long task0 debug 脚本 |
| `debug_vlaadp_nora_long_task0.sh` | VLA-ADP-NORA Long task0 debug 脚本 |

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
| FasterVLM   | Active standalone baseline | Active via compare patch | Active adapter |
| VisionZip   | Active standalone baseline | Active via compare patch | Active adapter |
| PyramidDrop | Active standalone baseline | Active via compare patch | Active adapter |
| VLA-Pruner  | Active             | Active                   | Planned        |
| VLA-ADP     | Active             | Active                   | Planned        |

## 实验指标

OpenVLA-OFT 和 NORA 当前实验主要记录：

- `successful-rate`：LIBERO task success rate。
- `latency_ms/step`：每步动作预测总延迟。
- `model_latency_ms`、`predict_action_latency_ms`、`decoder_forward_latency_ms`：模型内部延迟拆分。
- `pruned_decoder_compute_latency_ms`：剪枝后 decoder 层计算延迟。
- `flops_reduction`：基于每层实际 token 数的近似 FLOPs 缩减率。
- `final_visual_tokens`、`keep_count_schedule`、`view_keep_counts`：剪枝预算和最终视觉 token 数。
- `vlapruner_*`：VLA-Pruner 的 temporal window、history ready、semantic/temporal score mass 等状态。
- `vlaadp_*`：VLA-ADP 的 dynamic gate state、trajectory delta、cold-start、forced-full 和连续剪枝限制状态。

这些字段通过各 patch 的 `latest_stats` / runtime stats 暴露，由评测入口汇总，保证 AutoPrune、FastV、SparseVLM、VLA-Pruner、VLA-ADP 等方法比较时使用同一套日志接口。

## 运行与批量实验

OpenVLA-OFT 单次实验示例：

```bash
cd integrations/openvla_oft/scripts
GPU=0 TASK_SUITE_NAME=libero_spatial TASK_ID=0 KEEP_VISUAL_TOKENS=224 NUM_TRIALS=20 \
  bash run_vlapruner_oft_single.sh

GPU=0 TASK_SUITE_NAME=libero_10 TASK_ID=0 KEEP_VISUAL_TOKENS=224 NUM_TRIALS=20 \
  bash run_vlaadp_oft_single.sh
```

NORA 单次实验示例：

```bash
cd integrations/nora/scripts
GPU=0 TASK_SUITE=libero_10 TASK_START_ID=0 KEEP_VISUAL_TOKENS=224 NUM_TRIALS=20 \
  bash run_vlapruner_nora_libero.sh

GPU=0 TASK_SUITE=libero_10 TASK_START_ID=0 KEEP_VISUAL_TOKENS=224 NUM_TRIALS=20 \
  bash run_vlaadp_nora_libero.sh
```

通用表格调度脚本：

```text
scripts/launch_pruning_table_tmux.sh
```

该脚本用于把多个 suite/task/keep 组合放入 tmux 队列，具体方法侧队列脚本位于 `integrations/openvla_oft/scripts/` 和 `integrations/nora/scripts/`。
