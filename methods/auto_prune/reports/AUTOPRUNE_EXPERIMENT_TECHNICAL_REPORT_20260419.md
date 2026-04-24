# AutoPrune 技术实验报告（OpenVLA-OFT / NORA / Mantis）

## 1. 报告目标

这份文档聚焦回答两个问题：

1. 我们在 `Auto_openvla-oft`、`auto_nora`、`Mantis` 三套系统上，具体是怎样开展实验的。
2. 三套系统的任务形态、评测入口、关键配置和结果组织方式分别是什么，为什么这些差异会影响 AutoPrune 的表现。

本文只描述**实验实施方式**与**工程协议**，不展开算法细节；算法本身与实现改动放在另一份文档中。

---

## 2. 总体实验框架

### 2.1 三套基线的定位

- `Auto_openvla-oft`
  - 任务类型：`VLA`，机器人闭环控制
  - 主要 benchmark：`LIBERO`
  - 输出形式：动作序列 / action chunk
  - 核心关注：成功率、step latency、视觉 token 保留规模

- `auto_nora`
  - 任务类型：`VLA`，机器人闭环控制
  - 主要 benchmark：`LIBERO`
  - 输出形式：动作序列 / action chunk
  - 核心关注：成功率、step latency、预算-性能曲线、系统级加速效果

- `Mantis`
  - 任务类型：多图视觉语言推理
  - 主要 benchmark：`BLINK / Multi-view_Reasoning`
  - 输出形式：离散答案或受控文本输出
  - 核心关注：准确率、端到端延迟、剪枝层数压缩效果

### 2.2 为什么要同时做这三套

我们不是只想证明 AutoPrune 在某一个模型上“能跑”，而是要验证它在三类不同形态上的适用性：

- 在 `OpenVLA-OFT` 上，验证 AutoPrune 对多视角机器人控制是否有效。
- 在 `NORA` 上，验证 AutoPrune 是否能迁移到另一套 VLA 主干，并结合更激进的系统侧优化。
- 在 `Mantis` 上，验证 AutoPrune 在多图视觉推理任务中的收益，并剥离掉 VLA 闭环中的环境噪声。

因此，这三套实验不是重复劳动，而是分别承担：

- `VLA 主验证`
- `VLA 迁移验证`
- `多图推理解释性验证`

---

## 3. OpenVLA-OFT 实验如何进行

### 3.1 代码入口与核心文件

OpenVLA 相关实验主要依赖以下文件：

- 评测入口：
  - [run_libero_eval.py](/home/jarvis/Auto_openvla-oft/experiments/robot/libero/run_libero_eval.py)
- AutoPrune 主 patch：
  - [autoprune_patch_improved_1.py](/home/jarvis/Auto_openvla-oft/autoprune_patch_improved_1.py)
- 预算策略：
  - [adaptive_budget_strategy.py](/home/jarvis/Auto_openvla-oft/adaptive_budget_strategy.py)
- 原始 AutoPrune 参考实现：
  - [modelling_sparse_llama.py](/home/jarvis/Auto_openvla-oft/AutoPrune/llava/model/language_model/modelling_sparse_llama.py)

### 3.2 实验对象与任务设置

OpenVLA-OFT 这条线使用的是 `LIBERO` 四个 suite：

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`

常见实验规模包括：

- `4 suite × 10 task × 20 episode`
- 或小规模 smoke：
  - `单 suite / 单 task / 10 trial`
  - 用于 patch 验证、基线迁移验证和 latency 校验

### 3.3 OpenVLA 上的关键实验参数

`run_libero_eval.py` 中暴露了完整的 AutoPrune 控制面，核心参数包括：

- 是否启用 AutoPrune：
  - `use_auto_prune`
- 目标视觉预算：
  - `target_token_num`
  - `visual_tokens`
- 原始预算曲线参数：
  - `x0`
  - `k0`
  - `gamma`
- 剪枝起始层：
  - `pruning_start_layer`
- 文本采样与安全边界：
  - `text_sample_ratio`
  - `safe_ratio`
  - `min_keep_per_view`
- 预算模式：
  - `budget_mode`
  - `absolute_anchor_mode`
  - `absolute_budget_scale`
  - `absolute_auto_bounds`
- 层数策略：
  - `enable_adaptive_layer_policy`
  - `adaptive_layer_choices`
- one-shot fallback：
  - `enable_oneshot_fallback`
- patch 版本切换：
  - `autoprune_patch_variant`

这说明 OpenVLA 上的实验不是“只改一个 target token 数”，而是一个完整的多维控制系统。

### 3.4 OpenVLA 上的实验流程

典型实验流程如下：

1. 在 `run_libero_eval.py` 中指定 checkpoint、suite、task 和 AutoPrune 参数。
2. 由 `autoprune_patch_improved_1.py` 将 AutoPrune 挂接到 OpenVLA 的多模态 forward 流程中。
3. 在 `LIBERO` 环境中执行 rollout。
4. 记录：
   - episode success
   - step latency
   - pruning statistics
   - budget / keep schedule / active pruning layers
5. 将结果写入 `experiments/logs/...`。

### 3.5 OpenVLA 上做过的实验类型

OpenVLA 这条线我们主要做了四类实验：

1. **AutoPrune 主实验**
   - 测试不同预算下的成功率与时延
   - 比较不同预算语义与层数策略

2. **compare baseline 迁移实验**
   - 将 `VisionZip`、`FasterVLM`、`PyramidDrop`、`SparseVLM` 等迁移到 OpenVLA
   - 检查公平挂载与 faithful port 的影响

3. **单卡真实 latency 实验**
   - 为避免同卡多任务并发污染 latency
   - 在单卡单任务条件下重测不同预算的真实 step latency

4. **调试/诊断实验**
   - 检查 compare baseline 的挂载公平性
   - 检查是否错误借用了 AutoPrune 的打分或预算逻辑
   - 检查日志字段是否正确记录

### 3.6 OpenVLA 实验的主要工程特点

OpenVLA 这条线的难点不在于“把 token 删掉”，而在于**删掉以后模型还能正确输出动作**。因此实验里必须维护：

- `position_ids`
- `attention_mask`
- `global_indices`
- 多视角 visual span 对齐
- action chunk 的正确提取位置

也正因为如此，OpenVLA 的实验不仅是算法实验，也是一个强工程约束下的系统实验。

---

## 4. NORA 实验如何进行

### 4.1 代码入口与核心文件

NORA 相关实验主要依赖：

- AutoPrune 实现：
  - [autoprune_nora.py](/home/jarvis/auto_nora/experiments/libero/autoprune_nora.py)
- 评测入口：
  - [run_libero_eval_nora.py](/home/jarvis/auto_nora/experiments/libero/run_libero_eval_nora.py)
- 项目总说明：
  - [README.md](/home/jarvis/auto_nora/README.md)

### 4.2 NORA 的任务与评测口径

NORA 这条线同样使用 `LIBERO` 系列任务，但它的目标不仅是“把 AutoPrune 迁过去”，还包括：

- 验证 AutoPrune 在另一套 VLA 主干上的可迁移性
- 验证更激进的系统优化是否会改变预算-性能曲线
- 验证 progressive pruning 与 one-shot fallback 在 NORA 上的边界条件

### 4.3 NORA 上的配置组织方式

NORA 在工程上比 OpenVLA 更激进，配置项也更丰富。其 `NoraAutoPruneConfig` 中除了基本 AutoPrune 参数外，还包括：

- 基础参数
  - `target_token_num`
  - `visual_tokens`
  - `start_layer`
  - `num_views`

- 预算控制
  - `allocation_temp`
  - `x0`
  - `k_shape`
  - `b_min`
  - `b_max`
  - `mu`
  - `tau`
  - `budget_mode`
  - `absolute_anchor_mode`
  - `absolute_budget_scale`
  - `absolute_auto_bounds`

- 层数与 schedule 控制
  - `enable_adaptive_layer_policy`
  - `adaptive_layer_choices`
  - `enable_oneshot_fallback`
  - `piecewise_linear_keep_schedule`
  - `budget_semantics`

- 系统优化控制
  - `prefer_flash_attention`
  - `manual_attention_probe`
  - `prune_only_on_anchor_layers`
  - `budget_aware_x0_shift`
  - `compaction_delta_threshold_tokens`
  - `physical_compaction_max_shots`

这说明 NORA 上的实验不仅在做算法，还在做**推理系统层面的工程压缩**。

### 4.4 NORA 的实验流程

典型 NORA 实验流程是：

1. 在 `run_libero_eval_nora.py` 中选择 AutoPrune preset 或手动指定参数。
2. 用 `autoprune_nora.py` 中的 patch 与配置接入 NORA 主干。
3. 在 `LIBERO` 环境中 rollout。
4. 记录 success rate、latency、budget schedule、fallback 触发情况、层压缩统计。

### 4.5 NORA 上的典型实验形态

NORA 特别适合比较以下几种设定：

- `progressive pruning`
- `single shot pruning`
- `flash attention + pruning`
- `adaptive layer count`
- `one-shot fallback`

也就是说，NORA 不是单纯复刻 OpenVLA，而是更像一个“AutoPrune 在 VLA 上的系统优化实验台”。

### 4.6 NORA 与 OpenVLA 的差异

尽管两者都跑 `LIBERO`，但 NORA 和 OpenVLA 的实验意义不同：

- OpenVLA 更强调**基准 VLA 主干上的可靠迁移**
- NORA 更强调**在更复杂系统优化下，AutoPrune 的可扩展性**

因此我们在 NORA 上的实验，更适合回答：

- AutoPrune 是否能与更激进的推理优化共存
- adaptive layer policy 和 one-shot fallback 是否真的有系统价值

---

## 5. Mantis 实验如何进行

### 5.1 代码入口与核心文件

Mantis 相关实现主要依赖：

- patch 层：
  - [autoprune_mantis_patch.py](/home/jarvis/Mantis/mantis/models/mllava/autoprune_mantis_patch.py)
- 剪枝算法层：
  - [autoprune_mantis_pruning.py](/home/jarvis/Mantis/mantis/models/mllava/autoprune_mantis_pruning.py)
- 配置与公共函数：
  - [autoprune_mantis_common.py](/home/jarvis/Mantis/mantis/models/mllava/autoprune_mantis_common.py)
- 项目说明：
  - [README.md](/home/jarvis/Mantis/README.md)
- 已有阶段性实验总结：
  - [MANTIS_AUTOPRUNE_GROUP_REPORT_20260331.md](/home/jarvis/Mantis/MANTIS_AUTOPRUNE_GROUP_REPORT_20260331.md)

### 5.2 Mantis 的任务定义

Mantis 上的 AutoPrune 实验主要不是闭环控制，而是多图推理，尤其是：

- `BLINK`
- `Multi-view_Reasoning`
- `ablation1`
- `loss-batched` 协议

这种任务相较于 VLA 有两个关键优势：

1. 输出空间更可控
2. 延迟更容易稳定测量

因此，Mantis 很适合验证：

- 剪枝对精度的影响
- progressive pruning 是否真的带来 latency 收益
- 层数压缩与预算器是否合理

### 5.3 Mantis 上的核心实验参数

`MantisAutoPruneConfig` 中的关键配置包括：

- 开关与层数
  - `enabled`
  - `start_layer`
  - `max_pruning_layers`
  - `enable_adaptive_layer_policy`
  - `enable_oneshot_fallback`
  - `adaptive_layer_choices`

- 预算器
  - `target_token_num`
  - `budget_mode`
  - `absolute_auto_bounds`
  - `absolute_anchor_mode`
  - `absolute_budget_scale`
  - `b_min`
  - `b_max`
  - `mu`
  - `tau`
  - `x0`
  - `k_shape`

- 多视角/分配
  - `min_keep_per_view`
  - `allocation_temp`
  - `coverage_min`
  - `coverage_max`
  - `enable_image_prior`
  - `image_prior_weight`
  - `warm_keep_factor`

- 诊断与 ablation
  - `monitor_only`
  - `disable_token_prune`
  - `disable_progressive_prune`
  - `disable_kv_cache`
  - `debug`

### 5.4 Mantis 的实验流程

Mantis 实验流程通常是：

1. 构造多图输入与问题文本。
2. 在 `autoprune_mantis_patch.py` 中注入 AutoPrune patch。
3. 由 `autoprune_mantis_pruning.py` 负责 layer policy、schedule 压缩、fallback 判断和 token 选择。
4. 运行 BLINK 推理。
5. 汇总：
   - 准确率
   - latency
   - 平均 active pruning layers
   - 平均保留 token 数

### 5.5 为什么 Mantis 很重要

Mantis 这条线在整个研究里承担了一个非常重要的角色：

- VILA 这类开放式长文本生成任务，很难把视觉剪枝的理论收益直接转成端到端收益。
- Mantis 的多图推理任务则更容易把视觉 token 压缩与最终性能直接对应起来。

因此，Mantis 不是“附带实验”，而是帮助我们把：

- 算法设计
- latency 改善
- 精度保持

三件事真正放到同一个平台上解释清楚。

---

## 6. 三套实验的共同点与差异

### 6.1 共同点

三套实验都围绕同一核心问题：

> 如何在不重新训练模型的前提下，对视觉 token 做自适应剪枝，并尽量保住性能同时降低计算量。

共同结构包括：

- progressive pruning
- 动态预算
- keep schedule
- 多视角场景下的 token 分配
- 对比不同预算点的性能

### 6.2 差异

三套实验最核心的差异在于任务输出形态不同：

- `OpenVLA-OFT / NORA`
  - 是动作控制问题
  - 一个局部视觉错误会在 rollout 中累计
  - success rate 对视觉细节更敏感

- `Mantis`
  - 是多图受控推理问题
  - 没有动作闭环累积误差
  - 更适合分析预算器和层压缩的纯算法价值

### 6.3 为什么这对实验解释很重要

正因为任务形态不同，我们不能简单地拿：

- VLM 上的视觉 token 剪枝结论

直接推导到：

- VLA 上一定也成立

这也是为什么我们必须同时在 OpenVLA、NORA、Mantis 三条线上做实验。

---

## 7. 我们在三套系统上实际验证的内容

综合来看，这三套实验系统一共验证了 AutoPrune 的三类能力：

### 7.1 算法有效性

- 预算是否能够随样本复杂度变化
- progressive pruning 是否优于简单 one-shot
- adaptive layer policy 是否比固定层数更合理

### 7.2 工程可迁移性

- 是否能迁到不同模型主干
- 是否能兼容多视角输入
- 是否能保持序列对齐、位置编码和索引一致性

### 7.3 结果可解释性

- 不同任务下为什么预算会变化
- 为什么 VLM 风格基线迁到 VLA 上时会退化
- 为什么 Mantis 上的 latency 收益更容易显现

---

## 8. 当前实验结论的技术意义

从实验设计本身来看，目前最重要的结论不是“哪条线数字最高”，而是：

1. AutoPrune 已经不再局限于原始单模型场景，而是可以迁移到三种明显不同的系统中。
2. OpenVLA 与 NORA 证明了它在 VLA 闭环任务中的可用性与工程可接入性。
3. Mantis 证明了它在多图推理任务上的解释性和稳定 latency 收益。
4. 三套系统合在一起，使我们可以把 AutoPrune 说成一个**跨平台的视觉 token 自适应剪枝框架**，而不只是某一份 patch。

---

## 9. 总结

这三套实验并不是三份彼此独立的工程，而是一套完整验证链：

- `OpenVLA-OFT`：验证基准 VLA 主干上的有效性
- `NORA`：验证迁移性与系统优化兼容性
- `Mantis`：验证多图推理中的算法解释性与 latency 收益

因此，如果后续要对外说明 AutoPrune 的实验设计，最准确的说法是：

> 我们分别在机器人闭环控制基线（OpenVLA-OFT、NORA）和多图视觉语言推理基线（Mantis）上进行了 AutoPrune 的系统性实验，通过统一的视觉 token 剪枝思想，在不同任务形态下验证了算法有效性、工程可迁移性和结果可解释性。

