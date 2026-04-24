# AutoPrune 算法实现报告（原始算法 + OpenVLA-OFT / NORA / Mantis 三套实现）

## 1. 报告目标

这份文档专门回答三个问题：

1. 原始 AutoPrune 算法本身是什么。
2. 我们后来新增或强化了哪些关键设计。
3. AutoPrune 在 `OpenVLA-OFT`、`NORA`、`Mantis` 三套系统里，具体是怎么实现的。

本文重点是**算法与实现机制**，不重复展开实验 protocol。

---

## 2. 原始 AutoPrune 算法是什么

### 2.1 原始问题定义

原始 AutoPrune 处理的是一个典型问题：

> 多模态模型在进入语言模型后，视觉 token 数量太大，推理计算随层数累积，导致延迟和显存压力都很高。

因此，它的目标不是重新训练一个稀疏模型，而是：

- 在推理时
- 不额外训练
- 对视觉 token 做逐层压缩
- 同时尽量保持任务性能

### 2.2 原始算法的核心思想

原始 AutoPrune 不是一次性把视觉 token 剪掉，而是：

1. 在若干候选层上观察文本对视觉区域的关注情况。
2. 给每个视觉 token 一个重要性分数。
3. 按预先决定的保留预算，在当前层留下更重要的 token。
4. 把剩余 token 继续送到后续层。
5. 在后续层重复这一过程。

因此，原始 AutoPrune 的本质是：

**progressive layerwise visual token pruning**

它的关键不是“某一层 top-k”，而是“预算如何逐层下降、在哪些层剪、每层保留多少”。

### 2.3 原始实现参考位置

在当前项目中，原始风格最接近的参考文件是：

- [modelling_sparse_llama.py](/home/jarvis/Auto_openvla-oft/AutoPrune/llava/model/language_model/modelling_sparse_llama.py)

它体现了原始 AutoPrune/Sparse 模型族的几个典型特征：

- 剪枝发生在语言模型内部层间
- 使用文本-视觉相关信号决定保留集合
- 有 progressive 的 layer schedule
- 有动态预算或保留比例逻辑

### 2.4 原始算法的几个关键组件

#### 2.4.1 复杂度信号

原始 AutoPrune 并不是固定预算，而是想根据样本复杂度调节保留量。  
复杂度通常来自两类信号：

- 文本信息复杂度
- 视觉 token 信息密度

在我们后续代码中，最常见的实现是：

- 计算 `MI`、`KL`
- 或使用与之等价的信息量指标

#### 2.4.2 动态预算

原始思想可以概括为：

> 文本越复杂，需要更多视觉 token。  
> 单个视觉 token 信息越密，则可保留得更少。

也就是典型的：

`预算 ~ 文本复杂度 / 视觉信息密度`

#### 2.4.3 Progressive keep schedule

原始 AutoPrune 不是把最终预算直接应用在第一层，而是先生成一条 schedule，例如：

- 第 1 个剪枝层保留较多
- 中间层逐渐减少
- 后面层收缩到目标预算

#### 2.4.4 层间索引维护

一旦 token 被剪掉，模型内部必须持续追踪：

- 当前 token 对应原始哪一块图像区域
- 后续层的 visual span 怎么更新

这也是原始算法在工程落地时最容易出错的地方之一。

---

## 3. 我们在原始算法上的新增设计

原始 AutoPrune 提供了基本框架，但直接拿来并不能很好适应三套系统。因此我们逐步增加了几类设计。

### 3.1 预算从“固定上下限”走向“absolute + anchor”

原始动态预算逻辑虽然有解释性，但很容易依赖人工上下限。  
我们后续把预算语义扩展成：

- `absolute`
  - 保留原来的“文本复杂度 / 视觉信息密度”解释链
- `anchor`
  - 用样本自身注意力分布中的拐点或自然预算位置做锚点
- `scale`
  - 给不同模型、不同实验提供统一粗调旋钮

对应实现主要在：

- [adaptive_budget_strategy.py](/home/jarvis/Auto_openvla-oft/adaptive_budget_strategy.py)

### 3.2 剪枝层数自适应压缩

原始 progressive 方法往往默认固定候选层集。  
我们后续增加了：

- `adaptive layer policy`
- `dense schedule compression`

核心思想是：

1. 先生成完整 schedule
2. 再删掉那些实际上没有变化、没有必要执行的层

这样简单样本会自动退化成更少层数，复杂样本仍保留多层 progressive。

### 3.3 One-shot fallback

对于极简单样本，如果 progressive schedule 最后只剩一个真正有效的变化点，继续走多层流程没有意义。  
因此我们增加：

- `one-shot fallback`

它不是放弃 progressive，而是把 progressive 在极简单样本上的自然退化显式化。

### 3.4 多视角 token 分配

当任务是多视角输入时，预算不仅是“总共保留多少”，还包括：

> 每个视角各保留多少

这要求增加：

- 多视角 keep count 分配
- 最小保留数约束
- coverage / score mass / warm keep 等多视角保护机制

### 3.5 运行正确性保障

在 VLA 和多图任务里，单纯“算出 token 排序”不够，还必须保证剪枝后模型行为仍正确。  
因此三套实现都强调：

- `position_ids` 一致性
- `attention_mask` 一致性
- `global_indices` 一致性
- visual span / action span / image span 的持续对齐

---

## 4. AutoPrune 在 OpenVLA-OFT 中如何实现

### 4.1 关键文件

- [autoprune_patch_improved_1.py](/home/jarvis/Auto_openvla-oft/autoprune_patch_improved_1.py)
- [adaptive_budget_strategy.py](/home/jarvis/Auto_openvla-oft/adaptive_budget_strategy.py)
- [run_libero_eval.py](/home/jarvis/Auto_openvla-oft/experiments/robot/libero/run_libero_eval.py)

### 4.2 OpenVLA 实现的总体结构

OpenVLA 版本不是简单替换一个函数，而是在模型 forward 链路中插入一套 patch，完成：

1. 获取多视角 visual token
2. 计算文本-视觉相关信号
3. 决定动态预算
4. 生成 progressive keep schedule
5. 在指定层执行 token gather
6. 更新索引与序列结构
7. 继续让后续层在压缩后的 token 序列上前向

### 4.3 OpenVLA 中的预算器实现

OpenVLA 的现代预算器核心在：

- `AdaptiveBudgetStrategy`
- `decide_budget(...)`

它支持两类口径：

1. 传统 sigmoid 预算映射
2. `absolute + anchor + scale`

输出通常包括：

- `raw_budget`
- 预算来源说明
- anchor / scale / bound 相关统计

### 4.4 OpenVLA 中的多视角处理

OpenVLA 的一个关键难点是输入通常不是单张图，而是多视角图像。  
因此 patch 里需要解决：

- 多个 view 的 visual token span 切分
- 每个 view 的局部重要性统计
- 总预算如何分给各个 view
- 最小保留阈值如何约束

这使得 OpenVLA 版本不是原始单视角 AutoPrune 的直接复制，而是一个**多视角扩展版**。

### 4.5 OpenVLA 中的运行正确性实现

OpenVLA 上最关键的不是“选谁保留”，而是“保留之后还能正确输出动作”。  
因此 patch 中必须维护：

- `position_ids`
- `attention_mask`
- `global_indices`
- image token span
- action token / regression token 的最终提取位置

如果这些不维护，最直接后果就是：

- rotary embedding 错位
- 序列 gather 后动作头取错位置
- rollout 成功率异常崩塌

### 4.6 OpenVLA 中 compare baseline 的接法

OpenVLA 还承担了 compare baseline 迁移实验，因此我们在实验中进一步区分：

- **可共享的运行基础设施**
  - 动作 chunk 提取
  - `position_ids / attention_mask / global_indices` 维护

- **不能借给 baseline 的 AutoPrune 策略**
  - `absolute + anchor` 预算器
  - 自己的 text-to-vision 打分
  - 跨视角预算分配策略
  - adaptive layer policy
  - dense schedule compression

这个边界对于保证 compare 实验公平性非常重要。

---

## 5. AutoPrune 在 NORA 中如何实现

### 5.1 关键文件

- [autoprune_nora.py](/home/jarvis/auto_nora/experiments/libero/autoprune_nora.py)
- [run_libero_eval_nora.py](/home/jarvis/auto_nora/experiments/libero/run_libero_eval_nora.py)

### 5.2 NORA 实现的总体思路

NORA 版本可以理解为：

> AutoPrune 算法核心 + 更强的系统优化与执行控制

它保留了 AutoPrune 的核心思想：

- 动态预算
- progressive pruning
- adaptive layer policy
- one-shot fallback

同时加入更多执行层策略，以便更好适配 NORA 的推理栈。

### 5.3 NORA 中的核心算法组件

在 `autoprune_nora.py` 中，以下函数最能代表算法主体：

- `select_progressive_pruning_layers`
- `resolve_adaptive_pruning_layer_count`
- `compress_keep_count_schedule`
- `should_use_oneshot_fallback`
- `allocate_keep_counts(...)`

这说明 NORA 中的 AutoPrune 已经不是一个单函数逻辑，而是一套完整的：

- 层选择
- 预算决定
- schedule 压缩
- fallback 触发
- 多视角分配

联合系统。

### 5.4 NORA 中的系统增强点

NORA 比 OpenVLA 额外强调下面这些系统能力：

- `prefer_flash_attention`
- `manual_attention_probe`
- `prune_only_on_anchor_layers`
- `piecewise_linear_keep_schedule`
- `budget_aware_x0_shift`
- `compaction_delta_threshold_tokens`
- `physical_compaction_max_shots`

这些配置表明：

NORA 的实现并不满足于“理论预算正确”，而是继续追求：

- 更少无效 compaction
- 更稳的注意力探针
- 更强的硬件友好性

### 5.5 NORA 中的意义

如果说 OpenVLA 版本主要回答：

> AutoPrune 能否在标准 VLA 主干中正确工作？

那么 NORA 版本回答的则是：

> AutoPrune 能否在一个更激进、更工程化的 VLA 系统中，继续保持可解释性和可控性？

因此 NORA 是 AutoPrune 从“算法 patch”走向“推理系统组件”的关键一步。

---

## 6. AutoPrune 在 Mantis 中如何实现

### 6.1 关键文件

- [autoprune_mantis_common.py](/home/jarvis/Mantis/mantis/models/mllava/autoprune_mantis_common.py)
- [autoprune_mantis_pruning.py](/home/jarvis/Mantis/mantis/models/mllava/autoprune_mantis_pruning.py)
- [autoprune_mantis_patch.py](/home/jarvis/Mantis/mantis/models/mllava/autoprune_mantis_patch.py)

### 6.2 Mantis 实现的总体结构

Mantis 的实现结构非常清晰，可以分成三层：

1. **common 层**
   - 配置
   - 预算器
   - 公共信息量函数

2. **pruning 层**
   - 层数选择
   - schedule 生成与压缩
   - fallback 逻辑

3. **patch 层**
   - 挂接模型 forward
   - 保存运行时状态
   - 在具体 decoder 层执行 gather

这是一套相对模块化、最利于解释的 AutoPrune 落地结构。

### 6.3 Mantis 中的配置语义

`MantisAutoPruneConfig` 里最关键的语义是三件事：

1. **预算怎么来**
   - `target_token_num`
   - `budget_mode`
   - `absolute_anchor_mode`
   - `absolute_budget_scale`

2. **剪多少层**
   - `max_pruning_layers`
   - `enable_adaptive_layer_policy`
   - `adaptive_layer_choices`
   - `enable_oneshot_fallback`

3. **多图/多视角如何保护**
   - `min_keep_per_view`
   - `coverage_min`
   - `coverage_max`
   - `enable_image_prior`
   - `image_prior_weight`
   - `warm_keep_factor`

### 6.4 Mantis 中的 schedule 压缩

Mantis 版本把“剪枝层数自适应压缩”做得最明确。  
`autoprune_mantis_pruning.py` 中的关键函数有：

- `generate_schedule_aligned`
- `select_progressive_pruning_layers`
- `resolve_adaptive_pruning_layer_count`
- `compress_keep_count_schedule`
- `should_use_oneshot_fallback`

这使得 Mantis 成为我们解释 AutoPrune 新增设计最理想的平台，因为：

- 结构清楚
- 输出指标稳定
- latency 更好测

### 6.5 Mantis 中的 patch 执行逻辑

`MantisAutoPrunePatch` 的职责不是做算法决策，而是把算法决策正确执行进模型：

- 保存输入布局
- 维护运行时 `global_indices`
- 记录 `keep_count_schedule`
- 在各层触发 pruning
- 在 token gather 后同步更新：
  - `position_ids`
  - `attention_mask`
  - 序列布局

因此，Mantis 的 patch 与 OpenVLA patch 一样，本质上都承担了一个职责：

> 把“应该删谁”的算法决策，转换成“删完以后模型仍然正确前向”的运行逻辑

---

## 7. 三套实现之间的关系

### 7.1 不变的算法核心

无论在 OpenVLA、NORA 还是 Mantis 中，AutoPrune 的核心都没有变：

- 估计样本复杂度
- 决定视觉预算
- 生成 progressive keep schedule
- 在若干层逐步压缩视觉 token
- 尽量保住任务表现

### 7.2 因任务不同而变化的部分

真正变化的是：

1. **输入结构**
   - 单图 / 多图 / 多视角

2. **输出结构**
   - 文本生成
   - 动作回归
   - 受控推理答案

3. **运行正确性要求**
   - 是否需要 action chunk 对齐
   - 是否有多视角 span
   - 是否要同步 image token 与文本 token 的复合布局

### 7.3 因系统目标不同而变化的部分

- `OpenVLA`
  - 更偏“可靠正确地接入标准 VLA 主干”
- `NORA`
  - 更偏“系统优化与 AutoPrune 并存”
- `Mantis`
  - 更偏“把 AutoPrune 的预算器与层压缩解释清楚”

---

## 8. 为什么 compare baseline 不能直接借用 AutoPrune 逻辑

在实际工程里，很容易出现一种诱惑：

> 为了让 compare baseline 在系统里跑得更稳，直接借用 AutoPrune 的预算器、层压缩或打分方式。

但这会破坏公平性。

因此我们后续逐步明确了边界：

### 8.1 可以共享的

- 动作 chunk 提取方式
- `position_ids` 一致性维护
- `attention_mask` 一致性维护
- `global_indices` 一致性维护
- 正确的 patch 入口

这些属于“系统正确运行基础设施”。

### 8.2 不能共享的

- `absolute + anchor` 预算器
- 自己的 text-to-vision 打分逻辑
- 跨视角预算分配策略
- adaptive layer policy
- dense schedule compression
- one-shot fallback 决策逻辑

这些属于 AutoPrune 的算法贡献，不应该借给对比方法。

---

## 9. 总结

原始 AutoPrune 的核心是一套：

- 基于样本复杂度的动态预算
- 基于层间 schedule 的 progressive visual token pruning

而我们后续在三套系统上把它发展成了一个更完整的框架：

- 在 `OpenVLA-OFT` 中，重点解决多视角 VLA 的预算、索引与动作对齐问题。
- 在 `NORA` 中，重点验证 AutoPrune 与更复杂系统优化的兼容性。
- 在 `Mantis` 中，重点把预算器、层压缩和 fallback 机制做成最清晰、最可解释的实现。

因此，当前的 AutoPrune 已经不能简单理解为“一个在 LLaMA 某几层做 top-k gather 的小 patch”，而应该理解为：

> 一套跨多模态平台的、可解释的、可迁移的视觉 token 自适应剪枝框架。

