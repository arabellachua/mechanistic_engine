# 第三轮：试验文本与非线性模型

日期：2026-09-21。

本轮找到了更有潜力的特征方向，但**尚未达到可靠显著提升的目标**。试验文本与 biology core 联合建模，在新 target–disease 组合验证上有较大增益；时间验证中的改善集中在较早窗口。预先定义的嵌套选模验收条件未通过，因此没有替换主模型或声称独立测试改善。

## 做了什么

- 固定旧快照的 472 条、2015 年及以前的训练记录，其中 92 条正例。166 条较晚年份记录本轮未评分。
- 原始 CT.gov 数据匹配到 469 条；3 条缺失记录保留，显式标记缺失，没有删行。原始文件、标签、旧模型和旧结果均未覆盖。
- 新输入严格限定为 `conditionsModule.conditions`、DRUG/BIOLOGICAL 干预的 `name`、`eligibilityModule.eligibilityCriteria`。不读取标题、摘要、状态、结果、结局或标签来源作为模型输入。
- 文本使用 TF-IDF 单词/双词特征；词表、缺失填充、标准化均在各训练折内拟合。
- 预设 12 个候选：数值与类别参照、较强正则化、RBF SVM、二阶交互、条件/药物文本、入组标准文本，以及文本与数值联合模型。未安装新依赖。
- 三折滚动时间验证和三折未见 target–disease 组合验证；每个外层训练池再执行内层选模，先保存选择再评分外层。共 180 次内层拟合、72 次外层拟合。
- 预设实际增益目标：两种验证中，嵌套选择相对数值参照平均 AUC 增加至少 0.05，按 target–disease pair 聚类的描述性区间下界大于 0，且 Brier 不恶化。这只是开发阶段门槛，不是独立显著性确认。

## 固定候选结果

所有数字均为三个外层验证折指标的算术平均，不是测试集指标。

| 模型 | 时间 AUC | 新组合 AUC | 时间 Brier ↓ | 新组合 Brier ↓ |
|---|---:|---:|---:|---:|
| 原 biology core 数值参照 | 0.6865 | 0.6207 | 0.1473 | 0.1439 |
| 上轮 target / disease / modality 模型 | 0.6932 | 0.6380 | 0.1585 | 0.1519 |
| 本轮文本 + core，C=1 | **0.7075** | **0.6954** | 0.1504 | **0.1410** |
| 本轮文本 + core，C=10 | 0.7110 | 0.6994 | 0.1652 | 0.1501 |
| 严格内层选择的模型流程 | 0.6627 | 0.6615 | 0.1680 | 0.1498 |

`protocol_core_c1` 是看过外层结果后重点诊断的候选，不能视为预先指定的唯一假设。相对数值参照：

- 新组合 AUC 增加 **0.0747**；2,000 次聚类 bootstrap 描述性 95% 区间为 **[0.0237, 0.1338]**。三个外层折分别增加 0.0469、0.0930、0.0843；Average Precision 从 0.4088 到 0.4539。
- 时间 AUC 增加 **0.0211**；对应区间为 **[-0.0435, 0.0883]**。Average Precision 从 0.4395 到 0.4938，但 Brier 略变差。
- 相对上轮类别模型，新组合 AUC 增加 **0.0574**，时间 AUC 增加 **0.0143**。

这些候选区间**未校正多模型比较**，且在反复使用过的开发集上计算，不能据此宣称已获得可泛化的统计显著提升。置信区间条件于固定预测，不包含重新训练与选模的不确定性。

## 为什么没有通过验收

固定 C=1 联合模型的时间验证如下：

| 验证年份 | 数值参照 AUC | 文本 + core AUC | 差值 |
|---|---:|---:|---:|
| 2007–2009 | 0.7015 | 0.8176 | +0.1161 |
| 2010–2012 | 0.6652 | 0.6709 | +0.0057 |
| 2013–2015 | 0.6926 | 0.6341 | **−0.0585** |

内层选模流程的时间 AUC 差值为 −0.0238，区间 [-0.1198, 0.0758]；新组合差值为 +0.0408，区间 [-0.0136, 0.1016]，且两种验证的 Brier 均变差。因此，不能挑表现最好的固定候选替代严格流程的结论。

系数诊断还发现 `informed consent`、`characteristics`、数字及实验室单位等非特异文本。这说明该表示可能利用注册模板与人群信息，尚不能解释为纯机制信号，也不能从系数推出因果关系。当前文本模型已扩展为 biology + trial context，应与 biology-only 基线分别汇报。

## 下一轮应解决的问题

1. **把有效的试验信息结构化。** 将全篇入组文本转为可审核的分子亚型、既往治疗/耐药、治疗线别、联合药物；保留肯定/否定和纳入/排除区别，避免关键词计数把“排除 EGFR mutation”当作阳性富集。
2. **优先审核标签。** 当前 472 条中有 361 条的来源是 `completed+results+no_positive_signal`，即没有阳性关键词就记为阴性。该来源不等于经核实的 endpoint 失败，不能用增加模型复杂度来解决这个定义问题。已有审核队列尚需 trial-specific 证据核实。
3. **建立独立确认条件。** 历史协议、数据库时间截断和最终未用于选模的数据仍未落实。先冻结提取规范和 outcome 定义，再验证，才有条件把“开发集增益”升级为可靠性能结论。

## 复现与验证

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run_trial_context_experiments.py \
  --output-dir mechanistic_engine_output/trial_context_replication_2026-09-21

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest \
  tests.test_trial_context_experiments tests.test_context_experiments \
  tests.test_performance_experiments -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  mechanistic_engine_output/trial_context_experiment_2026-09-21/verify_and_diagnose.py
```

输出目录必须是新目录，脚本拒绝覆盖已有实验。6 项新增测试、7 项相关回归测试全部通过。额外核对源文件 hash、完整训练 cohort/标签顺序、内外层 ID 隔离及 post-2015 排除，均通过。没有运行依赖 SHAP 的全项目回归。

本环境 SVC 的 `probability=True` 发出未来版本弃用警告，但本次全部拟合成功；迁移至未来 sklearn 版本时需要更新概率校准实现并重验。当前保存的协议是后来下载的记录，不能保证为试验开始时版本；排除结果字段也不能消除所有历史时间泄漏。

实现：[run_trial_context_experiments.py](../run_trial_context_experiments.py)；测试：[test_trial_context_experiments.py](../tests/test_trial_context_experiments.py)。

证据：[预设方案](../mechanistic_engine_output/trial_context_experiment_2026-09-21/experiment_plan.json)、[完整指标](../mechanistic_engine_output/trial_context_experiment_2026-09-21/validation_summary.csv)、[嵌套验收](../mechanistic_engine_output/trial_context_experiment_2026-09-21/development_gate.json)、[候选事后区间](../mechanistic_engine_output/trial_context_experiment_2026-09-21/posthoc_candidate_intervals.json)、[数据与切分核验](../mechanistic_engine_output/trial_context_experiment_2026-09-21/verification.json)。
