# Mechanistic Engine 最新进展

更新日期：2026-09-21。最新实验见下节；其余部分保留 9 月 14 日已完成工作的记录。

## 9 月 21 日第三轮：试验文本带来较大新组合增益，显著提升尚未确认

完成 12 个候选的嵌套开发验证（252 次拟合），加入疾病、干预药物和入组标准，保留全部 472 条训练记录。文本 + biology core 的 C=1 固定候选，新 target–disease 组合 AUC 从数值参照的 0.621 到 0.695（相对上轮类别模型 0.638 也有改善），时间平均 AUC 从 0.686 到 0.708。

但最近时间窗口 AUC 从 0.693 降至 0.634，严格内层选模流程也未通过预设门槛。候选的新组合区间虽高于零，属于看过外层结果后的未校正开发集分析，**不能称为已经实现可靠显著提升**。本轮未评分 166 条 post-2015 记录，也未替换主模型。6 项新增测试及 7 项相关回归测试通过，源文件 hash 和内外层切分隔离通过。

详情与可复现命令：[第三轮试验文本实验](trial_context_experiment_2026-09-21.md)。下一步重点是可审核的分子亚型/耐药/联合用药表示、真实 endpoint 标签和独立验证。

## 9 月 21 日第二轮：类别信息有小幅开发集增益

仅在训练期的 472 条记录上比较类别编码与固定 ensemble。增加 target、disease、modality 后，时间验证平均 AUC 从 0.686 到 0.693；新 target–disease 组合分组验证从 0.621 到 0.638。但 Brier 变差，新组合 Average Precision 也下降，因此并非全面改善。

本轮没有重新评分 166 条 post-2015 测试记录，没有替换主模型。3 项新增测试与分割隔离检查通过。详情见 [类别信息对照实验](context_experiment_2026-09-21.md)。这是值得继续验证的候选方向，不能宣称已改善独立测试性能。

## 9 月 21 日新增：模型性能对照实验

已在固定的 638 条旧快照上完成 16 个候选配置的训练内滚动验证，尝试标准化、正则化、较浅的树、刷新 PubMed 表示及精简 biology 特征。没有改动原始标签或替换主模型。

- 训练池 472 条、较晚年份测试集 166 条；用验证结果选定模型后再做最终对照。
- 最佳平均验证 AUC 从参考模型的 0.646 提高到 0.686；**最终测试 AUC 从 0.513 变为 0.497，未实现测试性能提升**。
- Brier 从 0.194 降至 0.178，但常数概率基线为 0.152，概率预测仍无优势。
- 新增 4 项测试通过，输入 hash 和验证/测试分离检查通过。
- 本轮完成的是固定旧快照上的探索性重训；统一新 cohort、人工标签审核、历史时间泄漏和独立验证仍未完成。

详情与可复现命令：[9 月 21 日 performance 实验](performance_experiment_2026-09-21.md)。下方“没有新 AUC/尚未重训”的陈述仅描述 9 月 14 日当时的状态，不代表本轮没有执行实验。

## 9 月 14 日记录

## 当前结论

**PubMed 查询可靠性修复和已映射组合的全量刷新已完成。数据版本统一、标签人工审核和新模型验证尚未完成。** 目前没有刷新后的新 AUC，不能把旧模型数字当成本次成果。

## 已完成的工作

### 1. PubMed：从单次测试推进到全量完成

- 修复“错误响应被误记为查询成功、文献数为 0”的问题；无效响应不缓存，失败保留为缺失并增加 `pubmed_data_missing`。
- 增加逐请求限速、429/服务器错误重试、缓存校验和带日期的缓存键。
- 近期文献窗口更新为 2023-01-01 至 2026-09-14；同时保留原始计数、检索式和查询状态。
- 保存的 5,000 条原始记录经当前代码筛选后为 4,303 条，其中 2,465 条有已知 target，对应 **335 个唯一 target–disease 组合，335/335 全部刷新成功**。
- 39 个组合返回真实零文献；未完成组合为 0。另有 1,838 条未知 target 的 trial，未进行 pair-level 查询，不能算作数据已补齐。
- 全部 335 个组合通过缓存复读和特征计算衔接检查；禁止联网的复读未发出请求，续跑未新增查询记录。

证据：[运行摘要](../mechanistic_engine_output/pubmed_refresh_2026-09-14/summary.json)、[验收记录](../mechanistic_engine_output/pubmed_refresh_2026-09-14/verification.json)、[完整结果](../mechanistic_engine_output/pubmed_refresh_2026-09-14/pubmed_pair_features.csv)。实现与复现见 [PubMed 说明](pubmed_refresh.md)。

### 2. 数据与标签：完成审计，尚未完成修正

- 两个旧特征表分别为 638 和 1,038 行，只共享 105 个 NCT ID；其中 **46 个 strict 标签不一致**。这证明版本冲突，不说明哪个标签正确。
- 共同试验中还存在 drug、target、disease 映射差异，尚未生成统一数据快照。
- 已准备 150 条盲审队列和独立标签对照文件，**尚未完成人工审核**；全部 46 条跨版本冲突另有清单。

证据：[跨版本对照](../mechanistic_engine_output/research_audit_2026-09-14/snapshot_comparison.csv)、[待审队列](../mechanistic_engine_output/research_audit_2026-09-14/label_review_queue.csv)、[冲突清单](../mechanistic_engine_output/research_audit_2026-09-14/cross_snapshot_label_conflicts.csv)。

### 3. 泛化与不确定性：完成诊断，尚未重训

- 638 行旧快照的 166 条时间测试记录中，已知且训练中未见的 drug 有 16 条、target 有 3 条、drug–disease pair 有 82 条；各划分另有 11 条未知实体记录。这些是 trial 行数，不是实体数量。
- 只有 3 条 unseen-target trial，不能据此稳定评估 unseen-target AUC；独立泛化实验尚未完成。
- 给旧快照的 419 条 deep predictions 补充了置信区间：BiologyOnlyMLP AUC 为 **0.530，trial bootstrap 95% CI 为 0.468–0.592**（2,000 次，seed=42）。这是对固定旧预测的统计检查，不包含训练、调参或标签不确定性，也不是新训练结果。

证据：[实体重叠](../mechanistic_engine_output/research_audit_2026-09-14/entity_overlap.csv)、[AUC 区间](../mechanistic_engine_output/research_audit_2026-09-14/deep_auc_intervals.csv)、[计算记录](research_snapshot_audit_2026-09-14.ipynb)。

## 问题状态表

| 问题 | 当前状态 | 还需要完成什么 |
|---|---|---|
| PubMed 错误处理、缓存、限流与全量刷新 | 已完成本次范围 | 新快照采用新结果后重算相关特征和综合评分 |
| 未知 target 的 PubMed coverage | 未解决 | 改善实体映射并单独报告不可映射记录 |
| 不同数据版本、标签冲突 | 已定位，未修正 | 固定一个可追溯的 cohort 与 label 版本 |
| 标签可靠性 | 审核清单已准备 | 逐条核对 trial-specific endpoint 证据 |
| 时间信息泄漏 | 未解决 | 按预测时点限制证据，说明历史索引不可恢复的局限 |
| Unseen drug/target 泛化 | 重叠统计已完成 | 在统一数据上运行验证，报告样本量和不确定性 |
| 预测置信区间 | 部分完成 | 统一数据重训后补齐各 baseline 指标 |
| cBioPortal 数据问题 | 未解决 | 核实覆盖率与缺失原因，决定修复或从主要模型排除 |
| 旧代码修复后的全流程验证 | 未完成 | 完整回归通过，并生成一致的新输出 |

PubMed 的日期上限目前统一为 2026-09-14，**未按各 trial 开始日期截断**；publication date 限制也不等于当年的 PubMed 索引快照。因此，本次 API 修复不能作为时间泄漏已经解决的依据。

## 验证状态

- 保存的执行记录显示：5 项审计测试通过，另有 10 项 PubMed/续跑测试通过。不能合称为“完整测试集通过”。
- PubMed 实际全量运行中的 HTTP 429 与无效响应重试后恢复；335 个结果的子集计数均不超过总计数。
- 完整旧回归测试在加载 SHAP/Numba 依赖时长时间停留后中断，未确认通过；此前也出现过 Matplotlib 初始化耗时。
- Notebook 的 5 个代码单元已用项目 Python 顺序执行并保存输出，但没有 Jupyter kernel/查看器验证。
- 原始数据、旧特征表和已有模型报告未覆盖；新 PubMed 结果独立保存。

## 下一步与讨论重点

1. **先统一数据和 outcome 定义。** 确定 cohort、标签证据要求，保存输入 hash、代码版本及 split ID。
2. **先审核 10–20 条试验对齐规则，再扩大到 150 条。** 单独记录无法判定的 trial，不把没有阳性关键词直接当作疗效失败。
3. **在统一快照中接入刷新结果，重算特征与 composite，再重训 baseline。** 比较必须基于相同样本与切分，不将新 PubMed 列与旧综合评分混用。
4. **安排 temporal 与 unseen drug–disease pair 验证。** Unseen-target 因样本少先作为探索性分析，同时处理历史证据可用性问题。

上述是待讨论的建议，不是已获同意的安排。9 月 7–13 日会议原始记录尚未获得；详细议程见 [9 月 15 日讨论大纲](meeting_outline_2026-09-15.md)。

## 可直接汇报的英文版本

> I completed the PubMed reliability fixes and refreshed all 335 mapped target–disease pairs from the saved trial corpus. All queries completed successfully, and cache-only replay and resume checks passed without additional network requests. I also audited the existing datasets, identified 46 label disagreements among 105 shared trials, and prepared 150 trials for manual review. The remaining priorities are to agree on one cohort and outcome definition, address historical evidence availability, and retrain the baselines using a consistent feature snapshot. I have not yet produced new model performance results.
