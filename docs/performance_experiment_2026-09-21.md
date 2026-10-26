# Model performance：2026-09-21 实验与下一步

## 结论

完成了一轮固定数据、固定候选范围的模型对照。**验证集分数提高，但较晚年份测试集未提高，不能宣称模型性能已经改善，也没有替换现有主模型。**

本轮最有价值的结果是：单纯调整模型与 PubMed 表示没有解决时间外推问题，后续应优先核对 outcome labels，并提高 biology 特征对不同 trial 的区分能力。

## 实验范围与可比性

- 唯一输入 cohort：旧 `classification_matrix.csv`，638 条，保留原始 strict 标签；没有与 1,038 行的另一份数据混合。
- 训练池：开始年份 ≤2015，472 条；测试集：>2015，166 条，31 阳性、135 阴性。没有因 label 或年份缺失排除行。
- 三个训练内滚动验证窗口：2007–2009、2010–2012、2013–2015；每折只用该窗口之前的 trial 训练。
- 共 16 个预先定义候选，以平均验证 AUC 选择，平局时优先较小 Brier，再按名称确定。预处理只在每折训练样本上拟合。
- 选择结果先写入文件，然后仅评估参考模型和选中模型的最终测试预测。没有看测试结果后继续选择别的候选。
- 测试集此前已在项目中被查看过，因此仍属于探索性的回顾性实验，不是全新外部验证。
- 保留旧数据与报告，所有新输出在 `mechanistic_engine_output/performance_experiment_2026-09-21/`。

## 实际尝试了什么

1. **标准化的正则化 Logistic Regression**：训练内缺失值填补、StandardScaler、C=0.1/1.0；现有主训练代码的 LR 实际未做标准化。
2. **更简单的树模型**：深度 1/2 的 Gradient Boosting 与深度 4 的 Random Forest。
3. **三种特征表示**：旧 composite 集合；刷新后的 PubMed + 原始计数 log1p + 临床试验文献占比；22 列精简 biology core。
4. **刷新特征一致性**：PubMed 匹配 593/638 行；45 行未匹配，保留缺失标志、不删行。更新 MCS、EMS、BIOLOGY_SCORE，保留其他综合分数。临床试验文献占比使用原始计数，不再用两个不同缩放比例的特征相除。
5. **参考模型**：旧 GradBoost 参数，使用同一 638 行 cohort 与同一切分重新训练；不是之前其他 cohort 的 0.566 或旧文档的 0.67。

刷新表示作为一个组合改动测试，不能从本轮归因到某一个 PubMed 特征。精简 biology core 减少了文献/获批相关输入，但不代表消除了时间泄漏。

## 结果

| 模型 | 三折平均验证 AUC | 最终测试 AUC | 测试 Average Precision | 测试 Brier（越低越好） |
|---|---:|---:|---:|---:|
| 参考：旧特征 + GradBoost | 0.646 | 0.513 | 0.195 | 0.194 |
| 验证选中：biology core + 标准化 LR，C=1 | 0.686 | 0.497 | 0.189 | 0.178 |
| 常数预测：训练集阳性率 | — | 0.500 | 0.187 | 0.152 |

刷新 PubMed 表示中，验证表现最高的是标准化 LR（C=1），平均验证 AUC 为 0.656；它没有赢得预先规定的模型选择，因此没有为挑选更高分数而额外查看它的测试 AUC。

- 验证 AUC 提高约 0.040，但测试 AUC 差值为 -0.016。
- 2,000 次成对 trial bootstrap，测试 AUC 差值的 95% percentile CI 为 **[-0.121, 0.086]**；按 canonical drug 聚类重采样为 **[-0.118, 0.082]**。
- 选中模型 Brier 比参考树模型低，但仍不如常数概率基线，不能据此声称概率预测已经可靠。
- 区间条件于固定模型预测，不涵盖模型选择、训练或标签不确定性。Drug 聚类将相同 canonical drug（含相同 unmapped 占位符）视为一组，不能完整代表多药方案依赖。

## 为什么需要改变改进重点

以下是测试结束后的诊断，不用于继续在这个测试集上选模型。

**标签证据较弱：** 训练集 361/472 条、测试集 118/166 条来自 `completed+results+no_positive_signal` 规则，全部被标为阴性。“没有抓到阳性文字”未必等于“试验疗效阴性”；这些记录需要证据审核，不能为了提高分数直接翻转或删除标签。

**不同 trial 缺乏可区分的输入：** 旧特征的 166 条测试记录只有 86 种完全不同的数值向量；74 条记录属于“同一特征向量内同时有阳性和阴性标签”的组。精简 core 的 166 条测试记录只剩 49 种向量，119 条落在混合标签组。相同输入的确定性模型无法区分组内 trial，但这不证明这些标签必然错误，也不能由此断言 biology 没有信号。

## 提升 performance 的优先顺序

1. **建立可审查的 outcome 标准。** 先审核 10–20 条弱阴性及跨版本冲突记录，核对预先指定终点、结果方向和 trial-specific 来源，再推广到现有 150 条队列。暂不自动改变 ground truth。
2. **补充能区分 trial 的生物学上下文。** 优先评估分子亚型、biomarker selection、组合用药等当前单一 primary target/disease 表示可能遗漏的信息；这些是待验证方向，不承诺加上就会提升。
3. **冻结统一数据快照后重新验证。** 明确标签证据规则、历史证据时间边界、相同 split IDs，再比较简单 baseline 与 biology 增量。若加入 trial metadata，应单独标为扩展任务，不能再称纯 biology-only。
4. **保持验证选择与测试评估分离。** 本轮独立入口已实现这一点；旧 `modeling.py` 主入口仍按测试 AUC 选最佳配置，不应把这种结果当作无偏性能估计。该旧入口本次未改写。

不建议现在根据这 166 条测试记录继续调参找高 AUC；下一轮方法选择需要新的冻结评估设计，最好最终有后续独立 cohort。

## 文件与验证

- 新实验入口：`run_performance_experiments.py`
- 新测试：`tests/test_performance_experiments.py`，4 项全部通过。
- 运行配置与输入/代码 hash：输出目录的 `experiment_plan.json`
- 全部验证比较：`validation_comparison.csv`；逐折结果：`validation_folds.csv`
- 冻结的模型选择：`selected_model.json`
- 最终测试结果与逐 trial 预测：`holdout_comparison.csv`、`holdout_predictions.csv`
- 成对区间：`paired_bootstrap.json`
- 后验诊断与复现脚本：`label_source_diagnostics.csv`、`feature_resolution_diagnostics.csv`、`diagnose.py`

验证了所有输入 hash 未变、638 条 cohort 保留、验证 fold 与测试集无交叉、每折训练年份早于验证年份。没有改动生产模型、原始标签、原始数据或旧模型报告，也没有新增依赖。此次没有重跑依赖 SHAP/Numba 的完整旧测试集。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run_performance_experiments.py \
  --output-dir mechanistic_engine_output/performance_experiment_repeat
.venv/bin/python -m unittest discover -s tests -p 'test_performance_experiments.py' -v
```

输出目录必须是新的。复跑是结果复现，不应把同一测试集当成未见数据再次调参。
