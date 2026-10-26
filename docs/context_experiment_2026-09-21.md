# 第二轮：类别信息能否改善模型表现？

日期：2026-09-21。结论：**有小幅训练期验证 AUC 增益，但不是全面改善；尚未证明较晚年份测试性能提高。**

## 方法

- 只使用旧 classification 快照中 2015 年及以前的 472 条记录（92 阳性）。保留原始 strict 标签；166 条 post-2015 记录不参与拟合、候选比较或评分。
- 数值参考为上一轮的 22 列 biology core + 缺失值填补 + 标准化 + Logistic Regression（C=1）。
- 在固定数值参考上加入训练折内拟合的 one-hot 类别：target/disease/modality；再比较增加 drug identity 和 drug–disease 交互类别。未见类别采用 ignore，不在验证数据上重拟合编码器。
- 另外测试数值参考与 drug-context 模型的固定 50:50 概率平均，不调混合权重。
- 时间验证沿用三个训练内窗口：2007–2009、2010–2012、2013–2015；每折训练记录早于验证窗口。
- 新增三折 StratifiedGroupKFold，以 target–disease pair 分组，同一组合不跨 train/validation。它不保证单独的 drug/target 未见，也不保证时间外推：分组验证会混合训练期内的年份。
- 两类验证均为开发阶段检查；时间验证窗口此前已被使用，不能视作全新确认性验证。

## 结果

| 方法 | 时间验证平均 AUC | 新 target–disease 组合平均 AUC | 时间验证 Brier | 新组合 Brier |
|---|---:|---:|---:|---:|
| 数值参考 | 0.686 | 0.621 | **0.147** | **0.144** |
| + target / disease / modality | **0.693** | 0.638 | 0.159 | 0.152 |
| + drug identity | 0.676 | 0.640 | 0.162 | 0.152 |
| + drug–disease 类别 | 0.670 | 0.638 | 0.163 | 0.150 |
| 固定 50:50 ensemble | 0.685 | **0.640** | 0.153 | 0.146 |

均为三折指标的算术平均；Brier 越低越好。两种验证样本和分割方式不同，不应直接比较它们的绝对 AUC 来量化泛化下降。

target/disease/modality 方案相对参考：

- 时间验证 AUC 增加 0.0067，三折差值为 -0.0096、+0.0053、+0.0244；并非每折都改善。
- 新组合验证 AUC 增加 0.0174，三折差值为 +0.0058、+0.0005、+0.0458。
- 时间验证 Average Precision 从 0.439 到 0.450，但新组合验证从 0.409 降至 0.373。
- 两种验证的 Brier 都变差，因此不能宣称概率预测改善。三折差值也不足以证明统计显著性；没有做确认性显著性检验。

没有重新计算 post-2015 测试 AUC，没有根据它选择这些候选，没有替换主模型。上一轮较晚年份测试表现不佳的结论仍然有效，本轮不能覆盖或修正那个结果。

## 下一步建议

1. **保留 target/disease/modality 作为候选表示。** 先验证生物学上下文的增量；这轮不支持为了时间泛化而继续加入更多 drug identity 类别。
2. **单独处理概率可靠性。** 若目标是输出成功概率，而不仅是排序，需要训练内校准或更强正则化，并同时比较 Brier 和阳性率常数基线；不能只挑 AUC。
3. **把主要精力投入标签审核与 trial 生物学信息。** 当前标签部分依赖获批/文本规则，类别效果也可能反映标签构造，不能解释成已学到机制。分子亚型、biomarker selection、组合方案仍是待验证方向。
4. **确认性评价需要预先冻结方案与独立数据。** 当前小幅开发集增益只能用于确定下一步优先级，不能把旧测试集重复用于挑选高分结果。

## 产物与验证

- 新入口：`run_context_experiments.py`
- 新测试：`tests/test_context_experiments.py`，3 项通过：较晚记录的标签不影响训练样本、group 完全隔离、未知类别不会触发重新拟合。
- 输出目录：`mechanistic_engine_output/context_experiment_2026-09-21/`
- `experiment_plan.json`：候选配置、样本范围、seed、代码/输入 hash 和限制。
- `validation_summary.csv`、`validation_folds.csv`：完整比较及成对折间差值。
- `validation_predictions.csv`、`split_ids.csv`：逐行验证预测和分割清单。
- 已检查所有 split 年份 ≤2015，输入/代码 hash 未改变，未修改旧数据、标签或模型报告。本轮未重跑完整旧回归测试。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run_context_experiments.py \
  --output-dir mechanistic_engine_output/context_experiment_repeat
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests \
  -p 'test_context_experiments.py' -v
```

输出目录必须为新目录；不需要新依赖或联网。
