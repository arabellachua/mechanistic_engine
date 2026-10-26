# 9 月 15 日讨论大纲：先统一数据，再验证 mechanistic signal

准备日期：2026-09-14。建议讨论时间：20–25 分钟。

当前完成状态以 [最新进展](latest_progress.md) 为准；本大纲保留各项讨论的详细依据。

**来源边界：尚未找到 9 月 7–13 日会议原始记录。以下是依据 8 月 31 日进展文档与本次实际审计准备的暂定大纲，不代表她上周已经同意的安排。拿到记录后应对齐她的具体要求。**

## 追加完成：PubMed 修复与全量刷新（2026-09-14）

在前述审计之后，已进一步完成 PubMed 工作，可以作为新的实际 progress 汇报：

> I fixed a bug that treated PubMed error responses as successful zero-publication results. I added validated caching, retries, explicit missingness, and a dated recent-publication window. I then refreshed all 335 mapped target–disease pairs from the saved raw trial corpus. All 335 completed successfully, including recovery from an actual HTTP 429 response. Cache-only replay and resume checks passed without additional network requests.

- 5,000 条原始记录经当前代码保留 4,303 条，2,465 条有已知 target，形成 335 个组合；1,838 条未知 target 的记录仍不具备 pair-level PubMed 数据。
- 335/335 完成，39 个真实零文献组合，未完成 0。
- 10 项 PubMed/续跑测试通过；完整旧回归测试受 SHAP/Numba 依赖加载停留影响，未确认通过。
- 结果、代码和验证说明见 `docs/pubmed_refresh.md`；主结果为 `mechanistic_engine_output/pubmed_refresh_2026-09-14/pubmed_pair_features.csv`。
- **尚未重训模型，也未按各 trial 开始日期截断文献。** 历史时间泄漏、统一模型数据快照和未知 target 映射仍需处理。下文旧模型统计仍是独立快照审计，不能当成本次刷新的模型表现。

## 1. 开场：这次完成了什么（2 分钟）

可以直接说：

> 我这次完成了 PubMed 的错误处理、缓存和限流修复，并跑完全部 335 个已映射 target–disease 组合，缓存复用和续跑检查也通过了。另外完成了数据版本、标签一致性和 drug/target 重叠审计，给已有 deep predictions 补上置信区间，并准备了 150 条待人工审核清单。目前还没有新的模型训练结果，下一步需要统一数据快照和 outcome 定义，再接入刷新的特征进行验证。

英文版本：

> I completed the PubMed reliability fixes and refreshed all 335 mapped target–disease pairs. Cache-only replay and resume checks passed without additional network requests. I also audited the saved datasets, found 46 label disagreements among 105 shared trials, added confidence intervals to existing deep-model predictions, and prepared 150 trials for manual review. I have not yet retrained the models. Before further comparisons, I would like to agree on one dataset snapshot and outcome definition.

## 2. 第一项发现：现有结果不是同一个 cohort（5 分钟）

| 检查项 | classification_matrix | full_feature_matrix |
|---|---:|---:|
| 试验数 | 638 | 1,038 |
| 试验开始年份范围 | 1994–2024 | 1993–2026 |
| ≤2015 的有效年份试验 | 472 | 611 |
| >2015 的有效年份试验 | 166 | 419 |
| 缺失年份，无法进入时间切分 | 0 | 8 |
| strict 与 balanced 不一致 | 0 | 51 |
| balanced 与 permissive 不一致 | 0 | 0 |

- 两表只有 105 个相同 NCT ID；其中 46 个 strict 标签不一致（43.8%）。这证明标签不一致，**不证明其中哪一份是正确答案**。
- 相同 NCT ID 中，17 个 canonical drug、15 个 primary target、37 个 disease 字段不同，说明差异不止来自 label。
- classification CSV 中 `drug_is_mapped` 的原始表头重复 4 次。当前代码已有相关修复，但旧文件仍保留旧结构；代码修好不等于历史输出已刷新。
- 独立 deep runner 读取 `full_feature_matrix`；419 条保存的 deep predictions 与该表的时间测试集 ID、标签、drug、target、disease、年份全部匹配。但这不足以完整追溯训练过程。
- 当前 `model_comparison.csv` 最大 AUC 为 permissive/balanced 0.566、strict 0.528；8 月 31 日文档写的是约 0.667/0.685。**不能把这解释成模型退步，也不能混用两个版本的数字。** 本次没有重训来复现旧文档结果。

讨论句：

> Can we agree on one versioned cohort and one label definition before making further model comparisons?

建议决定：下一轮从明确输入生成新的独立运行目录，保存输入 hash、代码版本、label 版本和 train/test ID。保留所有旧输出。

## 3. 第二项发现：strict 仍然需要人工核对（4 分钟）

638 行快照的 strict 标签来源包括：

- 479/638（75.1%）：`completed+results+no_positive_signal`，即“没有识别到阳性信号”被当作阴性。
- 29/638（4.5%）：`completed+positive_text+approved`，包含药物适应症获批信息。

因此，不能仅凭 `strict` 这个名称就把它描述为人工确认的 primary endpoint outcome。当前 `labels.py` 仍包含这些规则，且 `balanced = strict`。

已完成：150 条分层抽样待审队列，附 ClinicalTrials.gov 链接；原有 heuristic labels 放在单独的 key 文件，便于盲审。**尚未完成任何人工 adjudication。** 另输出了全部 46 条跨版本 label 冲突供优先核查。

讨论句：

> Should the primary outcome require trial-specific evidence that the prespecified efficacy endpoint was met, with approval-based or missing-positive-signal labels reserved for sensitivity analyses?

建议决定：先共同审核 10–20 条建立规则，再扩大到 150 条；记录 endpoint、效应方向、证据来源、日期和不确定性。不能仅凭 p<0.05 判定成功，也不能将 trial completion 等同于疗效成功。是否需要第二位 reviewer 和分歧仲裁由她确定。

## 4. 第三项发现：unseen-target 验证受样本量限制（5 分钟）

以下只描述 **638 行快照的 166 条时间测试记录**；不是新训练实验。

| 测试记录按实体划分 | 训练中见过 | 训练中未见、身份已知 | 身份未知 |
|---|---:|---:|---:|
| canonical drug | 139（83.7%） | 16：5 阳性 / 11 阴性 | 11 |
| primary target | 152（91.6%） | 3：2 阳性 / 1 阴性 | 11 |
| drug–disease pair | 73（44.0%） | 82：18 阳性 / 64 阴性 | 11 |

这是 trial 行数，**不是不同 drug 或 target 的数量**。只按 canonical drug/primary target 判断，不覆盖多药组合的所有成分或所有 targets。

- 未知 drug/target 不应归入“新 drug/target”，本次已单独计数。
- 只有 3 条 unseen-target 试验，不能支撑稳定的 subgroup AUC。
- 82 条 unseen drug–disease pair 记录更适合先探索，但实体映射有跨版本差异，必须先统一数据。

讨论句：

> Given the very small number of unseen-target trials, should we prioritize temporal validation and unseen drug–disease pairs, and treat target-held-out evaluation as exploratory?

建议决定：固定 cohort 后先报告 temporal 与 seen/unseen subgroup；若增加 target-grouped CV，明确它回答的是跨 target 泛化，不能替代时间外推评估。

## 5. Deep model 的新统计检查（3 分钟）

在旧 full-feature 快照的 419 条 permissive test predictions 上，用 2,000 次 trial bootstrap、seed=42 得到：

| 已保存模型 | AUC | 95% percentile CI |
|---|---:|---:|
| BiologyOnlyMLP | 0.530 | 0.468–0.592 |
| EntityOnlyMLP | 0.478 | 0.421–0.538 |
| DeepMLP | 0.478 | 0.419–0.536 |
| TrialTransformer | 0.439 | 0.384–0.499 |

补充 canonical-drug cluster bootstrap 后，BiologyOnlyMLP 区间为 0.458–0.583。未知 drug 合并为一个 cluster，这只是敏感性检查。

这些区间只覆盖固定预测上的重采样变异，不包含训练、调参、模型选择或标签不确定性。不能当成新模型结果或外部验证。这里最高的 deep AUC 区间跨过 0.5，暂时不足以支持在这批数据上增加模型复杂度的优先级；也不能据此证明 biology 没有价值。

## 6. 希望带走的三个决定（3 分钟）

1. **数据：** 哪个输入、cohort 筛选和 label 定义作为新实验的唯一基准？
2. **验证：** 是否优先 temporal + unseen drug–disease pair，并将 unseen-target 作为探索性分析？
3. **下一周交付：** 是否先完成 10–20 条共同审核、冻结规则，再完成 150 条审核和同一 cohort 上的 baseline？

建议下周验收标准（待讨论，不是已完成事项）：

- 每个模型结果都能追溯到相同输入 hash、label 版本和 split IDs。
- label audit 有逐条证据，不只给成功/失败二值；不确定 trial 单列。
- 比较同一 cohort 上的简单 baseline 与 biology features，并报告 AUC、PR-AUC、Brier 和不确定性。
- 历史数据库时间问题仍未解决；现阶段明确写 retrospective evaluation。
- 对数据库 missingness 先核实来源；638 行快照中 cBioPortal missing flag 为 638/638，不能解释成“这些试验没有 mutation”。

## 本次产物和复现

- 脚本：`audit_research_snapshot.py`
- 测试：`tests/test_research_snapshot_audit.py`
- 可检查的计算记录：`docs/research_snapshot_audit_2026-09-14.ipynb`
- 审计输出：`mechanistic_engine_output/research_audit_2026-09-14/`
- 数据输入和 SHA-256：该目录的 `manifest.json`
- 150 条盲审队列：`label_review_queue.csv`；对照标签：`label_review_key.csv`
- 46 条跨版本冲突：`cross_snapshot_label_conflicts.csv`
- 重叠统计：`entity_overlap.csv`；置信区间：`deep_auc_intervals.csv`

从仓库根目录运行（输出目录必须是新的，脚本拒绝覆盖）：

```bash
.venv/bin/python audit_research_snapshot.py \
  --output-dir mechanistic_engine_output/research_audit_repeat
.venv/bin/python -m unittest discover -s tests -p 'test_research_snapshot_audit.py' -v
```

快照审计阶段没有调用外部 API；之后的 PubMed 刷新已实际调用 NCBI API，并单独保存新结果。现有 label、模型、数据及原有报告未覆盖，也未完成新模型训练。上述工作不是对上周会议要求已全部落实的声明。

### 验证状态

- 新增的 5 项审计测试全部通过：未知实体与 unseen 的区分、无效标签/缺失年份计数、bootstrap 可重复性、单一类别处理、cluster 重采样和非法概率检查。
- 3 个审计源文件的 SHA-256 在结束时仍一致；原始输入未改变。
- Companion notebook 的 5 个代码单元已用项目 Python 按顺序执行并保存 stdout，重新核对标签冲突和 BiologyOnlyMLP 区间，与已有模型报告的 AUC 一致。环境没有 Jupyter/nbformat/nbclient，因此未做 Jupyter kernel 执行或 notebook viewer 视觉检查。
- 尝试完整 regression suite 时，旧测试模块在 Matplotlib 字体/样式初始化长时间停留，手动中断后显示 import error / KeyboardInterrupt；**不能宣称完整测试集通过**。可重试：

```bash
XDG_CACHE_HOME=/private/tmp/research-audit-cache MPLCONFIGDIR=/private/tmp/mpl-cache \
  .venv/bin/python -m unittest discover -s tests -v
```
