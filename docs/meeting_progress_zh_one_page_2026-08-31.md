# Mechanistic Engine：Meeting 前中文速读版

## 这个项目在做什么？

我们想研究：能不能使用公开的药物、target、disease 和 biological mechanism 数据，判断一个 Phase 2 肿瘤临床试验更可能成功还是失败。

目前最合适的研究问题是：

> **公开的 mechanistic biology evidence 在什么情况下有预测作用，在什么情况下没有？**

## 这次我完成了什么？

我重新运行了当前 pipeline，并修复了四个问题：

1. Raw、composite 和 hybrid feature sets 以前存在重复，现在已经真正分开。
2. Two-stage model 因为重复列不能运行，现在已经修好。
3. OS、PFS、ORR 等 endpoint 以前经常被错误分成 `other`，现在可以正确识别。
4. PubMed 查询容易遇到 429 rate-limit，现在加入了等待和自动重试。

另外，我加入了 5 个 regression tests，目前全部通过。

## 数据和模型结果

- ClinicalTrials.gov 原始记录：5,000
- 筛选后的 Phase 2 oncology trials：4,274
- 有可用 permissive label 的试验：832
- 训练集：617（2015 年及以前）
- 测试集：215（2015 年以后）

主要结果：

| 分析 | 测试集 AUC |
|---|---:|
| Strict label | 0.685 |
| Permissive label | 0.667 |
| PFS subgroup | 0.759，但 test n=35 |
| Response subgroup | 0.443 |
| Two-stage random forest | 0.660 |
| 最好 deep model | 约 0.53 |

## 这些结果说明什么？

1. 模型确实学到了一些 signal，但整体预测能力目前只能算中等。
2. Composite biology features 目前表现最好。
3. Two-stage 和 deep model 都没有带来提升，所以现在没必要继续堆更复杂的模型。
4. PFS 结果看起来不错，但样本只有 35 个，暂时不能下结论。
5. Response endpoint 表现很差，说明不同 endpoint 可能需要分开研究。

## 当前最重要的问题

- `balanced` label 在代码里实际上等于 `strict`，所以不能把它们当成两个不同结果。
- 训练集和测试集可能出现相同 drug 或 target，模型可能记住了已有药物。
- 现在使用的是最新公共数据库，其中可能包含 trial 结束以后才出现的信息。
- cBioPortal 当前没有获得有效数据，不能把其中的 0 当成真正的 biological absence。
- PubMed 只做了单次实时测试，还没有完成全量刷新。

## Paper 可以怎么设计？

论文不应该重点强调“我们做了一个更复杂的模型”，而应该研究：

1. Biology features 是否比普通 trial information 更有用？
2. 模型在新 drug 和新 target 上还能不能工作？
3. 为什么 PFS 和 response endpoint 的表现不同？
4. 数据缺失和数据库 coverage 会怎样影响结果？
5. 哪些 mechanistic evidence 真正有稳定的预测价值？

暂定题目：

> **When Does Public Mechanistic Evidence Generalize to Phase 2 Oncology Trial Outcomes?**

## 下一步最应该做什么？

1. 决定 strict 还是 permissive 作为主要 label。
2. 删除或重新定义 balanced label。
3. 人工检查 150–200 个 trial labels。
4. 增加 unseen-drug 和 unseen-target 测试。
5. 给 AUC 加 bootstrap confidence intervals。
6. 完整刷新 PubMed features。
7. 修复或暂时移除 cBioPortal。

## 明天可以怎么汇报？

> I reran and audited the current pipeline. I fixed feature duplication, endpoint classification, the two-stage model input, and PubMed rate-limit handling, and added five regression tests. The corrected main performance is stable, with an AUC of about 0.67 under the permissive label and 0.685 under the strict label. The composite biology features remain the strongest representation, while the two-stage and deep models do not improve the baseline. Endpoint analysis shows a potentially interesting but small PFS result and poor performance for response endpoints. I therefore propose focusing the paper on when mechanistic evidence generalizes, rather than on increasing model complexity.

## Meeting 中最值得问的三个问题

1. Strict 还是 permissive 应该作为主要 outcome label？
2. Paper 是否应该重点研究 unseen drug、unseen target 和 endpoint differences？
3. 第一投稿目标是否定为 JCO Clinical Cancer Informatics？
