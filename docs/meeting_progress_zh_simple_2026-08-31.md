# Mechanistic Engine 项目进展（中文简版）

## 一句话总结

我重新运行并检查了目前的模型。修复几个代码和数据处理问题后，模型的主要结果基本稳定：测试集 AUC 大约是 **0.67–0.69**。目前更值得研究的方向不是继续堆复杂模型，而是弄清楚：**这些公开的生物学证据在什么情况下能预测 Phase 2 肿瘤临床试验结果，在什么情况下不能。**

## 1. 这次我具体做了什么

### 重新运行当前 pipeline

这次运行使用了和之前一致的设置：

- 从 ClinicalTrials.gov 获取 5,000 条记录；
- 筛选出 4,274 个 Phase 2 肿瘤试验；
- 其中 832 个试验有可以用于 permissive label 的确定结果；
- 按试验开始年份进行时间切分；
- 2015 年及以前作为训练集，共 617 个试验；
- 2015 年以后作为测试集，共 215 个试验。

这样做的目的是尽量用较早的试验训练模型，再检查模型能否推广到较晚的试验。

### 修复了四个主要问题

#### 1. 特征组有重复

原来的 raw、composite 和 hybrid 特征组没有完全分开，因此不同特征组之间的比较不够清楚。

修复后：

- Composite：33 个特征；
- Raw：69 个特征；
- Hybrid：76 个特征。

现在可以更公平地比较：原始数据库特征、人工组合的生物学评分，以及两者结合以后哪个效果更好。

#### 2. Two-stage model 无法正常运行

数据中出现了两个同名的 `drug_is_mapped` 列，导致第二阶段模型报错。

修复重复列以后，two-stage model 已经可以完整运行。

#### 3. Endpoint 分类不准确

原来的程序只读取了一小部分 endpoint 文本，所以很多 OS、PFS、ORR endpoint 都被错误地分到了 `other`。

现在程序会一起读取：

- endpoint 名称；
- endpoint 描述；
- endpoint 时间范围。

同时可以识别：

- OS：Overall Survival；
- PFS：Progression-Free Survival；
- ORR：Objective Response Rate；
- response、clinical benefit 和 biomarker endpoint。

#### 4. PubMed 查询容易遇到 429 错误

原来的程序连续请求 NCBI/PubMed 时速度较快，可能触发 rate limit。

现在加入了：

- 每次请求之间的等待；
- 遇到 429 后自动重试；
- 根据 `Retry-After` 等待；
- 服务器错误时逐步延长等待时间。

我已经用一个 EGFR–NSCLC 查询做了实时测试，可以正常返回结果。不过完整 PubMed 数据还需要重新跑一次。

## 2. 当前模型结果

### 整体结果

| Label | 最好模型 | 最好特征组 | 测试集 AUC |
|---|---|---|---:|
| Strict | LightGBM | Composite | 0.685 |
| Permissive | Gradient Boosting | Composite | 0.667 |

Permissive label 的五折交叉验证结果是：

> **AUC = 0.677 ± 0.065**

简单理解：

- 模型确实学到了一些有用信息；
- 但是预测能力目前只能算中等；
- 修复代码以后，整体结果没有突然上升或下降；
- 说明原来的主要结果大致稳定，不完全是代码错误造成的。

### 一个需要注意的 label 问题

目前代码中的 balanced label 实际上和 strict label 完全一样，因此不能把它们当成两个独立结果。

下一步应该二选一：

1. 重新设计一个真正不同的 balanced label；或者
2. 删除 balanced，只报告 strict 和 permissive。

我目前更倾向于：

- strict 作为主要分析；
- permissive 作为敏感性分析。

因为 permissive 样本更多，但其中包含“药物后来在该适应症获批”这一类较宽松的判断。

## 3. Endpoint 分析发现了什么

修复 endpoint 分类以后，不同 endpoint 的结果差别很大。

| Endpoint | 可用标签数量 | 测试集数量 | 测试集 AUC |
|---|---:|---:|---:|
| PFS | 120 | 35 | 0.759 |
| Response | 236 | 84 | 0.443 |
| Overall Survival | 27 | 太少 | 暂不计算 |
| Other/缺失 | 441 | 87 | 0.685 |

### 怎么理解

PFS 的 AUC 0.759 看起来比较好，但是测试集只有 35 个试验，所以这个结果还不稳定，暂时只能说是一个值得继续检查的信号。

Response endpoint 的 AUC 是 0.443，说明目前这些生物学特征对 response 类试验的推广效果很差。

这个差异可能说明：

- 不同 endpoint 对应的生物学机制不同；
- 当前 label 对不同 endpoint 的可靠性不同；
- 某些 mechanistic features 更适合 PFS，但不适合 ORR/response；
- 或者 PFS 的好结果只是小样本波动。

所以接下来需要加置信区间和更严格的数据切分，不能现在就下结论说模型特别适合 PFS。

## 4. 更复杂的模型有没有帮助

### Two-stage model

| 模型 | 测试集 AUC |
|---|---:|
| 第一阶段 biology model | 0.663 |
| 第二阶段 Logistic Regression | 0.598 |
| 第二阶段 Random Forest | 0.660 |

第二阶段没有超过第一阶段，所以目前没有证据表明 two-stage design 更好。

### Deep model

目前最好的 deep model AUC 大约是 0.53，也没有超过 tree-based baseline。

这说明当前最主要的问题可能不是模型不够复杂，而是：

- label 还有噪声；
- 样本量偏小；
- 不同 endpoint 混在一起；
- drug 和 target 可能在训练集、测试集中重复；
- 外部数据库 coverage 不完整；
- 部分数据库信息可能是在 trial 结束以后才产生的。

因此现在不建议继续优先增加更复杂的神经网络。

## 5. 我认为论文可以怎么写

### 核心问题

论文可以研究：

> **公开的 mechanistic biology evidence 在什么情况下可以推广到未来的 Phase 2 肿瘤试验结果？**

重点不是证明模型可以准确预测所有试验，而是系统研究：

1. biology features 是否比普通 trial metadata 更有用；
2. composite biology score 是否比大量 raw features 更稳定；
3. 不同 endpoint 的模型表现为什么不同；
4. 新药物和新 target 上是否还能保持效果；
5. 数据缺失和数据库 coverage 会不会影响结果。

### 可以分成三个研究目标

#### Aim 1：建立可靠的 trial outcome 数据集

- 明确定义什么是成功和失败；
- 保留每个 label 的判断来源；
- 人工检查一部分 labels；
- 报告多少试验因为结果不明确而被排除。

#### Aim 2：测试 biology 是否真的可以 generalize

- 比较 metadata、raw biology、composite biology 和 hybrid；
- 使用时间切分；
- 增加 unseen-drug 和 unseen-target 测试；
- 同时报告 AUC、PR-AUC、calibration 和置信区间。

#### Aim 3：研究模型什么时候有效、什么时候失败

- 比较 PFS、response 和 OS；
- 比较不同 cancer type 和 drug modality；
- 检查 missing data；
- 分析 false positive 和 false negative。

## 6. 目前最大的研究风险

### 时间信息泄漏

虽然训练集和测试集按照 trial 年份切分，但现在查询的是最新版本的公共数据库。

例如，一个 2010 年开始的 trial 可能使用了 2025 年以后才积累出来的 PubMed 或 Open Targets 证据。这样模型虽然在“预测旧 trial”，却看到了后来才出现的信息。

因此，在解决这个问题以前，最好把项目描述成：

> retrospective evaluation

而不要直接说成：

> prospective prediction

### Drug/target 重复

训练集和测试集中可能出现相同 drug 或相同 target。模型可能只是记住某个成功药物，而不是真的学会可以推广的 biology。

下一步需要增加：

- unseen-drug split；
- unseen-target split；
- unseen drug–disease pair split。

### cBioPortal 没有可用数据

本次运行中，cBioPortal 对 315 个查询组合的有效 coverage 是 0。

在修复以前：

- 不能把 0 当成“没有 mutation”；
- 只能把它理解为“没有成功获得数据”；
- primary model 最好暂时排除这部分特征。

## 7. 下一步建议

### 第一优先级

1. 决定 strict 还是 permissive 作为主要 label。
2. 删除或重新定义 balanced label。
3. 人工检查 150–200 个 trial labels。
4. 增加 unseen-drug 和 unseen-target 测试。
5. 给所有 AUC 加 bootstrap 95% confidence interval。

### 第二优先级

1. 完整重新运行 PubMed features。
2. 检查每个数据库的有效 coverage。
3. 修复或暂时移除 cBioPortal。
4. 检查每个数据库是否存在时间泄漏风险。
5. 制作 endpoint-specific 和 missingness 分析。

### 之后制作 paper figures

1. Cohort flowchart。
2. 不同 feature set 的性能比较图。
3. Temporal、unseen-drug、unseen-target 的比较图。
4. PFS、response、OS endpoint 分析图。
5. 各数据源 coverage 和 missingness 图。

## 8. 投稿方向

### 第一选择

**JCO Clinical Cancer Informatics**

适合把文章写成一个临床肿瘤信息学研究，重点是 mechanistic evidence 如何帮助评估或 prioritization oncology trials。

### 其他选择

- **JAMIA/JAMIA Open**：如果重点是数据整合、evaluation framework 和 reproducibility。
- **Bioinformatics Advances**：如果重点是 mechanistic biology data integration 和 computational analysis。
- **Bioinformatics**：要求更高，目前可能需要更强的方法创新或 external validation。

## 9. 明天 meeting 可以问的问题

1. 我们是否同意把 paper 的重点放在“mechanistic biology 什么时候有效”，而不是复杂模型？
2. 我们真正想预测的是 Phase 2 efficacy，还是更广义的 trial success？
3. Strict 和 permissive 哪一个应该作为主要 label？
4. Balanced label 应该重新定义，还是直接删除？
5. Unseen-drug 和 unseen-target 是否应该作为论文主要实验？
6. 无法获得历史版本的数据库应该怎样处理？
7. cBioPortal 是否先从 primary model 中移除？
8. 投稿应该偏向 clinical cancer informatics，还是 computational bioinformatics？

## 10. 明天可以直接这样说

### 中文理解版

我重新运行并检查了现在的 pipeline。这次主要修复了特征重复、endpoint 分类、two-stage model 重复列和 PubMed rate limit 四个问题，也加了五个 regression tests。修复以后，主要结果基本稳定：permissive label 的 AUC 大约是 0.67，strict label 大约是 0.685。

目前 composite biology features 仍然表现最好，但 two-stage 和 deep model 都没有带来提升。Endpoint 修复以后发现不同 endpoint 的差异比较明显：PFS 的 AUC 是 0.759，但测试集只有 35 个，所以现在只能作为 preliminary signal；response endpoint 的 AUC 是 0.443，表现很差。

所以我认为下一步不应该继续优先增加复杂模型，而应该重点检查 label quality、时间信息泄漏、unseen drug/target generalization，以及不同 endpoint 和数据库 coverage 的影响。论文可以围绕“公开 mechanistic biology evidence 在什么情况下能推广到未来 Phase 2 oncology trial outcomes”来设计。

### 英文简短版

> I reran and audited the current pipeline. I fixed feature duplication, endpoint classification, the two-stage model input, and PubMed rate-limit handling, and I added five regression tests. The corrected main result is stable, with an AUC of about 0.67 under the permissive label and 0.685 under the strict label. The composite biology features remain the strongest representation, while the two-stage and deep models do not improve the baseline. Endpoint analysis suggests potentially useful but preliminary PFS performance, while response endpoints perform poorly. I therefore propose focusing the paper on when mechanistic evidence generalizes, rather than on increasing model complexity.

## 11. 哪些结果现在可以说，哪些不能说

### 现在可以说

- Pipeline 已经重新运行并完成检查。
- 四个具体实现问题已经修复。
- 五个 regression tests 全部通过。
- 修复后整体 AUC 仍然在 0.67–0.685 左右。
- Two-stage 和 deep model 目前没有超过 baseline。
- 不同 endpoint 之间可能存在明显差异。

### 现在还不能说

- 不能说模型已经可以真正预测未来临床试验。
- 不能说 PFS 的效果已经得到确认。
- 不能说所有外部数据库都提供了可靠数据。
- 不能把 cBioPortal 的零值解释为没有 mutation evidence。
- 不能把 strict 和 balanced 当成两个不同 label。
- 在完成 unseen-drug/target 测试以前，不能确定模型学到的是可推广的 biology。
