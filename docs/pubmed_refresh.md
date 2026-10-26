# PubMed 查询修复与单独刷新

## 2026-09-14 实际完成结果

- 保存的原始记录 5,000 条，当前过滤/标准化代码保留 4,303 条 trial；其中 2,465 条有已知 target，形成 335 个唯一 target–disease pair。1,838 条未知 target 的记录未发起 PubMed pair 查询，不能算作已补齐数据。
- **335/335 个组合全部成功，未完成 0。** 三类原始计数均无缺失，子集计数均不超过总计数。
- 39 个组合的总文献数为真实零结果；与请求失败明确区分。
- 真实运行中出现的 HTTP 429 和无效响应均重试恢复。
- 对全部 335 个组合重新调用客户端并禁止网络访问，全部命中有效缓存，实际网络请求为 0。
- 再次执行刷新入口，335 个已完成组合全部跳过，查询日志没有新增记录。
- 335 个结果均通过原始特征与 composite 计算接口检查；10 项 PubMed/续跑测试通过，语法编译及 `git diff --check` 通过。
- 完整旧测试集未确认通过：加载已有 SHAP/Numba 依赖长时间停留后中断；不是 PubMed 专项测试失败。

结果目录：`mechanistic_engine_output/pubmed_refresh_2026-09-14/`；验收记录：`verification.json`。这些是本次当前代码对保存原始数据的结果，不能与旧 638/1,038 行模型快照混作同一 cohort。

## 修复范围

- 原实现会将 HTTP 200 的 `{"error": ...}`、缺少 count 的响应记为成功且文献数为 0。新客户端严格校验响应；失败为 `None`，真实零结果才为 0。
- 三类查询（总文献、clinical trial、近期文献）分别验证；只有三项齐全且子集计数不超过总数，`has_real_pubmed_data` 才为真。
- 每次实际请求和重试均限速至最多每 0.4 秒一次；HTTP 429/5xx、网络错误、无效 JSON 可重试。遵守数字或 HTTP-date 格式的 `Retry-After`，不再把较长服务端等待截成 30 秒。
- 缓存键包含完整查询参数和日期，并使用 `pubmed_v2_` 新命名空间，不复用旧的可疑缓存。无效响应不缓存，损坏缓存重新获取。
- 日期窗口由明确的 `as_of` 决定。2026-09-14 的近期窗口为 2023-01-01 至 2026-09-14，不再固定为 2021–2024。
- 保留原始计数、检索式、日期、状态；保留既有的三个缩放特征以尽量减少模型接口变动。
- 特征生成中保留失败计数为 NaN，并加入 `pubmed_data_missing`；模型准备阶段沿用项目的 -1 填充规则。Composite 分数仍沿用既有缺失项按 0 加权的实现，因此 missing flag 必须与分数一起解释。

实现位于 `pubmed_client.py`，`clients.query_pubmed_pair` 仍是可用入口。没有新增依赖。

## 完整刷新和续跑

在项目根目录执行：

```bash
.venv/bin/python refresh_pubmed.py \
  --as-of 2026-09-14 \
  --output-dir mechanistic_engine_output/pubmed_refresh_2026-09-14
```

默认读取保存的 `mechanistic_engine_output/data/raw_trials.json`，沿用当前 pipeline 的过滤和标准化，对所有已知 primary target–disease pair 刷新 PubMed，不调用其他数据库。未知 target 的 trial 数单独报告。

相同命令可以续跑：已完成 pair 跳过；partial/failed pair 重试，已成功的子查询可走缓存。输入、标准化代码、客户端或日期变化时要求使用新的输出目录。不要并行启动多份刷新，进程间不共享限速器。

输出：

- `input_manifest.json`：原始数据与相关代码 SHA-256、截止日期。
- `pair_attempts.jsonl`：逐 pair 追加记录，保留失败历史。
- `pubmed_pair_features.csv`：各 pair 最新结果，包括原始计数和状态。
- `summary.json`：筛选、映射、成功和未完成的数量。

连续三项完全失败时停止并保存进度，以便排查联网问题。非零退出码表示尚未全量完成，不应直接把结果当作完整刷新。

这些文件是独立的 PubMed 特征表，不会覆盖原始数据、已有 feature matrix 或模型报告。下一次运行完整 pipeline 时将使用新客户端和匹配日期的缓存；已有模型需要在新数据快照上重训后才能报告新指标，不能只替换三列而保留旧 composite 分数。

## 验证命令

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_pubmed_client.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_refresh_pubmed.py' -v
```

## 尚未由本次解决的研究问题

`as_of` 限制的是 publication date，不等于恢复当年的 PubMed 索引快照。本次使用统一的 2026-09-14 上限，**不是按每个 trial 的开始时间截断**，因此不能宣称历史时间泄漏已经解决。

`pair_pub_acceleration` 保留旧公式：`min(recent_count / max(total_count * 0.4, 1), 1)`，它是近期占比 proxy，不是严格的增长率。Target/disease 的别名扩展、模糊 target（例如 DNA）、`cancer nos` 的检索特异性也仍需要后续研究判断。成功返回计数仅表示请求有效，不保证召回了所有相关文献。

NCBI 官方说明：[E-utilities 使用政策](https://www.ncbi.nlm.nih.gov/books/NBK25497/)；[ESearch 日期参数](https://www.ncbi.nlm.nih.gov/books/NBK25499/)。
