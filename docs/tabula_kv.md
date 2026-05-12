# Tabula KV Compression

本文记录当前项目中的 `tabula` 压缩方法。它是一个面向 WikiTableQuestions 的 strict-budget、question-aware、table-aware KV Cache 压缩策略。

## 目标

`tabula` 的目标是在表格很长、KV budget 很小的情况下，利用问题 token 对表格 token 的注意力分数筛选 KV，同时给表格结构一个轻量归纳偏置：

- prompt 中的 few-shot 示例作为 protected prefix，直接常驻 KV cache，不占 `layer_budget`。
- 真实表格、真实问题和 `Answer:` 等当前样本内容受 `layer_budget` 约束。
- 表头不再完整保留，但每个 header cell 至少保留一个 token 的 K/V。
- body cell token 的分数会融合其对应列表头的分数。

## 相关文件

- `src/artifacts/artifact_infer/cache_mngr/tabula.py`: 压缩器主体。
- `src/artifacts/artifact_infer/cache_mngr/layerwise.py`: 在压缩前构造当前 KV 对应的表格 metadata，并在压缩后同步 metadata。
- `eval/wiki_table_utils.py`: 构造 list-of-lists prompt，并记录真实表格 header/body cell 的字符 span。
- `eval/test_wiki.py`: 用 tokenizer offset mapping 把字符 span 转成 token-level metadata。
- `eval/config_eval_tabula.yaml`: 默认评测配置。

## 输入 Prompt

模型输入仍然是普通 WTQ prompt：

```text
Table:[few-shot table]
Question: ...
Answer: ...
---
Table:[[header...], [row...], ...]
Question: ...
Answer:
```

其中 few-shot 部分由 `protected_token_span` 标记。`tabula` 只压缩 protected prefix 之后的 KV。

## Token Metadata

`eval/test_wiki.py` 为完整 prompt 的每个 token 生成三组标签：

- `header_columns`: token 属于哪个真实表头列；非 header token 为 `-1`。
- `header_cell_ids`: token 属于第几个 header cell；非 header token 为 `-1`。
- `cell_columns`: token 属于哪个真实 body cell 列；非 body cell token 为 `-1`。

这些 metadata 会跟随 strict prefill 的 chunk 进入 `CacheManager`。每轮压缩后，`tabula` 返回被保留位置对应的新 metadata，供下一轮继续使用。

## 压缩流程

`tabula` 走 strict-budget chunked prefill 路径：

1. 先把实际 question 单独编码，保存每层的 `question_q_cache`。
2. Prompt 按 `strict_prefill_chunk_size` 分块进入模型。
3. 每个 chunk prefill 完成后，候选 KV 为：
   `previous compressed KV + current chunk KV`
4. protected prefix KV 不进入候选集合，直接保留。
5. 用 question Q states 对候选 KV 计算 cross-attention。
6. 根据表格 metadata 修正 token 分数。
7. 选出最多 `layer_budget` 个候选 KV，写回 KV cache。

## 分数计算

设 question-aware attention 得到每个候选 token 的基础分数为 `S_token`。

当前实现中 attention 分数处理为：

1. `compute_attention_scores(question_q, candidate_k)`
2. 对 KV 维度做 softmax。
3. 对 question token 维度做 `max` 或 `mean` 聚合，默认 `max`。
4. 对 head 维度取平均，得到每个 token 一个分数。

对 body cell token，如果它属于列 `c`，则找到同列 header token 的最大基础分数作为 `S_header(c)`，并更新：

```text
S = lambda * S_token + (1 - lambda) * S_header(c)
```

`lambda` 来自配置项 `model.tabula_lambda`。当 `lambda=1.0` 时，等价于不注入 header 分数，只使用 question-aware attention。

## 保留规则

在 protected prefix 之外，`layer_budget` 是硬上限。选择规则：

1. 如果候选 KV 长度不超过 `layer_budget`，不压缩。
2. 对每个 header cell，强制保留该 cell 内分数最高的一个 token。
3. 强制保留最近 `query_window_size` 个候选 token。
4. 剩余 budget 按修正后的 token 分数 top-k 填满。
5. 最终保留索引按原始顺序排序，避免打乱 KV 顺序。

注意：header cell 的保底 token 也占用 `layer_budget`。如果 header cell 数量本身超过 `layer_budget`，当前实现会报错。

## 配置示例

```yaml
model:
  cache_compressor: tabula
  enforce_eager: true
  layer_budget: 128
  query_window_size: 32
  question_window_size: 64
  strict_prefill_chunk_size: 64
  protected_kv_cache_size: 256
  tabula_lambda: 1.0
```

运行：

```bash
CUDA_VISIBLE_DEVICES=2 python eval/test_wiki.py --config-file eval/config_eval_tabula.yaml
```

## 与其他方法的公平比较

建议只和 strict-budget 方法比较：

- `rkv_new`
- `snapkv_new`
- `question_new`
- `sliding_new`
- `tabula`

这些方法都不先构建完整 prompt KV cache，而是在 prefill 期间分块压缩。非 strict 的 `sliding`、`rkv`、`snapkv` 是 full-prefill 后压缩，不应直接和 `tabula` 比较。

当前默认配置中，只有 `tabula` 使用 eager 模式，其他 strict-budget 方法默认保留 CUDA graph 路径以加快评测。调试新压缩器或定位数值问题时，可以临时打开：

```yaml
enforce_eager: true
```

## 当前限制

- 目前只支持单请求评测：`batch_size=1`、`max_num_seqs=1`。
- 依赖 tokenizer offset mapping，适用于当前 WTQ list-of-lists prompt。
- 只记录真实表格的 header/body metadata，few-shot 示例不参与 `tabula` 的结构化打分。
- `tabula_lambda` 目前是全局超参，没有 layer-wise 或 head-wise 调度。
- Header cell 至少保留一个 token，但不保证完整 header 文本可恢复。
