# Query-Aware KV Compression

`query` 是一个近似 TableKV/Finch 思路的 one-shot 压缩策略。这里的 query 指 prompt 里的 `Question:` 字段，而不是 Transformer 术语里的 query projection。它不改 prefill 流程，也不把表格逐块送入模型；它在完整 prompt 已经进入 KV cache 后，用问题 token 对全量 KV cache 打分，并按 top-k 保留 KV token。

## 代码位置

- `src/artifacts/nanovllm_v5/cache_mngr/query.py`: `QueryAwareKV` 压缩器。
- `src/services/nanovllm_v5/model_runner/model_runner.py`: `cache_compressor: query` 的注册入口。
- `eval/config_eval_query.yaml`: WikiTableQuestions 的 query-aware 实验配置。

## 算法

输入形状沿用现有 compressor 接口：

```python
query_states: (batch, num_q_heads, query_window, head_dim)
key_states:   (batch, num_kv_heads, kv_len, head_dim)
value_states: (batch, num_kv_heads, kv_len, head_dim)
```

当 `kv_len <= layer_budget` 时不压缩。否则：

1. `test_wiki.py` 在构造 prompt 时记录 `Question:` 后、`Answer:` 前的 question token span。
2. CacheManager 从 `q_cache` 中读取该 question span 对应的 query states。如果 question span 已不在最近 q-cache 窗口内，则退回使用可用的最近 query window。
3. 用 question query states 和全部 key states 计算注意力分数。
4. 对历史区域 `[:-query_window_size]` 做 softmax，并在 question token 维度上 `mean` 聚合。
5. 从历史 token 中选出 `layer_budget - query_window_size` 个分数最高的位置。
6. 拼接被选中的历史 KV 和最近 `query_window_size` 个 KV，得到长度为 `layer_budget` 的压缩 cache。

## 配置

```yaml
model:
  cache_compressor: query
  layer_budget: 128
  query_window_size: 32
  steps_between_cache_compressions: 1
```

含义：

- `layer_budget`: 压缩后总保留 KV token 数。
- `query_window_size`: 最近 token 必保留数量，同时也是当前 q-cache 可覆盖的窗口大小。为了让 question span 被完整读取，prompt 中的 `Question:` 应尽量靠近 prompt 末尾；当前 WikiTableQuestions prompt 满足这一点。
- `steps_between_cache_compressions`: 每隔多少个 decode step 压缩一次。

当前 `layer_budget: 128`、`query_window_size: 32` 表示保留最近 32 个 token，再从历史中按 query-aware attention 选 96 个 token。

## 与 Finch 的关系

这是 query-aware，但不是严格 Finch。严格 Finch 会把表格按 chunk 迭代输入，并在每个 chunk 后用问题 tokens 对“前一轮压缩 cache + 当前 chunk”做 top-k 更新。当前实现是完整 prefill 后的一次性近似，优点是能复用现有 CacheManager，适合作为 TableKV 方向的第一版实验基线。

## `question_new`

`question_new` 是新的 strict-budget question-aware 路径。它仍然使用原始 prompt 作为模型输入，但 `test_wiki.py` 会把 prompt 中的实际 question token 额外传给引擎。引擎在处理 prompt chunk 之前，先把这份 question 单独跑一遍模型，并把每层的 question Q states 存入 `question_q_cache`。

之后每个 prefill chunk 后，few-shot 示例对应的 protected prefix KV 会直接常驻，不参与 top-k，也不消耗 `layer_budget`。压缩器看到的候选 KV 是 protected prefix 之后的“上一轮压缩 cache + 当前 chunk”。`QuestionAwareKVNew` 用 `question_q_cache` 中的 question Q states 对这些候选 KV 计算 cross-attention；最近 `query_window_size` 个 KV token 直接保留，历史部分再选 top `layer_budget - query_window_size` 个 token，形成下一轮 compressed cache。

配置示例：

```yaml
model:
  cache_compressor: question_new
  layer_budget: 128
  query_window_size: 32
  question_window_size: 64
  strict_prefill_chunk_size: 64
  protected_kv_cache_size: 256
```

这和旧 `query` 的区别是：`query` 是完整 prompt prefill 后的一次性近似；`question_new` 是 chunked prefill 下的迭代式压缩，且用于剪枝的 question 是单独拎出来编码的。
