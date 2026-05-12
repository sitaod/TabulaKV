# Question-Aware KV Compression

本文记录当前项目中的 `question_new` 压缩方法。它是一个 strict-budget、chunked prefill 的 question-aware KV Cache 压缩策略。

## 目标

`question_new` 的目标是在不构建完整 prompt KV Cache 的前提下，让当前样本的真实 question 参与 KV 选择。模型主输入仍然是原始 prompt；question 会被额外单独编码一份，只用于计算剪枝分数。

该方法的核心特点是：

- prompt 中的 few-shot 示例作为 protected prefix，直接常驻 KV Cache，不占 `layer_budget`。
- 当前样本的真实表格、问题和回答引导部分受 `layer_budget` 约束。
- prefill 按 `strict_prefill_chunk_size` 分块执行，每个 chunk 后立即压缩。
- 每轮压缩时，用 question Q states 对“上一轮压缩 cache + 当前 chunk”做 cross-attention 打分。
- 最近 `query_window_size` 个 KV token 强制保留，剩余 budget 按 question-aware 分数 top-k 选择。

## 相关文件

- `src/artifacts/artifact_infer/cache_mngr/question_new.py`: `QuestionAwareKVNew` 压缩器。
- `src/artifacts/artifact_infer/cache_mngr/layerwise.py`: Cache Manager，在压缩前后读写 KV Cache。
- `src/services/artifact_infer/model_runner/model_runner.py`: `cache_compressor: question_new` 的注册入口，并负责准备 question cache。
- `eval/test_wiki.py`: 构造 prompt，提取当前样本 question，并传给推理引擎。
- `eval/config_eval_question_new.yaml`: WikiTableQuestions 默认评测配置。

## 输入与 Question Cache

模型实际生成时仍然读取完整 prompt：

```text
Table:[few-shot table]
Question: ...
Answer: ...
---
Table:[[header...], [row...], ...]
Question: ...
Answer:
```

其中 few-shot 示例部分会被标记为 protected prefix。`question_new` 不改变主 prompt 的 token 顺序，也不把 question 从 prompt 中删除。

在 strict prefill 开始前，`test_wiki.py` 会把当前样本的真实 question 单独传入引擎。`ModelRunner.prepare_question_cache()` 会单独运行一次 question tokens，并保存每一层的 question Q states。这个缓存只用于后续压缩打分，不直接参与最终生成。

## 压缩流程

`question_new` 走 strict-budget chunked prefill 路径：

1. 先单独编码当前样本 question，保存每层 `question_q_cache`。
2. Prompt 按 `strict_prefill_chunk_size` 分块进入模型。
3. 每个 chunk prefill 完成后，候选 KV 为 protected prefix 之后的：
   `previous compressed KV + current chunk KV`
4. protected prefix KV 不进入候选集合，直接保留。
5. 用当前层的 question Q states 对候选 K 计算 cross-attention 分数。
6. 强制保留最近 `query_window_size` 个候选 KV。
7. 剩余 `layer_budget - query_window_size` 个位置从历史候选中按分数 top-k 选择。
8. 保留索引按原始顺序排序，写回 KV Cache，作为下一轮 compressed cache。

## 分数计算

输入形状沿用 compressor 接口：

```python
query_states: (batch, num_q_heads, question_window, head_dim)
key_states:   (batch, num_kv_heads, kv_len, head_dim)
value_states: (batch, num_kv_heads, kv_len, head_dim)
```

其中 `query_states` 来自单独编码的 question，而不是当前 prompt 末尾的普通 q-cache。

当前实现的打分流程为：

1. 用 question Q states 和候选 K states 计算 attention logits。
2. 对 KV 维度做 softmax。
3. 在 question token 维度上聚合，得到每个 KV token 的相关性分数。
4. 在 head 维度上平均，得到每个候选 token 一个最终分数。
5. 对最近窗口之外的历史 token 做 top-k 选择。

## 配置示例

```yaml
model:
  cache_compressor: question_new
  layer_budget: 128
  query_window_size: 32
  question_window_size: 64
  strict_prefill_chunk_size: 64
  protected_kv_cache_size: 256
```

含义：

- `layer_budget`: protected prefix 之外，每轮压缩后最多保留的 KV token 数。
- `query_window_size`: 最近 token 必保留数量。
- `question_window_size`: 单独 question cache 中最多保存的 question token 数。
- `strict_prefill_chunk_size`: prompt 分块 prefill 的 chunk 大小。
- `protected_kv_cache_size`: 为 few-shot protected prefix 预留的 KV block 数。

运行：

```bash
python eval/test_wiki.py --config-file eval/config_eval_question_new.yaml
```

## 与其他方法的关系

`question_new` 与 `rkv_new`、`snapkv_new`、`sliding_new`、`tabula` 一样，属于 strict-budget 方法。它们都在 prefill 期间分块压缩，不先构建完整 prompt KV Cache。

`tabula` 是在 `question_new` 基础上加入表格结构信息的扩展方法。它同样使用 question Q states 做 cross-attention，但额外引入 header/body cell 的 token-level metadata，并用表头分数影响对应列 body cell 的保留分数。

## 当前限制

- 当前压缩路径主要面向单请求评测：`batch_size=1`、`max_num_seqs=1`。
- question cache 需要额外运行一次 question tokens，因此会带来少量额外 prefill 成本。
- 方法只根据 question-aware token 分数选择 KV，不显式建模表格行、列或 cell 结构；需要结构信息时可使用 `tabula`。
