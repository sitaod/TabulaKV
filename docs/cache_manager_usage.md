# Cache Manager 使用说明

本文对应当前保留的 `nanovllm_v5` 版本，主要说明 KV Cache 压缩研究中会接触到的 Cache Manager、压缩器和调度元数据流程。

## 代码位置

- `src/artifacts/nanovllm_v5/cache_mngr/layerwise.py`: Cache Manager 主体。
- `src/artifacts/nanovllm_v5/cache_mngr/snapKV.py`: SnapKV 压缩器。
- `src/artifacts/nanovllm_v5/cache_mngr/snapKV_new.py`: strict-budget SnapKV 压缩器入口。
- `src/artifacts/nanovllm_v5/cache_mngr/RKV.py`: RKV 压缩器。
- `src/artifacts/nanovllm_v5/cache_mngr/RKV_new.py`: strict-budget RKV 压缩器入口。
- `src/artifacts/nanovllm_v5/cache_mngr/sliding.py`: 非 strict sliding-window 压缩器。
- `src/artifacts/nanovllm_v5/cache_mngr/sliding_new.py`: strict-budget sliding-window 压缩器入口。
- `src/artifacts/nanovllm_v5/cache_mngr/tabula.py`: strict-budget table-aware question-aware 压缩器。
- `src/artifacts/nanovllm_v5/attention/flashinfer_attention.py`: 读写 KV/Q cache 的 Triton kernel 和 FlashInfer decode metadata。
- `src/services/nanovllm_v5/model_runner/model_runner.py`: 初始化 compressor 和 CacheManager，并在 prefill/decode 阶段调用 metadata 更新。

## 运行时关系

`LLMEngine` 创建 `ModelRunner` 和 `Scheduler`。`Scheduler` 负责给每个 `Sequence` 分配 `block_table` 和 `query_block_id`，`ModelRunner` 负责模型执行和 cache tensor。

`ModelRunner.__init__` 中会构造：

```python
self.compressor = SnapKV(
    window_size=config.query_window_size,
    budget=config.layer_budget,
)
self.cache_mngr = CacheManager(self.attention_backend, config, self.compressor)
```

`CacheManager` 通过 artifact 注册机制拿到 attention backend 暴露的方法，例如 `prepare_metadata_for_attn`、`init_forward_metadata_capture_cuda_graph` 和 `init_forward_metadata_replay_cuda_graph`。随后 `ModelRunner.prepare_decode()` 会先整理当前 batch 的 block metadata，再调用 Cache Manager 更新 FlashInfer 所需的 page table。

## 关键配置

这些配置在 `src/services/nanovllm_v5/config.py`：

- `kvcache_block_size`: 当前 v5 只支持 `1`。不要在压缩实验前改大。
- `query_window_size`: 最近 KV token 保留窗口长度，也是普通 `q_cache` 的窗口长度，默认 `32`。
- `question_window_size`: `question_new` 额外保存的独立 question Q cache 长度，默认 `64`。
- `layer_budget`: 压缩后的 KV token budget，默认 `320`。
- `cache_compressor`: 压缩算法名称。`none` 表示 full KV cache baseline；`sliding` 表示 full-prefill 后只保留最近的 KV token；`sliding_new` 表示 strict-budget chunked prefill 版 sliding；`query` 表示 query-aware top-k；`snapkv` 和 `rkv` 分别使用对应压缩器；`snapkv_new`、`rkv_new`、`question_new`、`tabula` 启用 strict-budget chunked prefill 路径。
- `steps_between_cache_compressions`: 压缩频率。`1` 表示每个 step 后尝试压缩一次，`2` 表示每两个 step 压缩一次。
- `strict_prefill_chunk_size`: 仅 `_new` 压缩器使用。prefill 时每次送入模型的 prompt chunk token 数，应不大于 `layer_budget`。
- `protected_kv_cache_size`: strict-budget 路径为 protected prefix 额外预留的 KV block 数。WTQ 的 few-shot 示例会作为 protected prefix 常驻 KV cache，不消耗 `layer_budget`。
- `tabula_lambda`: 仅 `tabula` 使用。cell token 分数会按 `S = lambda * S_cell + (1 - lambda) * S_header` 融合对应表头属性分数。

在当前实现里，`budget` 的含义取决于 compressor：

- `sliding`: `layer_budget` 就是 protected prefix 之后最终保留的最近 KV token 数，但它在完整 prefill 后才开始压缩，不适合和 `_new` strict-budget 结果直接公平比较。
- `sliding_new`: `layer_budget` 是 protected prefix 之外的持久化 KV cache 硬上限。prefill 分块执行，每个 chunk 后只保留 protected prefix + 最近 `layer_budget` 个非 protected KV token。
- `query`: `layer_budget` 是 protected prefix 之外最终保留的总 KV token 数，其中最近 `query_window_size` 个 token 必保留，剩余 `layer_budget - query_window_size` 个 token 用 query window 对历史 KV 的注意力分数选择。
- `snapkv` / `rkv`: `layer_budget` 是 protected prefix 之外最终保留的总 KV token 数，其中最近 `query_window_size` 个 token 必保留，剩余 `layer_budget - query_window_size` 个 token 从历史 KV 中按算法选择。
- `snapkv_new` / `rkv_new`: `layer_budget` 是 protected prefix 之外的持久化 KV cache 硬上限。prefill 会按 `strict_prefill_chunk_size` 分块，当前 chunk 结束后立即把 protected prefix 之外的“上一轮压缩结果 + 当前 chunk”压回 `layer_budget` 以内，其中最近 `query_window_size` 个 token 必保留。块内计算的瞬时工作集最多是 `protected_kv_cache_size + layer_budget + strict_prefill_chunk_size`，但不会先建立完整 prompt KV cache。
- `question_new`: `layer_budget` 是 protected prefix 之外每轮压缩后保留的 KV token 数。它会额外把 prompt 中的实际 question 单独编码到 `question_q_cache`，每个 chunk 后用 question Q states 对 protected prefix 之外的“上一轮压缩 cache + 当前 chunk”做 cross-attention；最近 `query_window_size` 个 token 必保留，剩余 `layer_budget - query_window_size` 个位置从历史候选中按分数选择。
- `tabula`: 基于 `question_new` 的 strict-budget 路径。它额外接收当前表格 token 的列属性 metadata，强制保留实际表头 KV token，表头 token 占用 `layer_budget`；对 body cell token，先用 question attention 得到 `S_cell`，再融合对应列表头的 `S_header`。最后在 protected prefix 之外按 budget 保留“表头 + 最近窗口 + top score token”。
- `none`: 不调用压缩，`layer_budget` 不影响 KV 保留长度。

RKV 的 similarity 计算需要额外临时显存，建议优先使用 `enforce_eager: true` 并降低 `gpu_memory_utilization`，给压缩计算留出余量。

## CacheManager 方法

`prepare_indices_flashinfer(seqs)`  
从 `seq.block_table` 拼出当前 batch 的 `page_indices` 和 `seq_lens`，并记录当前占用 page 数。decode 之前必须先调用它。

`update_indices()`  
非 CUDA graph 路径使用。它把 `seq_lens` 和 `page_indices` 交给 attention backend 的 `prepare_metadata_for_attn()`，为 FlashInfer decode 准备 metadata。

`update_indices_capture(bs)`  
CUDA graph capture 阶段使用。它为固定 batch size 创建并保存 FlashInfer decode wrapper。

`update_indices_replay(bs)`  
CUDA graph replay 阶段使用。它复用 capture 阶段保存的 wrapper，并把当前请求的 page indices 写入预分配 buffer。

`read_and_store_cache(q_cache, k_cache, v_cache)`  
真正执行压缩。它会先把 `seq.protected_token_span` 对应的 prefix KV 直接保留，然后只把后续 KV 传给 `self.compressor.update_kv()`，再把 protected KV + 压缩后的 KV 写回原 cache tensor，并截断当前序列的 `block_table`。

## 压缩器接口

压缩器只需要实现一个方法：

```python
class MyCompressor:
    def update_kv(self, query_states, key_states, value_states):
        # query_states: (batch, num_q_heads, query_window, head_dim)
        # key_states:   (batch, num_kv_heads, kv_len, head_dim)
        # value_states: (batch, num_kv_heads, kv_len, head_dim)
        return new_key_states, new_value_states
```

返回的 `new_key_states` 和 `new_value_states` 形状应保持为 `(batch, num_kv_heads, new_kv_len, head_dim)`，并且二者的 `new_kv_len` 必须一致。`SlidingWindowKV`、`SnapKV` 和 `RKV` 都可以作为模板。

切换算法时，优先使用配置文件：

```yaml
model:
  cache_compressor: sliding
  layer_budget: 320
  steps_between_cache_compressions: 1
```

如果要写死算法，也可以在 `ModelRunner.__init__` 中替换 compressor：

```python
from src.artifacts.nanovllm_v5.cache_mngr.RKV import RKV

self.compressor = RKV(
    window_size=config.query_window_size,
    budget=config.layer_budget,
)
```

## 如何启用压缩

当前自动压缩逻辑在 `src/services/nanovllm_v5/engine/llm_engine.py` 中由 `cache_compressor` 控制：

```python
if (
    cache_compressor not in ("", "none", "full", "full_kv")
    and not is_prefill
    and self.cur_step % self.config.steps_between_cache_compressions == 0
):
    self.model_runner.call("compress")
```

压缩只在 decode step 后触发。Prefill 路径尚未调用 `prepare_indices_flashinfer()`，没有可用于 `read_and_store_cache()` 的当前序列 metadata。

`sliding_new` / `snapkv_new` / `rkv_new` / `question_new` / `tabula` 是例外：它们不走完整 prefill，而是由 `LLMEngine` 驱动 chunked prefill。每个 prefill chunk 调用一次模型，然后立即调用 `compress(seqs)`，截断 `seq.block_table` 并释放被淘汰 block。KV pool 会按 `protected_kv_cache_size + layer_budget + strict_prefill_chunk_size` 个 block 预分配，而不是按 full context 剩余显存尽量开大。这个路径当前要求：

- `generation.batch_size: 1`
- `model.max_num_seqs: 1`
- `model.kvcache_block_size: 1`

示例：

```yaml
model:
  cache_compressor: question_new
  layer_budget: 128
  query_window_size: 32
  question_window_size: 64
  strict_prefill_chunk_size: 64
```

做实验时可以先在单请求、`enforce_eager=True` 下设置 `cache_compressor`，或在调试脚本中直接调用：

```python
llm.model_runner.call("compress")
```

建议先跑短上下文单请求，确认压缩后的 `seq.block_table`、FlashInfer metadata 和输出都正常，再打开 CUDA graph 路径。

## 当前限制

- `read_and_store_cache()` 当前有 `assert len(self.cu_seqs) == 1`，压缩路径只支持单请求。
- v5 的 block manager 明确不支持 prefix caching，并且只支持 `block_size == 1`。
- 压缩所有层时会先使用同一份原始 `seq.block_table` 读 full KV，所有层写回完成后再统一截断 `seq.block_table` 并释放被淘汰的 block。
- layer-wise metadata 的接口已经留下位置，但当前实现使用的是统一 `seq.block_table`，尚未真正维护每层独立 block table。
- CUDA graph 路径对 batch size 和预分配 buffer 更敏感。新增压缩策略时，建议先验证 eager decode，再验证 graph replay。

## 新增压缩算法建议流程

1. 在 `src/artifacts/nanovllm_v5/cache_mngr/` 下新增算法文件。
2. 实现 `update_kv(query_states, key_states, value_states)`。
3. 在 `ModelRunner.__init__` 中实例化新 compressor。
4. 用 `enforce_eager=True` 跑单请求短生成，检查压缩前后 `kv_len` 和输出。
5. 打开自动 `compress()` 调用，调小 `steps_between_cache_compressions` 做压力测试。
6. 最后再验证 `enforce_eager=False` 的 CUDA graph 路径。
