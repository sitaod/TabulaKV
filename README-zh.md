# TabulaKV

TabulaKV 是一个面向表格问答任务的 KV Cache 压缩研究原型。项目基于 Artifact-Infer 推理框架，加入了 Cache Manager、压缩器注册机制、WikiTableQuestions 评测流程，以及多种 KV Cache 压缩方法。

除了常见压缩基线外，仓库还实现了 TabulaKV，一种 question-aware 与 table-aware 结合的压缩方法，它会在 KV 选择过程中利用表头和列属性信息。

## 功能特点

- 基于 Artifact-Infer 的 Qwen 推理运行时。
- 统一的 KV Cache 压缩器接口。
- 支持 Full KV、RKV、SnapKV、Sliding Window、Question-aware 和 TabulaKV 等方法。
- 支持 strict-budget chunked prefill，用于小预算下的公平对比。
- 提供 WikiTableQuestions 预处理和评测流程。
- 记录逐条样本输出、正确性和汇总结果。

## 目录结构

```text
src/services/artifact_infer/          Artifact-Infer 推理运行时与 model runner
src/artifacts/artifact_infer/         attention、block manager 和 cache manager
src/artifacts/artifact_infer/cache_mngr/
                                      KV Cache 压缩算法
eval/                                 WikiTableQuestions 预处理与评测脚本
docs/                                 Cache Manager、Question-aware 和 TabulaKV 文档
datasets/wikitablequestions/          配置默认使用的 WTQ JSONL 数据
```

## 环境安装

先安装基础依赖：

```bash
pip install -r requirements.txt
```

FlashAttention 需要在 PyTorch 安装完成后单独安装：

```bash
pip install -r requirements-flash-attn.txt --no-build-isolation
```

可选的非贪心采样 kernel 可通过以下命令安装：

```bash
pip install -r requirements-sgl-kernel.txt
```

项目开发环境使用 Python 3.10、PyTorch 2.8、CUDA 12.9、FlashInfer 和本地 Qwen3 checkpoint。其他 CUDA 或 PyTorch 版本可能需要调整 FlashAttention 与 FlashInfer 的安装方式。

## 数据

评测代码面向 Hugging Face 数据集 `stanfordnlp/wikitablequestions`，配置为 `random-split-1`。

默认配置指向固定的 500 条 test 子集：

```text
datasets/wikitablequestions/random-split-1_test_seed42_sample500.jsonl
```

该子集由完整 test split 使用 Python `random.Random(42)` 打乱后取前 500 条得到，便于快速、可复现地运行实验。

如需重新生成预处理后的 JSONL 文件：

```bash
python eval/preprocess_wikitablequestions.py --config-file eval/config_eval.yaml
```

## 运行评测

先在 YAML 配置中将 `model.path` 改成本地 Qwen checkpoint 路径：

```yaml
model:
  path: /path/to/Qwen3-8B
```

运行 Full KV baseline：

```bash
python eval/test_wiki.py --config-file eval/config_eval_fullkv.yaml
```

运行压缩方法：

```bash
python eval/test_wiki.py --config-file eval/config_eval_rkv_new.yaml
python eval/test_wiki.py --config-file eval/config_eval_snapkv_new.yaml
python eval/test_wiki.py --config-file eval/config_eval_question_new.yaml
python eval/test_wiki.py --config-file eval/config_eval_sliding.yaml
python eval/test_wiki.py --config-file eval/config_eval_tabula.yaml
```

strict-budget 配置默认使用：

```yaml
model:
  layer_budget: 128
  query_window_size: 32
  strict_prefill_chunk_size: 64
  protected_kv_cache_size: 256
```

few-shot 示例部分会作为 protected prefix 常驻 KV Cache，不占当前样本的压缩预算。

## 压缩方法

`none` 表示保留完整 KV Cache。

`sliding_new` 使用 strict-budget chunked prefill，并保留最近的 KV token。

`rkv_new` 和 `snapkv_new` 是 RKV 与 SnapKV 的 strict-budget 版本。

`question_new` 会单独编码当前样本的真实 question，并用 question-to-prompt cross-attention 在 chunked prefill 过程中选择 KV token。

`tabula` 在 question-aware 压缩基础上加入表格结构信息。它会将 header cell 和 body cell 映射到 token-level metadata，保证每个 header cell 至少保留一个 token，并使用对应列表头的注意力分数修正 body cell token 分数：

```text
S = lambda * S_cell + (1 - lambda) * S_header
```

`model.tabula_lambda` 控制这个加权比例。

## 日志

评测输出默认写入 `logs/wiki_eval/`：

```text
*.correct.jsonl   每条样本的正确性、预测答案和标准答案
*.qa.jsonl        问题、原始输出、清洗后的预测和标准答案
*.summary.json    总样本数、正确数、accuracy 和日志路径
```

日志是运行生成文件，默认不会纳入 git。

## 文档

- `docs/cache_manager_usage.md`: Cache Manager 与压缩器接口说明。
- `docs/question_aware_kv.md`: Question-aware 压缩方法说明。
- `docs/tabula_kv.md`: TabulaKV 算法细节。

## 注意事项

当前压缩实验主要面向单请求评测：

```yaml
generation:
  batch_size: 1
model:
  max_num_seqs: 1
  kvcache_block_size: 1
```

TabulaKV 默认使用 `enforce_eager: true`，便于调试当前实现。其他配置默认使用 `enforce_eager: false`，以加快评测速度。
