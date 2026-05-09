# TabulaKV

TabulaKV 当前保留的是 `nanovllm_v5` 实验版，主要用于基于 Qwen 的表格问答和后续 KV Cache 压缩研究。

## 环境

```bash
conda activate tabulakv
pip install -r requirements.txt
pip install -r requirements-flash-attn.txt --no-build-isolation
```

当前依赖按本机环境整理：Ubuntu 22.04、x86_64、Python 3.10.20、6x NVIDIA RTX PRO 6000 Blackwell、Driver 575.57.08、CUDA 12.9。PyTorch 使用官方 CUDA 12.9 wheel channel `cu129`。

`flash-attn` 需要单独第二步安装：它的构建脚本会在 setup 阶段 import `torch`，如果直接放在 `requirements.txt` 里，pip 的 build isolation 会创建一个看不到 torch 的临时构建环境，从而报 `ModuleNotFoundError: No module named 'torch'`。因此先安装 `torch==2.8.0`、`triton==3.4.0`、`flashinfer-python==0.6.7` 等基础依赖，再用 `--no-build-isolation` 让 `flash-attn==2.8.3` 复用当前 conda 环境里的 torch/CUDA 配置。

`huggingface-hub` 锁定为 `0.35.3`，因为 `transformers==4.57.0` 要求 `huggingface-hub>=0.34.0,<1.0`。如果环境里已有 `huggingface-hub 1.x`，请让 `pip install -r requirements.txt` 将它降级到该兼容版本。

如果 `flash-attn` 需要从源码编译，请确认当前环境能找到 CUDA 12.9 的 `nvcc`。Blackwell GPU 通常需要生成 `sm_120`/compute capability 12.0 代码；如果编译脚本没有自动识别，可临时设置：

```bash
TORCH_CUDA_ARCH_LIST="12.0" pip install -r requirements-flash-attn.txt --no-build-isolation
```

安装后可做一次基础检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import flash_attn, flashinfer; print('flash-attn/flashinfer ok')"
flashinfer show-config
```

`flashinfer-python` 会在首次使用时编译或下载部分内核；如需减少首次运行等待或离线运行，可额外安装 FlashInfer 的预编译内核包：

```bash
pip install flashinfer-cubin
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
```

`sgl-kernel` 只用于非贪心采样的快速 kernel，当前 WikiTableQuestions 默认配置是贪心解码，不需要它。RTX PRO 6000 Blackwell + `torch==2.8.0`/`cu129` 下 `sgl-kernel==0.3.21` 可能在 import 时出现 PyTorch C++ 符号不匹配；v5 采样器已经内置 PyTorch fallback。如果后续确认某个 `sgl-kernel` 版本和当前 torch 兼容，可按需安装：

```bash
pip install -r requirements-sgl-kernel.txt
```

## WikiTableQuestions 评测

评测脚本位于 `eval/test_wiki.py`，预处理脚本位于 `eval/preprocess_wikitablequestions.py`。

数据集使用 Hugging Face 上的 `stanfordnlp/wikitablequestions`。预处理会把每个表格转成 list of lists，其中第一个子列表是表头，后续子列表是表格行。

所有评测参数默认从 [eval/config_eval.yaml](eval/config_eval.yaml) 读取，包括模型路径、v5 引擎显存参数、数据 split、预处理输出目录、生成参数和日志目录。修改模型目录时，编辑：

```yaml
model:
  path: qwen-8B
```

### 1. 预处理数据

```bash
python eval/preprocess_wikitablequestions.py
```

输出示例：

```text
datasets/wikitablequestions/random-split-1_validation.jsonl
```

如需使用另一份配置文件：

```bash
python eval/preprocess_wikitablequestions.py \
  --config-file eval/config_eval.yaml
```

### 2. 运行评测

模型、数据、batch size、输出 token 数、日志名都从 `eval/config_eval.yaml` 读取。`model.path` 需要是本地 Qwen 模型目录。

```bash
python eval/test_wiki.py
```

如需使用另一份配置文件：

```bash
python eval/test_wiki.py \
  --config-file eval/config_eval.yaml
```

### 32B 单卡显存参数

32B 模型在单张 96GB GPU 上运行时，不建议使用 v5 默认的 `max_num_seqs: 128` 和 `gpu_memory_utilization: 0.7`。默认并发会额外分配较大的 q-cache，70% 显存预算也会让 KV cache 可用空间被压到 0。评测 batch size 为 1 时可使用：

```yaml
model:
  path: /data/pretrain_models/Qwen3-32B
  tensor_parallel_size: 1
  max_num_seqs: 1
  max_model_len: 32768
  max_num_batched_tokens: 32768
  gpu_memory_utilization: 0.90
  query_window_size: 64
```

如果仍然提示 KV cache 显存不足，优先降低 `max_model_len` 或确认该 GPU 没有其他进程占用；如果要提高 `generation.batch_size`，同步提高 `model.max_num_seqs`。

### 限制评测条数

`test_wiki.py` 支持通过 `eval/config_eval.yaml` 限制评测样本数。把 `dataset.limit` 设为整数即可只评测前 N 条；保留为空表示评测全部样本。

```yaml
dataset:
  limit: 100
```

如果 `dataset.data_file` 指向已经预处理好的 JSONL，脚本会读取该文件后截取前 N 条；如果不设置 `data_file`，则会从 Hugging Face 数据集加载后截取前 N 条。

预处理阶段也可以单独限制输出条数，用于快速生成小规模调试集：

```yaml
preprocess:
  limit: 100
```

### Prompt 格式

脚本会生成如下格式：

```text
The table is arranged as a list of lists, where the first sub-list is the table header and each subsequent sub-list is a tuple in the table.
Table:
[["Name", "age", "sex"], ["John", 20, "Male"], ["Li", 19, "Female"], ["Zhang", 21, "Male"]]
Question: ...
Directly output the answer without any additional explanation.
```

默认使用完整表格。调试长表格时可在 `eval/config_eval.yaml` 中限制行数：

```yaml
prompt:
  max_table_rows: 20
```

### 输出日志

默认日志目录是 `logs/wiki_eval/`。每次运行会在该目录下新建当前时间文件夹，例如 `logs/wiki_eval/20260509_153012/`。如果 `logging.run_name` 留空，脚本会按 `model.path` 和 `dataset.split` 自动生成日志名前缀，例如 `/data/pretrain_models/Qwen3-8B` 会在时间文件夹中生成 `qwen3_8b_wtq_validation.*`。如果配置文件中显式设置 `logging.run_name: qwen8b_wtq_val`，则会生成：

- `qwen8b_wtq_val.correct.jsonl`: 每条样本是否正确、预测答案、标准答案。
- `qwen8b_wtq_val.qa.jsonl`: 问题、原始输出、清洗后的预测、标准答案。
- `qwen8b_wtq_val.summary.json`: 总样本数、正确数、accuracy 和日志路径。

当前准确率判断使用轻量字符串归一化匹配；`qa.jsonl` 保留了问题、输出、标准答案，便于后续接入更鲁棒的答案判断方案。

### 当前设置

`test_wiki.py` 默认使用 full KV cache baseline。配置文件中：

```yaml
model:
  cache_compressor: none
```

表示不调用 `compress()`，不会启用任何 KV Cache 压缩方法。做 sliding-window 压缩实验时改为：

```yaml
model:
  cache_compressor: sliding
  layer_budget: 320
  steps_between_cache_compressions: 1
```

此时 `layer_budget` 表示每次压缩后只保留最近 320 个 KV token。`snapkv` / `rkv` 也可填入 `cache_compressor`，但它们的 `layer_budget` 表示总保留 token 数，其中最近 `query_window_size` 个 token 必保留。
