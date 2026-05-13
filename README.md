# TabulaKV

TabulaKV is a research prototype for KV Cache compression in table question answering. It builds on the Artifact-Infer runtime and adds a Cache Manager, compressor registry, WikiTableQuestions evaluation scripts, and several KV Cache compression baselines.

In addition to standard compression baselines, this repository includes TabulaKV, a question-aware and table-aware compressor that uses table header and column metadata during KV selection.

## Features

- Artifact-Infer based Qwen inference runtime.
- Unified KV Cache compressor interface.
- Full KV, RKV, SnapKV, sliding-window, query-aware, question-aware, and TabulaKV compressors.
- Strict-budget chunked prefill variants for fair small-budget comparison.
- WikiTableQuestions preprocessing and evaluation pipeline.
- Per-sample QA, correctness, and summary logging.

## Repository Layout

```text
src/services/artifact_infer/          Artifact-Infer runtime and model runner
src/artifacts/artifact_infer/         attention, block manager, and cache manager code
src/artifacts/artifact_infer/cache_mngr/
                                      KV Cache compression algorithms
eval/                                 WikiTableQuestions preprocessing and evaluation
docs/                                 Cache manager, question-aware, and TabulaKV notes
datasets/wikitablequestions/          prepared WTQ JSONL files used by the configs
```

## Installation

Create a Python environment, then install the core requirements:

```bash
pip install -r requirements.txt
```

FlashAttention is installed separately because its build step imports PyTorch:

```bash
pip install -r requirements-flash-attn.txt --no-build-isolation
```

Optional non-greedy sampling kernels can be installed with:

```bash
pip install -r requirements-sgl-kernel.txt
```

The project was developed with Python 3.10, PyTorch 2.8, CUDA 12.9, FlashInfer, and Qwen3 local checkpoints. Other CUDA/PyTorch combinations may require adjusting the FlashAttention and FlashInfer wheels.

## Data

The evaluation code targets the Hugging Face dataset `stanfordnlp/wikitablequestions`, config `random-split-1`.

The default configs point to a fixed 500-example test subset:

```text
datasets/wikitablequestions/random-split-1_test_seed42_sample500.jsonl
```

This subset was generated from the WTQ test split by shuffling with Python `random.Random(42)` and taking the first 500 records. It keeps experiments fast and reproducible.

To regenerate preprocessed JSONL files:

```bash
python eval/preprocess_wikitablequestions.py --config-file eval/config_eval.yaml
```

## Running Evaluation

Set `model.path` in the YAML config to a local Qwen checkpoint:

```yaml
model:
  path: /path/to/Qwen3-8B
```

Run the full KV baseline:

```bash
python eval/test_wiki.py --config-file eval/config_eval_fullkv.yaml
```

Run compression methods:

```bash
python eval/test_wiki.py --config-file eval/config_eval_rkv_new.yaml
python eval/test_wiki.py --config-file eval/config_eval_snapkv_new.yaml
python eval/test_wiki.py --config-file eval/config_eval_question_new.yaml
python eval/test_wiki.py --config-file eval/config_eval_sliding.yaml
python eval/test_wiki.py --config-file eval/config_eval_tabula.yaml
```

The strict-budget configs use:

```yaml
model:
  layer_budget: 128
  query_window_size: 32
  strict_prefill_chunk_size: 64
  protected_kv_cache_size: 256
```

The few-shot prefix is protected and does not consume the current sample's compression budget.

## Compression Methods

`none` keeps the full KV Cache.

`sliding_new` performs strict-budget chunked prefill and keeps the most recent KV tokens.

`rkv_new` and `snapkv_new` are strict-budget versions of RKV and SnapKV.

`question_new` separately encodes the real question and uses question-to-prompt cross-attention to select KV tokens during chunked prefill.

`tabula` extends question-aware compression with table structure. It maps header and body cells to token-level metadata, guarantees at least one retained token per header cell, and adjusts body-cell scores with the attention score of the corresponding column header:

```text
S = lambda * S_cell + (1 - lambda) * S_header
```

`model.tabula_lambda` controls this balance.

## Logs

Evaluation outputs are written to `logs/wiki_eval/`:

```text
*.correct.jsonl   per-example correctness and predictions
*.qa.jsonl        question, raw output, cleaned prediction, answers
*.summary.json    aggregate accuracy and output paths
```

Logs are generated artifacts and are ignored by git.

## Documentation

- `docs/cache_manager_usage.md`: Cache Manager and compressor interface.
- `docs/question_aware_kv.md`: question-aware compression notes.
- `docs/tabula_kv.md`: TabulaKV algorithm details.

## Notes

Current compression experiments are designed for single-request evaluation:

```yaml
generation:
  batch_size: 1
model:
  max_num_seqs: 1
  kvcache_block_size: 1
```

TabulaKV uses `enforce_eager: true` by default because its current implementation is easier to debug in eager mode. Other configs keep `enforce_eager: false` for faster evaluation.
