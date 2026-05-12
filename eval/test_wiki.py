#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wiki_table_utils import (
    DEFAULT_CONFIG,
    DEFAULT_REVISION,
    SPLIT_NAMES,
    build_prompt,
    build_prompt_with_spans,
    clean_prediction,
    get_nested,
    is_correct_prediction,
    iter_records,
    load_eval_config,
    load_wikitablequestions,
    read_jsonl,
)

DEFAULT_CONFIG_FILE = SCRIPT_DIR / "config_eval.yaml"


def resolve_repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TabulaKV/artifact_infer on WikiTableQuestions."
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="YAML config file for WikiTableQuestions evaluation.",
    )
    return parser.parse_args()


def normalize_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    data_file = get_nested(raw_config, "dataset.data_file")
    cache_dir = get_nested(raw_config, "dataset.cache_dir")
    max_table_rows = int(get_nested(raw_config, "prompt.max_table_rows", 0) or 0)
    output_dir = get_nested(raw_config, "logging.output_dir", "logs/wiki_eval")
    return {
        "model": {
            "path": get_nested(raw_config, "model.path", "qwen-8B"),
            "tensor_parallel_size": int(get_nested(raw_config, "model.tensor_parallel_size", 1)),
            "enforce_eager": bool(get_nested(raw_config, "model.enforce_eager", False)),
            "max_num_batched_tokens": int(
                get_nested(raw_config, "model.max_num_batched_tokens", 262144)
            ),
            "max_num_seqs": int(get_nested(raw_config, "model.max_num_seqs", 128)),
            "max_model_len": int(get_nested(raw_config, "model.max_model_len", 32768)),
            "gpu_memory_utilization": float(
                get_nested(raw_config, "model.gpu_memory_utilization", 0.7)
            ),
            "kvcache_block_size": int(get_nested(raw_config, "model.kvcache_block_size", 1)),
            "query_window_size": int(get_nested(raw_config, "model.query_window_size", 64)),
            "question_window_size": int(get_nested(raw_config, "model.question_window_size", 64)),
            "layer_budget": int(get_nested(raw_config, "model.layer_budget", 320)),
            "cache_compressor": get_nested(raw_config, "model.cache_compressor", "none"),
            "strict_prefill_chunk_size": int(
                get_nested(raw_config, "model.strict_prefill_chunk_size", 64)
            ),
            "protected_kv_cache_size": int(
                get_nested(raw_config, "model.protected_kv_cache_size", 256)
            ),
            "tabula_lambda": float(get_nested(raw_config, "model.tabula_lambda", 0.5)),
            "steps_between_cache_compressions": int(
                get_nested(raw_config, "model.steps_between_cache_compressions", 1)
            ),
        },
        "dataset": {
            "data_file": resolve_repo_path(data_file) if data_file else None,
            "config": get_nested(raw_config, "dataset.config", DEFAULT_CONFIG),
            "split": get_nested(raw_config, "dataset.split", "validation"),
            "revision": get_nested(raw_config, "dataset.revision", DEFAULT_REVISION),
            "cache_dir": resolve_repo_path(cache_dir) if cache_dir else None,
            "limit": get_nested(raw_config, "dataset.limit"),
        },
        "generation": {
            "batch_size": int(get_nested(raw_config, "generation.batch_size", 1)),
            "max_tokens": int(get_nested(raw_config, "generation.max_tokens", 32)),
            "temperature": float(get_nested(raw_config, "generation.temperature", -1.0)),
            "top_k": int(get_nested(raw_config, "generation.top_k", 1)),
            "top_p": float(get_nested(raw_config, "generation.top_p", 1.0)),
            "min_p": float(get_nested(raw_config, "generation.min_p", 0.0)),
            "ignore_eos": bool(get_nested(raw_config, "generation.ignore_eos", False)),
        },
        "prompt": {
            "max_table_rows": max_table_rows if max_table_rows > 0 else None,
        },
        "logging": {
            "output_dir": resolve_repo_path(output_dir),
            "run_name": get_nested(raw_config, "logging.run_name"),
            "use_tqdm": bool(get_nested(raw_config, "logging.use_tqdm", True)),
        },
    }


def load_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_config = config["dataset"]
    if dataset_config["data_file"]:
        records = read_jsonl(dataset_config["data_file"])
    else:
        dataset = load_wikitablequestions(
            split=dataset_config["split"],
            cache_dir=dataset_config["cache_dir"],
            config=dataset_config["config"],
            revision=dataset_config["revision"],
        )
        records = list(iter_records(dataset, split=dataset_config["split"]))

    if dataset_config["limit"] is not None:
        records = records[: dataset_config["limit"]]
    return records


def batched(records: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def default_run_name(model_path: str | Path, split: str) -> str:
    model_name = Path(model_path).name or "model"
    normalized_model = re.sub(r"[^0-9A-Za-z]+", "_", model_name).strip("_").lower()
    normalized_split = re.sub(r"[^0-9A-Za-z]+", "_", split).strip("_").lower()
    return f"{normalized_model}_wtq_{normalized_split}"


def make_log_paths(output_dir: Path, run_name: str | None) -> tuple[Path, Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    if not run_name:
        run_name = "wiki"
    correct_log = run_dir / f"{run_name}.correct.jsonl"
    qa_log = run_dir / f"{run_name}.qa.jsonl"
    summary_log = run_dir / f"{run_name}.summary.json"
    return run_dir, correct_log, qa_log, summary_log


def build_prompt_input(llm, record: dict[str, Any], max_rows: int | None = None) -> dict[str, Any]:
    prompt_info = build_prompt_with_spans(record, max_rows=max_rows)
    prompt = prompt_info["prompt"]
    question_start, question_end = prompt_info["question_char_span"]
    protected_start, protected_end = prompt_info.get("protected_char_span", (0, 0))

    encoded = llm.tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    full_token_ids = llm.tokenizer.encode(prompt)
    offsets = encoded["offset_mapping"]

    prefix_token_ids = llm.tokenizer.encode(prompt[:question_start])
    question_token_ids = llm.tokenizer.encode(prompt[question_start:question_end])
    protected_token_ids = llm.tokenizer.encode(prompt[protected_start:protected_end])
    span_start = len(prefix_token_ids)
    span_end = span_start + len(question_token_ids)
    protected_span_start = len(llm.tokenizer.encode(prompt[:protected_start]))
    protected_span_end = protected_span_start + len(protected_token_ids)

    token_count = len(full_token_ids)
    header_columns = [-1] * token_count
    header_cell_ids = [-1] * token_count
    cell_columns = [-1] * token_count

    def mark_tokens(
        char_spans: list[dict[str, Any]],
        labels: list[int],
        cell_id_labels: list[int] | None = None,
    ) -> None:
        for cell_id, span in enumerate(char_spans):
            start = span["start"]
            end = span["end"]
            column = int(span["column"])
            for token_index, (token_start, token_end) in enumerate(offsets):
                if token_index >= len(labels):
                    break
                if token_start == token_end:
                    continue
                if token_start < end and token_end > start:
                    labels[token_index] = column
                    if cell_id_labels is not None:
                        cell_id_labels[token_index] = cell_id

    mark_tokens(
        prompt_info.get("table_header_cell_char_spans", []),
        header_columns,
        header_cell_ids,
    )
    mark_tokens(prompt_info.get("table_body_cell_char_spans", []), cell_columns)

    return {
        "prompt": prompt,
        "token_ids": full_token_ids,
        "question_token_span": (span_start, span_end),
        "question_token_ids": question_token_ids,
        "protected_token_span": (protected_span_start, protected_span_end),
        "tabula_token_metadata": {
            "header_columns": header_columns,
            "header_cell_ids": header_cell_ids,
            "cell_columns": cell_columns,
        },
    }


def main() -> None:
    args = parse_args()
    raw_config = load_eval_config(args.config_file)
    config = normalize_config(raw_config)
    records = load_records(config)
    if not records:
        raise ValueError("No WikiTableQuestions records to evaluate.")

    from src.services.artifact_infer import LLM, SamplingParams

    model_config = config["model"]
    llm = LLM(
        model_config["path"],
        enforce_eager=model_config["enforce_eager"],
        tensor_parallel_size=model_config["tensor_parallel_size"],
        max_num_batched_tokens=model_config["max_num_batched_tokens"],
        max_num_seqs=model_config["max_num_seqs"],
        max_model_len=model_config["max_model_len"],
        gpu_memory_utilization=model_config["gpu_memory_utilization"],
        kvcache_block_size=model_config["kvcache_block_size"],
        query_window_size=model_config["query_window_size"],
        question_window_size=model_config["question_window_size"],
        layer_budget=model_config["layer_budget"],
        cache_compressor=model_config["cache_compressor"],
        strict_prefill_chunk_size=model_config["strict_prefill_chunk_size"],
        protected_kv_cache_size=model_config["protected_kv_cache_size"],
        tabula_lambda=model_config["tabula_lambda"],
        steps_between_cache_compressions=model_config["steps_between_cache_compressions"],
    )
    generation_config = config["generation"]
    sampling_params = SamplingParams(
        temperature=generation_config["temperature"],
        top_k=generation_config["top_k"],
        top_p=generation_config["top_p"],
        min_p=generation_config["min_p"],
        max_tokens=generation_config["max_tokens"],
        ignore_eos=generation_config["ignore_eos"],
    )

    logging_config = config["logging"]
    run_name = logging_config["run_name"] or default_run_name(
        model_config["path"], config["dataset"]["split"]
    )
    run_dir, correct_log, qa_log, summary_log = make_log_paths(
        logging_config["output_dir"], run_name
    )
    correct_count = 0
    total_count = 0
    max_rows = config["prompt"]["max_table_rows"]

    with correct_log.open("w", encoding="utf-8") as correct_f, qa_log.open(
        "w", encoding="utf-8"
    ) as qa_f:
        for batch_records in batched(records, max(generation_config["batch_size"], 1)):
            prompts = [build_prompt_input(llm, record, max_rows=max_rows) for record in batch_records]
            # print(prompts)
            outputs = llm.generate(prompts, sampling_params, use_tqdm=logging_config["use_tqdm"])

            for record, output in zip(batch_records, outputs):
                raw_output = output["text"]
                prediction = clean_prediction(raw_output)
                answers = record["answers"]
                correct = is_correct_prediction(prediction, answers)
                correct_count += int(correct)
                total_count += 1

                correct_f.write(
                    json.dumps(
                        {
                            "index": record.get("index"),
                            "id": record.get("id"),
                            "split": record.get("split"),
                            "correct": correct,
                            "prediction": prediction,
                            "answers": answers,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                qa_f.write(
                    json.dumps(
                        {
                            "index": record.get("index"),
                            "id": record.get("id"),
                            "split": record.get("split"),
                            "question": record["question"],
                            "output": raw_output,
                            "prediction": prediction,
                            "standard_answer": answers,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                correct_f.flush()
                qa_f.flush()

    accuracy = correct_count / total_count if total_count else 0.0
    summary = {
        "dataset": "stanfordnlp/wikitablequestions",
        "config": config["dataset"]["config"],
        "split": config["dataset"]["split"],
        "model": config["model"]["path"],
        "total": total_count,
        "correct": correct_count,
        "accuracy": accuracy,
        "log_dir": str(run_dir.resolve()),
        "correct_log": str(correct_log.resolve()),
        "qa_log": str(qa_log.resolve()),
        "summary_log": str(summary_log.resolve()),
        "config_file": str(args.config_file),
    }
    summary_log.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
