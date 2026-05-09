from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Iterable

import yaml


DATASET_NAME = "stanfordnlp/wikitablequestions"
DEFAULT_CONFIG = "random-split-1"
DEFAULT_REVISION = "20a01e1d62a85afcfd00e6112ce158d6e93e2f04"
SPLIT_NAMES = ("train", "validation", "test")


def load_eval_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def get_nested(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def parquet_priority(repo_path: str, split: str) -> tuple[int, str]:
    if repo_path == f"{split}/0000.parquet":
        return (0, repo_path)
    if repo_path == f"wikitablequestions-{split}.parquet":
        return (1, repo_path)
    if repo_path == f"{split}-00000-of-00001.parquet":
        return (2, repo_path)
    if repo_path.startswith(f"{split}/") and repo_path.endswith(".parquet"):
        return (3, repo_path)
    if repo_path.endswith(".parquet"):
        return (4, repo_path)
    return (99, repo_path)


def load_wikitablequestions(
    split: str | None,
    cache_dir: Path | None,
    config: str = DEFAULT_CONFIG,
    revision: str = DEFAULT_REVISION,
):
    """Load WikiTableQuestions from Parquet files in the Hugging Face dataset repo."""
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    entries = list(
        api.list_repo_tree(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            path_in_repo=config,
            recursive=True,
            revision=revision,
        )
    )

    parquet_paths = [
        entry.path[len(config) + 1 :]
        for entry in entries
        if hasattr(entry, "path")
        and entry.path.startswith(f"{config}/")
        and entry.path.endswith(".parquet")
    ]

    selected_splits = (split,) if split else SPLIT_NAMES
    data_files: dict[str, str] = {}
    for split_name in selected_splits:
        matches = [
            repo_path
            for repo_path in parquet_paths
            if repo_path == f"wikitablequestions-{split_name}.parquet"
            or repo_path == f"{split_name}-00000-of-00001.parquet"
            or repo_path == f"{split_name}/0000.parquet"
            or repo_path.startswith(f"{split_name}/")
        ]
        if not matches:
            available = ", ".join(sorted(parquet_paths)) or "none"
            raise FileNotFoundError(
                f"No parquet file found for split '{split_name}' under config "
                f"'{config}' at revision '{revision}'. Available parquet paths: {available}"
            )

        repo_path = sorted(matches, key=lambda path: parquet_priority(path, split_name))[0]
        data_files[split_name] = hf_hub_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            filename=f"{config}/{repo_path}",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    kwargs: dict[str, Any] = {"path": "parquet", "data_files": data_files}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    if split:
        kwargs["split"] = split
    return load_dataset(**kwargs)


def coerce_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def table_to_list_of_lists(table: dict[str, Any]) -> list[list[Any]]:
    header = [coerce_cell(cell) for cell in table.get("header", [])]
    rows = table.get("rows") or []
    normalized_rows: list[list[Any]] = []
    for row in rows:
        row_values = [coerce_cell(cell) for cell in list(row)]
        if header and len(row_values) < len(header):
            row_values.extend([""] * (len(header) - len(row_values)))
        normalized_rows.append(row_values)
    return [header, *normalized_rows]


def convert_example(example: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    table = example.get("table") or {}
    answers = example.get("answers") or []
    if isinstance(answers, str):
        answers = [answers]
    return {
        "id": example.get("id"),
        "split": split,
        "index": index,
        "question": example.get("question", ""),
        "answers": [str(answer) for answer in answers],
        "table": table_to_list_of_lists(table),
        "table_name": table.get("name"),
    }


def iter_records(dataset: Any, split: str, limit: int | None = None) -> Iterable[dict[str, Any]]:
    count = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(count):
        yield convert_example(dataset[index], split=split, index=index)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_prompt(record: dict[str, Any], max_rows: int | None = None) -> str:
    table = record["table"]
    if max_rows is not None and max_rows > 0:
        table = table[: max_rows + 1]
    table_text = json.dumps(table, ensure_ascii=False)
    return f"""
Table:[["Name", "age", "sex"], ["John", 20, "Male"], ["Li", 19, "Female"], ["Zhang", 21, "Male"]]
Question: Who is male?
Answer: John, Zhang
---
Table: [['Rank', 'Nation', 'Gold', 'Total'], ['1', 'China', '38', '88'], ['2', 'USA', '39', '113'], ['3', 'Japan', '27', '58']]
Question: How many nations have more than 30 gold medals?
Answer: 2
---
Table: [['Year', 'City', 'Visitors'], ['2021', 'Tokyo', '1500'], ['2022', 'Paris', '2300'], ['2023', 'London', '1900']]
Question: Which city had the most visitors?
Answer: Paris
---
Table:{table_text}
Question: {record['question']}
Answer: """


def clean_prediction(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    text = lines[0] if lines else ""
    text = text.strip().strip("\"'`")
    return text


def normalize_answer(text: Any) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[-\u2013\u2014]{3,}", " ", text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_prediction_items(text: str) -> list[str]:
    text = clean_prediction(text)
    parts = re.split(r"\s*(?:,|;|\n|\band\b)\s*", text)
    normalized = [normalize_answer(part) for part in parts if normalize_answer(part)]
    return normalized or [normalize_answer(text)]


def is_correct_prediction(prediction: str, answers: list[str]) -> bool:
    gold = [normalize_answer(answer) for answer in answers if normalize_answer(answer)]
    if not gold:
        return normalize_answer(prediction) == ""

    pred = normalize_answer(clean_prediction(prediction))
    if pred in gold:
        return True

    pred_items = split_prediction_items(prediction)
    if len(gold) > 1 and set(pred_items) == set(gold):
        return True

    return False
