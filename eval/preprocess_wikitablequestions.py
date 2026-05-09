#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from wiki_table_utils import (
    DEFAULT_CONFIG,
    DEFAULT_REVISION,
    SPLIT_NAMES,
    get_nested,
    iter_records,
    load_eval_config,
    load_wikitablequestions,
    write_jsonl,
)


DEFAULT_CONFIG_FILE = SCRIPT_DIR / "config_eval.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess stanfordnlp/wikitablequestions into JSONL records whose "
            "tables are list-of-lists."
        )
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="YAML config file for preprocessing and evaluation.",
    )
    return parser.parse_args()


def preprocess_split(args: argparse.Namespace, split: str) -> Path:
    config = args.config
    dataset = load_wikitablequestions(
        split=split,
        cache_dir=config["cache_dir"],
        config=config["dataset_config"],
        revision=config["revision"],
    )
    output_path = config["output_dir"] / f"{config['dataset_config']}_{split}.jsonl"
    count = write_jsonl(output_path, iter_records(dataset, split=split, limit=config["limit"]))
    print(f"Wrote {count} records to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    config_file = args.config_file
    raw_config = load_eval_config(config_file)
    split = get_nested(raw_config, "dataset.split", "validation")
    if split not in SPLIT_NAMES + ("all",):
        raise ValueError(f"dataset.split must be one of {SPLIT_NAMES + ('all',)}, got {split!r}")
    args.config = {
        "dataset_config": get_nested(raw_config, "dataset.config", DEFAULT_CONFIG),
        "revision": get_nested(raw_config, "dataset.revision", DEFAULT_REVISION),
        "cache_dir": (
            Path(cache_dir) if (cache_dir := get_nested(raw_config, "dataset.cache_dir")) else None
        ),
        "output_dir": Path(get_nested(raw_config, "preprocess.output_dir", "datasets/wikitablequestions")),
        "limit": get_nested(raw_config, "preprocess.limit"),
    }
    splits = SPLIT_NAMES if split == "all" else (split,)
    for split in splits:
        preprocess_split(args, split)


if __name__ == "__main__":
    main()
