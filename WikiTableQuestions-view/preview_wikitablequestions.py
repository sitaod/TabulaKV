#!/usr/bin/env python3
"""Download WikiTableQuestions from Hugging Face and render an HTML preview."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download


DATASET_NAME = "stanfordnlp/wikitablequestions"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "preview" / "wikitablequestions_preview.html"
DEFAULT_CONFIG = "random-split-1"
# The dataset repo's main branch currently keeps only the legacy loader script.
# Use a known snapshot commit that contains the parquet exports for splits 1-5.
DEFAULT_REVISION = "20a01e1d62a85afcfd00e6112ce158d6e93e2f04"
SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download stanfordnlp/wikitablequestions from Hugging Face Parquet "
            "files and generate a static HTML preview."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        choices=tuple(f"random-split-{idx}" for idx in range(1, 6)),
        help="Dataset configuration to preview.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation", "test"),
        help="Dataset split to preview. Ignored when --all-splits is set.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Load train, validation, and test and include samples from each split.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of examples to render per selected split.",
    )
    parser.add_argument(
        "--table-rows",
        type=int,
        default=12,
        help="Maximum table rows shown for each example.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Dataset repo revision, branch, or tag to read from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="HTML preview output path.",
    )
    return parser.parse_args()


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


def load_wtq(
    split: str | None,
    cache_dir: Path | None,
    config: str,
    revision: str,
) -> Dataset | DatasetDict:
    """Load WTQ from Parquet files discovered in the dataset repo."""
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


def as_split_map(data: Dataset | DatasetDict, requested_split: str) -> dict[str, Dataset]:
    if isinstance(data, DatasetDict):
        return {name: split for name, split in data.items()}
    return {requested_split: data}


def cell(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def render_table(table: dict[str, Any], max_rows: int) -> str:
    header = table.get("header") or []
    rows = table.get("rows") or []
    visible_rows = rows[:max_rows]
    hidden_count = max(len(rows) - len(visible_rows), 0)

    thead = "".join(f"<th>{cell(column)}</th>" for column in header)
    body_rows = []
    for row in visible_rows:
        values = list(row)
        if len(values) < len(header):
            values.extend([""] * (len(header) - len(values)))
        body_rows.append(
            "<tr>" + "".join(f"<td>{cell(value)}</td>" for value in values) + "</tr>"
        )

    more = ""
    if hidden_count:
        colspan = max(len(header), 1)
        more = (
            f'<tr class="more"><td colspan="{colspan}">'
            f"{hidden_count} more rows hidden"
            "</td></tr>"
        )

    table_name = table.get("name") or "unknown table"
    return f"""
    <div class="table-meta">{cell(table_name)} · {len(rows)} rows</div>
    <div class="table-wrap">
      <table>
        <thead><tr>{thead}</tr></thead>
        <tbody>{''.join(body_rows)}{more}</tbody>
      </table>
    </div>
    """


def render_example(split: str, index: int, example: dict[str, Any], max_rows: int) -> str:
    answers = example.get("answers") or []
    answer_text = ", ".join(cell(answer) for answer in answers)
    table = example.get("table") or {}
    payload = json.dumps(example, ensure_ascii=False, indent=2)
    label = f"{split} #{index} · {example.get('id') or 'unknown'}"

    return f"""
    <article class="example" data-label="{cell(label)}">
      <div class="example-head">
        <span>{cell(split)} #{index}</span>
        <code>{cell(example.get("id"))}</code>
      </div>
      <h2>{cell(example.get("question"))}</h2>
      <p class="answer"><strong>Answer:</strong> {answer_text}</p>
      {render_table(table, max_rows)}
      <details>
        <summary>Raw JSON</summary>
        <pre>{cell(payload)}</pre>
      </details>
    </article>
    """


def iter_examples(dataset: Dataset, limit: int) -> Iterable[tuple[int, dict[str, Any]]]:
    for index in range(min(limit, len(dataset))):
        yield index, dataset[index]


def render_html(splits: dict[str, Dataset], samples: int, table_rows: int) -> str:
    example_items = [
        render_example(name, index, example, table_rows)
        for name, dataset in splits.items()
        for index, example in iter_examples(dataset, samples)
    ]
    split_cards = "".join(
        f"<li><strong>{cell(name)}</strong><span>{len(dataset):,} examples</span></li>"
        for name, dataset in splits.items()
    )
    examples = "".join(example_items)
    example_count = len(example_items)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WikiTableQuestions Preview</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #65727f;
      --line: #d8dee5;
      --panel: #ffffff;
      --page: #f5f7fa;
      --accent: #0f766e;
      --accent-soft: #d9f3ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--page);
      line-height: 1.5;
    }}
    header {{
      padding: 32px min(5vw, 56px) 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 4vw, 46px);
      letter-spacing: 0;
    }}
    header p {{
      max-width: 860px;
      margin: 0;
      color: var(--muted);
      font-size: 16px;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto 48px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      padding: 0;
      margin: 0 0 24px;
      list-style: none;
    }}
    .stats li {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .stats span {{ color: var(--muted); }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      margin: 0 0 18px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .toolbar-group {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .toolbar label {{
      font-size: 14px;
      color: var(--muted);
    }}
    .toolbar select,
    .toolbar button {{
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
    }}
    .toolbar select {{
      min-width: min(100%, 420px);
      padding: 0 12px;
    }}
    .toolbar button {{
      padding: 0 14px;
      cursor: pointer;
    }}
    .toolbar button:disabled {{
      opacity: 0.5;
      cursor: default;
    }}
    .toolbar-status {{
      color: var(--muted);
      font-size: 14px;
    }}
    .hint {{
      width: 100%;
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .example {{
      margin: 0;
      padding: 20px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .example[hidden] {{
      display: none;
    }}
    .example-head {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
    }}
    code {{
      padding: 2px 6px;
      border-radius: 5px;
      background: var(--accent-soft);
      color: #0b4d47;
      text-transform: none;
    }}
    h2 {{
      margin: 10px 0 8px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .answer {{
      margin: 0 0 16px;
      color: var(--accent);
    }}
    .table-meta {{
      margin: 12px 0 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      min-width: 680px;
      border-collapse: collapse;
      background: #fff;
    }}
    th, td {{
      padding: 9px 11px;
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eef2f5;
      font-weight: 650;
    }}
    th:last-child, td:last-child {{ border-right: 0; }}
    tr:last-child td {{ border-bottom: 0; }}
    .more td {{
      color: var(--muted);
      background: #fafbfc;
      text-align: center;
    }}
    details {{
      margin-top: 14px;
      color: var(--muted);
    }}
    summary {{ cursor: pointer; }}
    pre {{
      overflow: auto;
      padding: 14px;
      border-radius: 8px;
      background: #17212b;
      color: #e8edf2;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>WikiTableQuestions Preview</h1>
    <p>
      Samples from <strong>{DATASET_NAME}</strong>, rendered after downloading
      the dataset repo's Parquet exports with Hugging Face Datasets.
    </p>
  </header>
  <main>
    <ul class="stats">{split_cards}</ul>
    <section class="toolbar" aria-label="Question navigation">
      <div class="toolbar-group">
        <button type="button" id="prev-btn">Previous</button>
        <button type="button" id="next-btn">Next</button>
      </div>
      <div class="toolbar-group">
        <label for="question-select">Question</label>
        <select id="question-select"></select>
      </div>
      <div class="toolbar-status" id="question-status"></div>
      <p class="hint">This preview currently contains {example_count} rendered question(s).</p>
    </section>
    <section id="examples">{examples}</section>
  </main>
  <script>
    const examples = Array.from(document.querySelectorAll(".example"));
    const select = document.getElementById("question-select");
    const prevBtn = document.getElementById("prev-btn");
    const nextBtn = document.getElementById("next-btn");
    const status = document.getElementById("question-status");

    examples.forEach((example, index) => {{
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = example.dataset.label || `Question ${{index + 1}}`;
      select.appendChild(option);
    }});

    let currentIndex = 0;

    function renderCurrent(index) {{
      currentIndex = index;
      examples.forEach((example, exampleIndex) => {{
        example.hidden = exampleIndex !== currentIndex;
      }});
      select.value = String(currentIndex);
      status.textContent = examples.length
        ? `Showing ${{currentIndex + 1}} of ${{examples.length}}`
        : "No questions loaded";
      prevBtn.disabled = currentIndex <= 0;
      nextBtn.disabled = currentIndex >= examples.length - 1;
    }}

    select.addEventListener("change", () => {{
      renderCurrent(Number(select.value));
    }});
    prevBtn.addEventListener("click", () => {{
      if (currentIndex > 0) renderCurrent(currentIndex - 1);
    }});
    nextBtn.addEventListener("click", () => {{
      if (currentIndex < examples.length - 1) renderCurrent(currentIndex + 1);
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowLeft" && currentIndex > 0) renderCurrent(currentIndex - 1);
      if (event.key === "ArrowRight" && currentIndex < examples.length - 1) renderCurrent(currentIndex + 1);
    }});

    renderCurrent(0);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    selected_split = None if args.all_splits else args.split
    data = load_wtq(selected_split, args.cache_dir, args.config, args.revision)
    splits = as_split_map(data, args.split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_html(splits, args.samples, args.table_rows),
        encoding="utf-8",
    )
    print(f"Wrote preview: {args.output.resolve()}")


if __name__ == "__main__":
    main()
