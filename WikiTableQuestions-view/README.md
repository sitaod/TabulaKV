# WikiTableQuestions Preview

Download `stanfordnlp/wikitablequestions` from Hugging Face Parquet exports and
generate a static HTML preview with questions, answers, source tables, and raw
JSON.

```bash
cd WikiTableQuestions-view
python3 -m pip install -r requirements.txt
python3 preview_wikitablequestions.py
```

The default output is:

```text
WikiTableQuestions-view/preview/wikitablequestions_preview.html
```

Useful options:

```bash
python3 preview_wikitablequestions.py --split train --samples 10
python3 preview_wikitablequestions.py --all-splits --samples 3
python3 preview_wikitablequestions.py --config random-split-2
python3 preview_wikitablequestions.py --cache-dir ./hf_cache --output preview/all.html
```

The script discovers parquet files in the dataset repo through the Hugging Face
Hub API, downloads the matching split files into cache, and then loads them
locally. This avoids the deprecated loading script path that newer `datasets`
versions reject.

By default it uses a dataset snapshot revision that contains the parquet
exports, because the current `main` branch of this dataset does not expose the
`random-split-*` parquet directories.
