# localrag-cli

A small, dependency-free command line tool for searching your own notes and documents with BM25 ranking. Everything runs locally: no API keys, no network, no vector database.

## Features

- Chunks documents into overlapping word windows
- BM25 ranking (tunable `k1` and `b`)
- Saves and loads the index as JSON
- Simple CLI with `build` and `query` commands

## Usage

```bash
python -m localrag.cli build docs_sample -o index.json
python -m localrag.cli query "how does bm25 rank documents" -i index.json -k 3
```

Or use it as a library:

```python
from localrag import Index

index = Index()
index.add_directory("docs_sample")
for result in index.search("retrieval ranking", k=3):
    print(result.score, result.chunk.source)
```

## Tests

```bash
pip install pytest
pytest
```

## Why BM25?

For small personal corpora, BM25 is fast, explainable, and a strong baseline to compare against embedding-based retrieval.
