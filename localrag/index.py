"""BM25 index over text chunks, with JSON persistence."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .tokenize import tokenize


@dataclass
class Chunk:
    source: str
    position: int
    text: str


@dataclass
class SearchResult:
    score: float
    chunk: Chunk


def chunk_text(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Split text into word-window chunks with overlap."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


@dataclass
class Index:
    k1: float = 1.5
    b: float = 0.75
    chunks: list[Chunk] = field(default_factory=list)
    _tf: list[Counter] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _lengths: list[int] = field(default_factory=list, repr=False)

    def add_document(self, source: str, text: str, size: int = 120, overlap: int = 20) -> int:
        """Chunk and index a document. Returns the number of chunks added."""
        added = 0
        for i, piece in enumerate(chunk_text(text, size, overlap)):
            tokens = tokenize(piece)
            if not tokens:
                continue
            tf = Counter(tokens)
            self.chunks.append(Chunk(source, i, piece))
            self._tf.append(tf)
            self._lengths.append(len(tokens))
            for term in tf:
                self._df[term] += 1
            added += 1
        return added

    def add_directory(self, root: str | Path, suffixes=(".txt", ".md", ".rst")) -> int:
        total = 0
        for path in sorted(Path(root).rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                total += self.add_document(str(path), path.read_text(encoding="utf-8", errors="ignore"))
        return total

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        avg_len = sum(self._lengths) / len(self._lengths)
        scored = []
        for idx, tf in enumerate(self._tf):
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self._lengths[idx] / avg_len)
                score += self._idf(term) * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append(SearchResult(score, self.chunks[idx]))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def save(self, path: str | Path) -> None:
        payload = {"k1": self.k1, "b": self.b, "chunks": [asdict(c) for c in self.chunks]}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(k1=payload["k1"], b=payload["b"])
        for c in payload["chunks"]:
            chunk = Chunk(**c)
            tokens = tokenize(chunk.text)
            tf = Counter(tokens)
            index.chunks.append(chunk)
            index._tf.append(tf)
            index._lengths.append(len(tokens))
            for term in tf:
                index._df[term] += 1
        return index
