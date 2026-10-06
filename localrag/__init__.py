"""localrag: a small, dependency-free local document search and retrieval toolkit."""

from .index import Index, Chunk, SearchResult
from .tokenize import tokenize

__all__ = ["Index", "Chunk", "SearchResult", "tokenize"]
__version__ = "0.1.0"
