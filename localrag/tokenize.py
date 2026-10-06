"""Tokenization helpers."""

import re

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those not but if then so such can into out
    up about over after before""".split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase, split on non-alphanumerics, optionally drop stopwords."""
    tokens = _WORD.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
