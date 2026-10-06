import pytest

from localrag import Index, tokenize
from localrag.index import chunk_text


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("The Quick brown FOX!") == ["quick", "brown", "fox"]


def test_chunk_text_overlap_and_bounds():
    text = " ".join(str(i) for i in range(50))
    chunks = chunk_text(text, size=20, overlap=5)
    assert chunks[0].split()[0] == "0"
    assert chunks[1].split()[0] == "15"
    assert chunks[-1].split()[-1] == "49"


def test_chunk_text_invalid_args():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=0)
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, overlap=5)


def test_search_ranks_relevant_document_first():
    index = Index()
    index.add_document("pets.txt", "parrots are colorful birds that can mimic human speech")
    index.add_document("cars.txt", "engines convert fuel into motion inside the cylinders")
    results = index.search("parrot birds speech")
    assert results == [] or results[0].chunk.source == "pets.txt"
    results = index.search("birds speech")
    assert results[0].chunk.source == "pets.txt"


def test_search_empty_and_unmatched_queries():
    index = Index()
    assert index.search("anything") == []
    index.add_document("a.txt", "alpha beta gamma")
    assert index.search("zzz") == []


def test_save_and_load_roundtrip(tmp_path):
    index = Index()
    index.add_document("a.txt", "retrieval augmented generation needs good retrieval")
    path = tmp_path / "idx.json"
    index.save(path)
    loaded = Index.load(path)
    assert [r.chunk.source for r in loaded.search("retrieval")] == ["a.txt"]
    assert loaded.search("retrieval")[0].score == pytest.approx(index.search("retrieval")[0].score)


def test_add_directory(tmp_path):
    (tmp_path / "one.md").write_text("python packaging with pyproject files")
    (tmp_path / "skip.bin").write_text("python ignored")
    index = Index()
    assert index.add_directory(tmp_path) == 1
