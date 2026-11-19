import re

from ragsys.ingest.chunker import _tail, chunk_document
from ragsys.models import Document


def _doc(text: str) -> Document:
    return Document(doc_id="d1", title="t", source="SEBI", text=text)


def test_chunker_respects_size():
    text = "\n\n".join(f"{i}. clause number {i} " + "word " * 40 for i in range(1, 20))
    chunks = chunk_document(_doc(text), size=300, overlap=50)
    assert chunks, "expected at least one chunk"
    assert all(len(c.text) <= 300 + 50 for c in chunks)
    # ordinals are sequential and ids unique
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunker_splits_on_numbered_clauses():
    text = "1. first clause here.\n2. second clause here.\n3. third clause here."
    chunks = chunk_document(_doc(text), size=25, overlap=5)
    joined = " ".join(c.text for c in chunks)
    assert "first" in joined and "third" in joined


def test_empty_document_yields_no_chunks():
    assert chunk_document(_doc("")) == []


def test_oversized_clause_is_never_cut_mid_word():
    words = [f"word{i:03d}" for i in range(400)]
    text = "1. " + " ".join(words) + "."
    chunks = chunk_document(_doc(text), size=200, overlap=40)
    assert len(chunks) > 1
    for c in chunks:
        for token in re.findall(r"word\S*", c.text):
            assert re.fullmatch(r"word\d{3}\.?", token), f"cut mid-word: {token!r}"


def test_long_unit_is_split_on_sentence_boundaries():
    sentences = [f"Sentence number {i} says something specific about the rule." for i in range(30)]
    chunks = chunk_document(_doc("1. " + " ".join(sentences)), size=250, overlap=0)
    assert len(chunks) > 1
    assert all(c.text.rstrip().endswith(".") for c in chunks)


def test_overlap_tail_starts_on_a_sentence_when_possible():
    text = "The first sentence is here. The second sentence follows it closely."
    assert _tail(text, 45) == "The second sentence follows it closely."


def test_no_content_is_lost():
    text = "\n".join(f"{i}. Clause {i} requires disclosure within {i} days." for i in range(1, 40))
    joined = " ".join(c.text for c in chunk_document(_doc(text), size=200, overlap=30))
    for i in range(1, 40):
        assert f"within {i} days" in joined
