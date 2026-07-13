from ragsys.ingest.chunker import chunk_document
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
