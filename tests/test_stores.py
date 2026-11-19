import pytest

from ragsys.ingest.pipeline import ingest_documents
from ragsys.models import Chunk, Document
from ragsys.retrieval.hybrid import HybridRetriever
from ragsys.store import BM25Store, InMemoryVectorStore


def _chunk(cid: str, doc: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc, title="T", source="RBI", text=text)


def test_memory_store_round_trips_through_disk(tmp_path):
    a = InMemoryVectorStore(index_dir=tmp_path)
    a.upsert([_chunk("d1::0", "d1", "alpha"), _chunk("d1::1", "d1", "beta")],
             [[1.0, 0.0], [0.0, 1.0]])
    assert {p.name for p in tmp_path.iterdir()} == {"chunks.jsonl", "vectors.npy"}

    b = InMemoryVectorStore(index_dir=tmp_path)  # a fresh process would do this
    assert b.count() == 2
    top = b.search([0.9, 0.1], top_k=1)[0]
    assert top.chunk.text == "alpha" and top.dense_rank == 1
    assert top.semantic_score == pytest.approx(0.9, abs=1e-6)


def test_memory_store_search_is_ranked_and_bounded(tmp_path):
    s = InMemoryVectorStore(index_dir=tmp_path)
    s.upsert([_chunk(f"d::{i}", "d", str(i)) for i in range(4)],
             [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]])
    hits = s.search([1.0, 0.0], top_k=10)  # top_k larger than the store
    assert [h.chunk.text for h in hits] == ["0", "1", "2", "3"]
    assert [h.dense_rank for h in hits] == [1, 2, 3, 4]
    assert InMemoryVectorStore(index_dir=tmp_path / "empty").search([1.0, 0.0], 5) == []


def test_delete_docs_removes_only_that_document(tmp_path):
    s = InMemoryVectorStore(index_dir=tmp_path)
    s.upsert([_chunk("a::0", "a", "x"), _chunk("b::0", "b", "y")], [[1.0, 0.0], [0.0, 1.0]])
    s.delete_docs(["a"])
    assert [c.doc_id for c in s.all_chunks()] == ["b"]
    assert InMemoryVectorStore(index_dir=tmp_path).count() == 1  # persisted


def test_mismatched_lengths_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        InMemoryVectorStore(index_dir=tmp_path).upsert([_chunk("a::0", "a", "x")], [])


def test_inconsistent_index_files_fail_loudly(tmp_path):
    s = InMemoryVectorStore(index_dir=tmp_path)
    s.upsert([_chunk("a::0", "a", "x"), _chunk("a::1", "a", "y")], [[1.0, 0.0], [0.0, 1.0]])
    lines = (tmp_path / "chunks.jsonl").read_text().splitlines()
    (tmp_path / "chunks.jsonl").write_text(lines[0] + "\n")
    with pytest.raises(RuntimeError, match="inconsistent"):
        InMemoryVectorStore(index_dir=tmp_path)


def test_bm25_ranks_lexical_match_first_and_skips_non_matches():
    bm = BM25Store([
        _chunk("a::0", "a", "credit card closure within seven working days"),
        _chunk("b::0", "b", "payment aggregator net worth requirement"),
        _chunk("c::0", "c", "small account balance limits for deposits"),
    ])
    hits = bm.search("credit card closure", top_k=5)
    assert hits[0].chunk.chunk_id == "a::0" and hits[0].sparse_rank == 1
    assert all(h.score > 0 for h in hits)
    assert bm.search("zebra xylophone", top_k=5) == []


def test_bm25_searches_the_document_title_too():
    c = Chunk(chunk_id="a::0", doc_id="a", title="Internal Ombudsman Directions",
              source="RBI", text="The tenure shall be fixed.")
    bm = BM25Store([c, _chunk("b::0", "b", "unrelated text"), _chunk("c::0", "c", "more text")])
    assert bm.search("ombudsman", top_k=3)[0].chunk.chunk_id == "a::0"


def test_documents_ingested_after_startup_are_searchable(tmp_path, embedder):
    """Regression: the retriever used to hold a private copy of the index, so
    anything ingested after the first query was invisible until a restart."""
    vs = InMemoryVectorStore(index_dir=tmp_path)
    bm = BM25Store()
    retriever = HybridRetriever(vector_store=vs, bm25_store=bm, embedder=embedder)
    assert retriever.retrieve("credit card closure") == []  # first query, empty index

    docs = [
        Document(doc_id="cards", title="Card Directions", source="RBI",
                 text="19. Any request for closure of a credit card shall be honoured "
                      "within seven working days."),
        # BM25 needs a few documents before a term's IDF is meaningful.
        Document(doc_id="pa", title="Payment Aggregators", source="RBI",
                 text="6. A payment aggregator shall maintain a minimum net worth."),
        Document(doc_id="kyc", title="KYC", source="RBI",
                 text="3. Periodic updation applies to low risk customers."),
    ]
    ingest_documents(docs, vector_store=vs, bm25_store=bm, embedder=embedder)

    hits = retriever.retrieve("credit card closure")
    assert hits and hits[0].chunk.doc_id == "cards"
    assert hits[0].dense_rank == 1 and hits[0].sparse_rank == 1  # found by both retrievers


def test_reingesting_a_shorter_document_leaves_no_orphans(tmp_path, embedder):
    vs, bm = InMemoryVectorStore(index_dir=tmp_path), BM25Store()
    long = Document(doc_id="d", title="T", source="RBI",
                    text="\n".join(f"{i}. Clause {i} " + "word " * 60 for i in range(1, 9)))
    short = Document(doc_id="d", title="T", source="RBI", text="1. Only clause now.")
    first = ingest_documents([long], vector_store=vs, bm25_store=bm, embedder=embedder)
    second = ingest_documents([short], vector_store=vs, bm25_store=bm, embedder=embedder)
    assert first["vector_count"] > 1
    assert second["vector_count"] == 1 == second["bm25_count"]
    assert vs.all_chunks()[0].text == "1. Only clause now."


def test_seed_copies_file_index_into_an_empty_store_once(tmp_path):
    from ragsys.store import seed_from_index

    source = InMemoryVectorStore(index_dir=tmp_path / "baked")
    source.upsert([_chunk("a::0", "a", "x"), _chunk("a::1", "a", "y")], [[1.0, 0.0], [0.0, 1.0]])

    class FakePg(InMemoryVectorStore):
        backend = "pgvector"

    target = FakePg(index_dir=tmp_path / "pg")
    assert seed_from_index(tmp_path / "baked", store=target) == 2
    assert target.search([0.0, 1.0], top_k=1)[0].chunk.text == "y"  # vectors came across
    assert seed_from_index(tmp_path / "baked", store=target) == 0   # already has data
    assert seed_from_index(tmp_path / "baked", store=source) == 0   # memory backend: nothing to do


def test_upsert_replaces_an_existing_chunk_in_place(tmp_path):
    s = InMemoryVectorStore(index_dir=tmp_path)
    s.upsert([_chunk("a::0", "a", "old"), _chunk("a::1", "a", "other")], [[1.0, 0.0], [0.0, 1.0]])
    s.upsert([_chunk("a::0", "a", "new")], [[0.0, 1.0]])
    assert s.count() == 2
    assert {c.text for c in s.all_chunks()} == {"new", "other"}
    assert {h.chunk.text for h in s.search([0.0, 1.0], top_k=2) if h.semantic_score > 0.9} == {"new", "other"}
    assert InMemoryVectorStore(index_dir=tmp_path).count() == 2


def test_available_cpus_honours_container_quota(monkeypatch):
    import ragsys.runtime as runtime

    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda _pid: set(range(8)), raising=False)
    monkeypatch.setattr(runtime, "_cgroup_cpu_quota", lambda: 2.0)
    assert runtime.available_cpus() == 2          # 2-CPU container on an 8-core host
    monkeypatch.setattr(runtime, "_cgroup_cpu_quota", lambda: 0.5)
    assert runtime.available_cpus() == 1          # fractional quota still gets one thread
    monkeypatch.setattr(runtime, "_cgroup_cpu_quota", lambda: None)
    assert runtime.available_cpus() == 8          # no limit
