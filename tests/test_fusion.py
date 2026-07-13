from ragsys.models import Chunk, RetrievedChunk
from ragsys.retrieval.hybrid import reciprocal_rank_fusion


def _rc(cid: str, rank: int, dense=True) -> RetrievedChunk:
    chunk = Chunk(chunk_id=cid, doc_id="d", title="t", source="SEBI", text=cid)
    return RetrievedChunk(
        chunk=chunk,
        dense_rank=rank if dense else None,
        sparse_rank=None if dense else rank,
    )


def test_rrf_rewards_agreement_across_retrievers():
    dense = [_rc("a", 0), _rc("b", 1), _rc("c", 2)]
    sparse = [_rc("b", 0, dense=False), _rc("a", 1, dense=False), _rc("d", 2, dense=False)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [r.chunk.chunk_id for r in fused]
    # a and b appear in both lists near the top -> should outrank c and d
    assert set(ids[:2]) == {"a", "b"}
    assert "d" in ids and "c" in ids


def test_rrf_merges_rank_metadata():
    # "a" is first (position 0) in dense, and 4th (position 3) in sparse.
    dense = [_rc("a", 0)]
    sparse = [_rc("x", 0, dense=False), _rc("y", 1, dense=False),
              _rc("z", 2, dense=False), _rc("a", 3, dense=False)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    rc = next(r for r in fused if r.chunk.chunk_id == "a")
    assert rc.dense_rank == 0 and rc.sparse_rank == 3
    # RRF uses list position: 1/(60+0+1) from dense + 1/(60+3+1) from sparse
    assert rc.score == 1 / 61 + 1 / 64


def test_rrf_top_k_truncates():
    lists = [[_rc(str(i), i) for i in range(10)]]
    assert len(reciprocal_rank_fusion(lists, top_k=3)) == 3
