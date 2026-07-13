"""Dense vector store.

Two interchangeable backends:
  * PgVectorStore     -> Postgres + pgvector (production path, used in Docker).
  * InMemoryVectorStore -> numpy cosine search persisted to a pickle
                           (zero-infra path for tests / laptop demos).

`get_vector_store()` returns pgvector if it can connect, else the in-memory
store, so the pipeline always works.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from ragsys.config import settings
from ragsys.models import Chunk, RetrievedChunk

_INDEX_DIR = Path("data/index")


class BaseVectorStore:
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def search(self, query_vec: list[float], top_k: int) -> list[RetrievedChunk]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def count(self) -> int: ...


# --------------------------------------------------------------------------- #
# In-memory backend                                                           #
# --------------------------------------------------------------------------- #
class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, path: str | Path = _INDEX_DIR / "vectors.pkl") -> None:
        self.path = Path(path)
        self._chunks: dict[str, Chunk] = {}
        self._vecs: dict[str, list[float]] = {}
        if self.path.exists():
            self.load()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors):
            self._chunks[chunk.chunk_id] = chunk
            self._vecs[chunk.chunk_id] = vec
        self.save()

    def search(self, query_vec: list[float], top_k: int) -> list[RetrievedChunk]:
        import numpy as np

        if not self._vecs:
            return []
        ids = list(self._vecs.keys())
        mat = np.array([self._vecs[i] for i in ids], dtype="float32")
        q = np.array(query_vec, dtype="float32")
        # vectors are L2-normalised at embed time, so dot == cosine
        sims = mat @ q
        order = np.argsort(-sims)[:top_k]
        out: list[RetrievedChunk] = []
        for rank, idx in enumerate(order):
            cid = ids[idx]
            out.append(
                RetrievedChunk(
                    chunk=self._chunks[cid],
                    score=float(sims[idx]),
                    semantic_score=float(sims[idx]),
                    dense_rank=rank + 1,
                )
            )
        return out

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def count(self) -> int:
        return len(self._chunks)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fh:
            pickle.dump(
                {"chunks": {k: v.model_dump() for k, v in self._chunks.items()},
                 "vecs": self._vecs},
                fh,
            )

    def load(self) -> None:
        with self.path.open("rb") as fh:
            data = pickle.load(fh)
        self._chunks = {k: Chunk(**v) for k, v in data["chunks"].items()}
        self._vecs = data["vecs"]


# --------------------------------------------------------------------------- #
# pgvector backend                                                            #
# --------------------------------------------------------------------------- #
class PgVectorStore(BaseVectorStore):
    def __init__(self, dsn: str | None = None, dim: int | None = None) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim = dim or settings.embedding_dim
        self.conn = psycopg.connect(dsn or settings.pg_dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id   TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                title      TEXT,
                source     TEXT,
                url        TEXT,
                ordinal    INT,
                text       TEXT NOT NULL,
                meta       JSONB DEFAULT '{{}}'::jsonb,
                embedding  vector({self.dim})
            );
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
            "ON chunks USING hnsw (embedding vector_cosine_ops);"
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        import numpy as np

        with self.conn.cursor() as cur:
            for chunk, vec in zip(chunks, vectors):
                cur.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, doc_id, title, source, url, ordinal, text, meta, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET text = EXCLUDED.text, embedding = EXCLUDED.embedding,
                        meta = EXCLUDED.meta;
                    """,
                    (
                        chunk.chunk_id, chunk.doc_id, chunk.title, chunk.source,
                        chunk.url, chunk.ordinal, chunk.text,
                        _json(chunk.meta), np.array(vec, dtype="float32"),
                    ),
                )

    def search(self, query_vec: list[float], top_k: int) -> list[RetrievedChunk]:
        import numpy as np

        rows = self.conn.execute(
            """
            SELECT chunk_id, doc_id, title, source, url, ordinal, text, meta,
                   1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (np.array(query_vec, dtype="float32"), np.array(query_vec, dtype="float32"), top_k),
        ).fetchall()
        out: list[RetrievedChunk] = []
        for rank, r in enumerate(rows):
            out.append(
                RetrievedChunk(
                    chunk=_row_to_chunk(r),
                    score=float(r[8]),
                    semantic_score=float(r[8]),
                    dense_rank=rank + 1,
                )
            )
        return out

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT chunk_id, doc_id, title, source, url, ordinal, text, meta FROM chunks;"
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]


def _json(obj) -> str:
    import json

    return json.dumps(obj, default=str)


def _row_to_chunk(r) -> Chunk:
    meta = r[7]
    if isinstance(meta, str):
        import json

        meta = json.loads(meta)
    return Chunk(
        chunk_id=r[0], doc_id=r[1], title=r[2], source=r[3], url=r[4],
        ordinal=r[5] or 0, text=r[6], meta=meta or {},
    )


def get_vector_store() -> BaseVectorStore:
    """Prefer pgvector; fall back to in-memory if Postgres is unreachable."""
    try:
        store = PgVectorStore()
        return store
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"[vector_store] pgvector unavailable ({exc}); using in-memory store")
        return InMemoryVectorStore()
