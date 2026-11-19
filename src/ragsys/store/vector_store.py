"""Dense vector store.

Two interchangeable backends, selected explicitly with VECTOR_BACKEND:

  * InMemoryVectorStore -> numpy cosine search over two plain files in
                           INDEX_DIR (chunks.jsonl + vectors.npy). Zero infra,
                           and the files can be shipped inside the Docker image.
  * PgVectorStore       -> Postgres + pgvector with an HNSW index.

`get_vector_store()` returns one shared instance per process, so ingestion and
retrieval always see the same data. If pgvector is selected and unreachable we
raise instead of silently switching backend: a quiet fallback would serve an
empty or stale index and look like a retrieval bug.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ragsys.config import settings
from ragsys.models import Chunk, RetrievedChunk


class BaseVectorStore:
    backend = "base"

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def delete_docs(self, doc_ids: list[str]) -> None: ...
    def search(self, query_vec: list[float], top_k: int) -> list[RetrievedChunk]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def count(self) -> int: ...


# --------------------------------------------------------------------------- #
# In-memory backend                                                           #
# --------------------------------------------------------------------------- #
class InMemoryVectorStore(BaseVectorStore):
    """Chunks in a dict, vectors in one float32 matrix (row i belongs to _ids[i])."""

    backend = "memory"

    def __init__(self, index_dir: str | Path | None = None) -> None:
        self.dir = Path(index_dir or settings.index_dir)
        self._chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        self._matrix = None  # np.ndarray of shape (len(_ids), dim), or None when empty
        if self._chunks_path.exists() and self._vectors_path.exists():
            self.load()

    @property
    def _chunks_path(self) -> Path:
        return self.dir / "chunks.jsonl"

    @property
    def _vectors_path(self) -> Path:
        return self.dir / "vectors.npy"

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        import numpy as np

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        if not chunks:
            return
        incoming = {c.chunk_id: (c, v) for c, v in zip(chunks, vectors)}  # last one wins
        self._drop(set(incoming))  # replace any existing rows for these ids
        new_rows = np.array([v for _, v in incoming.values()], dtype="float32")
        self._matrix = new_rows if self._matrix is None else np.vstack([self._matrix, new_rows])
        self._ids.extend(incoming)
        self._chunks.update({cid: c for cid, (c, _) in incoming.items()})
        self.save()

    def delete_docs(self, doc_ids: list[str]) -> None:
        doomed = set(doc_ids)
        self._drop({cid for cid, ch in self._chunks.items() if ch.doc_id in doomed})
        self.save()

    def _drop(self, chunk_ids: set[str]) -> None:
        import numpy as np

        if not chunk_ids or not self._ids:
            return
        keep = [i for i, cid in enumerate(self._ids) if cid not in chunk_ids]
        if len(keep) == len(self._ids):
            return
        for cid in chunk_ids:
            self._chunks.pop(cid, None)
        self._ids = [self._ids[i] for i in keep]
        self._matrix = self._matrix[np.array(keep, dtype=int)] if keep else None

    def search(self, query_vec: list[float], top_k: int) -> list[RetrievedChunk]:
        import numpy as np

        if not self._ids:
            return []
        q = np.array(query_vec, dtype="float32")
        sims = self._matrix @ q  # vectors are L2-normalised at embed time, so dot == cosine
        k = min(top_k, len(self._ids))
        top = np.argpartition(-sims, k - 1)[:k]
        order = top[np.argsort(-sims[top])]
        return [
            RetrievedChunk(
                chunk=self._chunks[self._ids[idx]],
                score=float(sims[idx]),
                semantic_score=float(sims[idx]),
                dense_rank=rank + 1,
            )
            for rank, idx in enumerate(order)
        ]

    def vectors_for(self, chunk_ids: list[str]) -> list[list[float]]:
        row = {cid: i for i, cid in enumerate(self._ids)}
        return [self._matrix[row[cid]].tolist() for cid in chunk_ids]

    def all_chunks(self) -> list[Chunk]:
        return [self._chunks[cid] for cid in self._ids]

    def count(self) -> int:
        return len(self._ids)

    def save(self) -> None:
        import numpy as np

        self.dir.mkdir(parents=True, exist_ok=True)
        with self._chunks_path.open("w", encoding="utf-8") as fh:
            for cid in self._ids:
                fh.write(json.dumps(self._chunks[cid].model_dump(), ensure_ascii=False) + "\n")
        matrix = self._matrix if self._matrix is not None else np.zeros((0, 0), dtype="float32")
        with self._vectors_path.open("wb") as fh:
            np.save(fh, matrix)

    def load(self) -> None:
        import numpy as np

        chunks = [
            Chunk(**json.loads(line))
            for line in self._chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matrix = np.load(self._vectors_path)
        if len(chunks) != len(matrix):
            raise RuntimeError(
                f"index in {self.dir} is inconsistent: {len(chunks)} chunks vs "
                f"{len(matrix)} vectors. Rebuild it with `make ingest`."
            )
        self._chunks = {c.chunk_id: c for c in chunks}
        self._ids = [c.chunk_id for c in chunks]
        self._matrix = matrix.astype("float32", copy=False) if len(chunks) else None


# --------------------------------------------------------------------------- #
# pgvector backend                                                            #
# --------------------------------------------------------------------------- #
class PgVectorStore(BaseVectorStore):
    backend = "pgvector"

    def __init__(self, dsn: str | None = None, dim: int | None = None) -> None:
        self.dsn = dsn or settings.pg_dsn
        self.dim = dim or settings.embedding_dim
        self.conn = None
        self._connect()
        self._create_schema()

    def _connect(self) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(self.dsn, autocommit=True, connect_timeout=10)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def _execute(self, sql: str, params=None):
        """Run a statement, reconnecting once if the server dropped the connection
        (hosted free tiers suspend idle databases)."""
        import psycopg

        try:
            return self.conn.execute(sql, params)
        except psycopg.OperationalError:
            self._connect()
            return self.conn.execute(sql, params)

    def _create_schema(self) -> None:
        self._execute(
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
        self._execute("CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);")
        self._execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
            "ON chunks USING hnsw (embedding vector_cosine_ops);"
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        import numpy as np

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        rows = [
            (c.chunk_id, c.doc_id, c.title, c.source, c.url, c.ordinal, c.text,
             json.dumps(c.meta, default=str), np.array(v, dtype="float32"))
            for c, v in zip(chunks, vectors)
        ]
        self._execute("SELECT 1")  # make sure the connection is alive before the batch
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, doc_id, title, source, url, ordinal, text, meta, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_id) DO UPDATE
                SET doc_id = EXCLUDED.doc_id, title = EXCLUDED.title,
                    source = EXCLUDED.source, url = EXCLUDED.url,
                    ordinal = EXCLUDED.ordinal, text = EXCLUDED.text,
                    meta = EXCLUDED.meta, embedding = EXCLUDED.embedding;
                """,
                rows,
            )

    def delete_docs(self, doc_ids: list[str]) -> None:
        if doc_ids:
            self._execute("DELETE FROM chunks WHERE doc_id = ANY(%s);", (list(doc_ids),))

    def search(self, query_vec: list[float], top_k: int) -> list[RetrievedChunk]:
        import numpy as np

        q = np.array(query_vec, dtype="float32")
        rows = self._execute(
            """
            SELECT chunk_id, doc_id, title, source, url, ordinal, text, meta,
                   1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (q, q, top_k),
        ).fetchall()
        return [
            RetrievedChunk(
                chunk=_row_to_chunk(r),
                score=float(r[8]),
                semantic_score=float(r[8]),
                dense_rank=rank + 1,
            )
            for rank, r in enumerate(rows)
        ]

    def all_chunks(self) -> list[Chunk]:
        rows = self._execute(
            "SELECT chunk_id, doc_id, title, source, url, ordinal, text, meta "
            "FROM chunks ORDER BY doc_id, ordinal;"
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count(self) -> int:
        return self._execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]


def _row_to_chunk(r) -> Chunk:
    meta = r[7]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Chunk(
        chunk_id=r[0], doc_id=r[1], title=r[2] or "", source=r[3] or "", url=r[4],
        ordinal=r[5] or 0, text=r[6], meta=meta or {},
    )


@lru_cache
def get_vector_store() -> BaseVectorStore:
    """The process-wide vector store for the configured backend."""
    if settings.vector_backend == "pgvector":
        return PgVectorStore()
    return InMemoryVectorStore()


def seed_from_index(index_dir: str | Path | None = None, store: BaseVectorStore | None = None) -> int:
    """Copy the file-based index into an empty vector store, without re-embedding.

    This is how a fresh Postgres gets its data: the image already carries the
    chunks and their vectors in INDEX_DIR, so there is nothing to recompute.
    Returns the number of chunks copied (0 if the store already has data or
    there is no file index to copy from).
    """
    store = store or get_vector_store()
    if store.backend == "memory" or store.count() > 0:
        return 0
    source = InMemoryVectorStore(index_dir)
    chunks = source.all_chunks()
    if not chunks:
        return 0
    batch = 200
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        store.upsert(part, source.vectors_for([c.chunk_id for c in part]))
    return len(chunks)
