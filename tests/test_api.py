"""HTTP-level tests: routing, validation, rate limiting and admin gating."""
import pytest
from fastapi.testclient import TestClient

import ragsys.agent
import ragsys.api.security as security
import ragsys.store
from ragsys.api.app import app
from ragsys.api.security import RateLimiter
from ragsys.config import settings
from ragsys.models import QueryResponse


class FakePipeline:
    def __init__(self):
        self.calls = []

    def answer(self, question, top_k=None, rerank_top_n=None, provider=None):
        self.calls.append({"question": question, "top_k": top_k,
                           "rerank_top_n": rerank_top_n, "provider": provider})
        return QueryResponse(answer="Seven working days [1].", provider="groq", model="m",
                             latency_ms=1.0, retrieval_ms=0.5, generation_ms=0.5)


@pytest.fixture
def pipeline(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(ragsys.agent, "get_pipeline", lambda: fake)
    return fake


@pytest.fixture
def client(monkeypatch, stores):
    vs, bm = stores
    monkeypatch.setattr(ragsys.store, "get_vector_store", lambda: vs)
    monkeypatch.setattr(ragsys.store, "get_bm25_store", lambda: bm)
    # A generous limiter by default; rate-limit tests install their own.
    monkeypatch.setattr(security, "limiter", RateLimiter(1000, 1000, 1000))
    with TestClient(app) as c:
        yield c


def test_ui_is_served_at_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Sentinel" in resp.text


def test_health_reports_index_and_answer_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["vector_backend"] == "memory"
    assert body["vector_chunks"] == body["bm25_chunks"] >= 1
    assert body["llm_configured"] is False and body["answer_mode"] == "extractive"


def test_meta_lists_documents_with_links(client):
    body = client.get("/meta").json()
    assert body["document_count"] == 1
    doc = body["documents"][0]
    assert (doc["doc_id"], doc["source"], doc["url"]) == ("rbi-kyc", "RBI", "https://example.org/kyc")
    assert doc["chunks"] == body["chunk_count"]
    assert body["examples"] and body["max_question_chars"] == settings.max_input_chars


def test_query_returns_the_pipeline_answer(client, pipeline):
    resp = client.post("/query", json={"question": "How fast is card closure?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Seven working days [1]."
    assert pipeline.calls[0]["question"] == "How fast is card closure?"


@pytest.mark.parametrize("payload", [
    {}, {"question": ""}, {"question": "x" * 4001},
    {"question": "ok", "top_k": 0}, {"question": "ok", "top_k": 500},
    {"question": "ok", "rerank_top_n": 99}, {"question": "ok", "provider": "evil"},
])
def test_invalid_requests_are_rejected_before_reaching_the_pipeline(client, pipeline, payload):
    assert client.post("/query", json=payload).status_code == 422
    assert pipeline.calls == []


def test_provider_override_is_ignored_unless_enabled(client, pipeline, monkeypatch):
    client.post("/query", json={"question": "q", "provider": "anthropic"})
    assert pipeline.calls[-1]["provider"] is None  # a visitor cannot pick a paid model

    monkeypatch.setattr(settings, "allow_provider_override", True)
    client.post("/query", json={"question": "q", "provider": "anthropic"})
    assert pipeline.calls[-1]["provider"] == "anthropic"


def test_per_minute_rate_limit_returns_429_with_retry_after(client, pipeline, monkeypatch):
    monkeypatch.setattr(security, "limiter", RateLimiter(per_minute=2, per_day=100, global_per_day=100))
    assert [client.post("/query", json={"question": "q"}).status_code for _ in range(3)] == [200, 200, 429]
    blocked = client.post("/query", json={"question": "q"})
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["retry-after"]) <= 61
    assert "wait" in blocked.json()["detail"].lower()
    assert len(pipeline.calls) == 2  # blocked requests never reach the LLM
    assert client.get("/health").status_code == 200  # only /query is limited


def test_limiter_windows_and_global_cap():
    rl = RateLimiter(per_minute=2, per_day=3, global_per_day=4)
    t = 1_000_000.0
    assert rl.check("a", t)[0] and rl.check("a", t + 1)[0]
    allowed, retry, reason = rl.check("a", t + 2)
    assert (allowed, reason) == (False, "minute") and 58 <= retry <= 60
    assert rl.check("a", t + 61)[0]                       # minute window slid
    assert rl.check("a", t + 130)[2] == "day"             # 3 per day used up
    assert rl.check("b", t + 131)[0]                      # other clients unaffected
    assert rl.check("c", t + 132)[2] == "global"          # 4 across everyone
    assert rl.check("a", t + 86_400 + 200)[0]             # next day


def test_client_ip_uses_rightmost_forwarded_entry_only_when_trusted(monkeypatch):
    from starlette.requests import Request

    def req(headers):
        return Request({"type": "http", "client": ("10.0.0.9", 1234),
                        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]})

    spoofed = {"x-forwarded-for": "1.1.1.1, 203.0.113.7"}
    assert security.client_ip(req(spoofed)) == "10.0.0.9"         # header ignored by default
    monkeypatch.setattr(settings, "trust_proxy_headers", True)
    assert security.client_ip(req(spoofed)) == "203.0.113.7"      # not the client-supplied 1.1.1.1
    assert security.client_ip(req({})) == "10.0.0.9"
    # Two trusted hops (e.g. CDN + load balancer): the client is second from the right.
    monkeypatch.setattr(settings, "proxy_hops", 2)
    assert security.client_ip(req({"x-forwarded-for": "6.6.6.6, 198.51.100.4, 203.0.113.7"})) == "198.51.100.4"
    assert security.client_ip(req({"x-forwarded-for": "198.51.100.4"})) == "198.51.100.4"


def test_ingest_endpoints_do_not_exist_without_an_admin_token(client):
    assert client.post("/ingest/sample").status_code == 404
    assert client.post("/ingest/path", json={"path": "data"}).status_code == 404


def test_ingest_requires_the_right_token(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "s3cret")
    assert client.post("/ingest/path", json={"path": "data"}).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/ingest/path", json={"path": "data"}, headers=wrong).status_code == 401


def test_ingest_path_cannot_escape_the_data_directory(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "s3cret")
    auth = {"Authorization": "Bearer s3cret"}
    for path in ["/etc/passwd", "../", "data/../../secrets"]:
        resp = client.post("/ingest/path", json={"path": path}, headers=auth)
        assert resp.status_code == 400, path
    missing = client.post("/ingest/path", json={"path": "data/does-not-exist"}, headers=auth)
    assert missing.status_code == 404


def test_unhandled_errors_return_clean_json(monkeypatch, stores):
    class Broken:
        def answer(self, *a, **k):
            raise RuntimeError("secret internal detail")

    monkeypatch.setattr(ragsys.agent, "get_pipeline", lambda: Broken())
    monkeypatch.setattr(security, "limiter", RateLimiter(1000, 1000, 1000))
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.post("/query", json={"question": "q"})
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal error. Please try again."}
    assert "secret internal detail" not in resp.text
