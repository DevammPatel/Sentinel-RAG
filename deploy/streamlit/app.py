"""Sentinel-RAG as a Streamlit app.

The FastAPI service in `src/ragsys/api` is the main way to run this project. This
file exists for one reason: Streamlit Community Cloud hosts Streamlit apps for
free without a payment card, so it is the zero-cost, zero-card way to put a live
demo online. It runs the same pipeline in-process: same index, same retrieval,
same guardrails.

    streamlit run deploy/streamlit/app.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("INDEX_DIR", str(ROOT / "data" / "index"))

# Streamlit Cloud keeps secrets in st.secrets; the pipeline reads env vars.
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:  # no secrets file configured
    pass

st.set_page_config(page_title="Sentinel-RAG", page_icon="📑", layout="wide")

MAX_PER_SESSION_PER_MINUTE = 6
GLOBAL_PER_DAY = 1200
EXAMPLES = [
    "I asked my bank to cancel my credit card. How quickly do they have to close it?",
    "What limits apply to a 'Small Account' opened at a bank?",
    "What are the deadlines for reporting a cyber incident to SEBI?",
    "For how long is an accredited investor certificate valid?",
    "What is the price of Bitcoin today?",
]


@st.cache_resource(show_spinner="Loading models and index…")
def load_pipeline():
    from ragsys.agent import AgenticRAG

    pipeline = AgenticRAG()
    pipeline.retrieve("warm up")
    return pipeline


@st.cache_resource
def global_calls() -> list[float]:
    """Timestamps of every question asked, shared by all visitors."""
    return []


def allowed() -> tuple[bool, str]:
    now = time.time()
    mine = [t for t in st.session_state.setdefault("calls", []) if now - t < 60]
    everyone = global_calls()
    everyone[:] = [t for t in everyone if now - t < 86_400]
    if len(mine) >= MAX_PER_SESSION_PER_MINUTE:
        return False, "Too many questions in a short time. Please wait a moment and try again."
    if len(everyone) >= GLOBAL_PER_DAY:
        return False, "This demo has reached its daily limit. Please come back tomorrow."
    mine.append(now)
    everyone.append(now)
    st.session_state["calls"] = mine
    return True, ""


pipeline = load_pipeline()
chunks = pipeline.retriever.vector_store.all_chunks()
docs = {c.doc_id: c for c in chunks}

st.title("Sentinel-RAG")
st.caption(
    f"Ask {len(docs)} official SEBI and RBI documents ({len(chunks):,} passages). "
    "Every answer cites the passage it came from."
)

with st.sidebar:
    st.subheader("Try one")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state["question"] = example
    st.divider()
    with st.expander(f"What's indexed ({len(docs)} documents)"):
        for c in sorted(docs.values(), key=lambda c: (c.source, c.title)):
            st.markdown(f"**{c.source}** · [{c.title}]({c.url})" if c.url else f"**{c.source}** · {c.title}")
    st.caption(
        "Educational portfolio project. Not legal or compliance advice: always verify "
        "against the official SEBI and RBI documents linked from each passage."
    )

with st.form("ask"):
    question = st.text_area("Your question", key="question", max_chars=1000, height=90)
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and question.strip():
    ok, message = allowed()
    if not ok:
        st.warning(message)
        st.stop()
    with st.spinner("Searching the documents and reranking passages…"):
        resp = pipeline.answer(question)

    answer_col, source_col = st.columns([1.1, 1])
    with answer_col:
        label = {"generated": f"Generated answer · {resp.model}",
                 "extractive": "Quoted from sources",
                 "refused": "Declined to answer"}[resp.mode]
        (st.success if resp.mode == "generated" else st.error if resp.refused else st.info)(label)
        if resp.notice:
            st.caption(resp.notice)
        st.markdown(resp.answer)
        if resp.rewritten_query:
            st.caption(f"First search was weak, so the query was rewritten to: `{resp.rewritten_query}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", f"{resp.latency_ms:.0f} ms")
        m2.metric("Retrieve + rerank", f"{resp.retrieval_ms:.0f} ms")
        m3.metric("Generate", f"{resp.generation_ms:.0f} ms")
        m4.metric("Sources cited", f"{len(resp.citations)} of {len(resp.sources)}")
    with source_col:
        st.subheader("Retrieved passages")
        if not resp.sources:
            st.caption("No passages were retrieved for this question.")
        for s in resp.sources:
            title = f"[{s.title}]({s.url})" if s.url else s.title
            st.markdown(f"**{s.marker}** {title}")
            ref = f" · `{s.circular_no}`" if s.circular_no else ""
            cited = " · cited" if s.cited else ""
            st.caption(f"{s.source}{ref} · {s.relevance:.0%} relevant{cited}")
            st.progress(min(max(s.relevance, 0.0), 1.0))
            with st.expander("Passage"):
                st.write(s.text)
