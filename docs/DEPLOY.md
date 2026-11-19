# Deploying Sentinel-RAG for free

The app needs very little: one container, no database, no GPU. The index and
both models are baked into the image, so the only outbound call is to the LLM.

Measured on the image built from this repo (2 CPUs, 1 GB limit, no LLM key):

| | |
|---|---|
| Image size | 836 MB |
| Cold start to healthy | about 3 s |
| Memory after 40 questions | 333 MB container usage, 505 MB peak process RSS |
| Latency per question | 1.05 s mean, 1.24 s p95 (retrieval + rerank) |

So the target is **1 vCPU or more and 1 GB of RAM**. That rules out the common
512 MB / 0.1 vCPU free tiers: the process peaks at about 505 MB, and reranking
on a tenth of a CPU would take tens of seconds.

## What was checked, and when

Free tiers changed a lot during 2026. These were checked on 13 November 2026;
re-check before you rely on any of them.

| Host | Status for this app |
|---|---|
| **Google Cloud Run** | Fits. Monthly free allowance of 2 million requests, 180,000 vCPU-seconds and 360,000 GiB-seconds (as published on Google's pricing page; I could not load the page directly, the figures come from search results quoting it). Needs a billing account with a card on file. |
| **Streamlit Community Cloud** | Fits, with a simpler UI. Free, no card. Reported memory limit is between 1 GB and 2.7 GB depending on the source; the app uses about 490 MB. |
| Hugging Face Spaces (Docker) | **No longer free.** The official docs now say Docker and Gradio Spaces "require a paid plan to create". |
| Render free web service | Too small: 512 MB RAM, 0.1 CPU. |
| Koyeb | Too small (512 MB, 0.1 vCPU), and sources disagree on whether a free compute tier still exists. |

## Option A: Google Cloud Run (the real UI and API)

This runs the Docker image as-is and scales to zero when nobody is using it.
You are only billed while a request is being served, and a portfolio demo stays
far inside the free allowance: one question uses about one vCPU-second, so
180,000 vCPU-seconds is on the order of 100,000 questions a month.

It does need a card on file. To make sure it cannot cost anything meaningful,
keep `--max-instances 1` and set a budget alert at a small amount in the
Billing console.

```bash
gcloud run deploy sentinel-rag \
  --source . \
  --region us-central1 \
  --memory 1Gi --cpu 1 \
  --min-instances 0 --max-instances 1 \
  --allow-unauthenticated \
  --set-env-vars TRUST_PROXY_HEADERS=true,LLM_PROVIDER=gemini,GEMINI_API_KEY=your-key
```

Notes:

- Cloud Run injects its own `PORT`; the image already listens on whatever
  `PORT` says, so there is nothing to configure.
- `--source .` builds the Dockerfile with Cloud Build and stores the image in
  Artifact Registry. Both have their own free allowances; delete old image
  versions so stored images do not pile up.
- `TRUST_PROXY_HEADERS=true` makes the rate limiter read the caller's address
  from `X-Forwarded-For`. If every visitor seems to share one limit, the
  platform is adding more than one proxy hop: set `PROXY_HOPS=2`.
- For a real secret store, put the key in Secret Manager and use
  `--set-secrets GEMINI_API_KEY=gemini-key:latest` instead of `--set-env-vars`.

I have not run this deployment. The image itself is tested locally under the
same CPU and memory limits (see the numbers above).

## Option B: Streamlit Community Cloud (no card at all)

`deploy/streamlit/app.py` runs the same pipeline inside a Streamlit app: same
index, same retrieval, same guardrails, a plainer interface. Community Cloud
hosts it straight from the GitHub repo, for free.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub
   account and allow it to read the repository.
2. Choose **Create app**, then **Deploy a public app from GitHub**, and fill in:
   - Repository: `DevammPatel/Sentinel-RAG`
   - Branch: `main`
   - Main file path: `deploy/streamlit/app.py`
   - App URL: pick a name, for example `sentinel-rag`
3. Open **Advanced settings**:
   - Python version: **3.12** (the version tested below)
   - Secrets: paste the two lines from `deploy/streamlit/secrets.toml.example`
     with your real key:

     ```toml
     LLM_PROVIDER = "gemini"
     GEMINI_API_KEY = "your-key"
     ```

4. Click **Deploy**. The first build installs the dependencies and the first
   visit downloads about 90 MB of model weights, so the first load takes a few
   minutes. Later visits are fast until the app goes to sleep from inactivity;
   then the next visitor waits for it to wake.

You can add or change the key later under **App settings → Secrets** without
redeploying. Without a key the app still works and quotes source passages.

How it installs: Community Cloud looks for a requirements file next to the
entrypoint first, so it uses `deploy/streamlit/requirements.txt`, which lists
only what the app imports (no API server, Postgres or tracing). It runs the app
from the repository root, which is what the app expects.

**Tested** (19 November 2026): a clean `python:3.12-slim` Linux container,
limited to 1 GB of RAM and 2 CPUs, installed that file with `uv` (as Community
Cloud does) and started the app from the repo root. With the Gemini key given
as a Streamlit secret (not an environment variable), it wrote a correct, cited
answer in about 2.5 seconds; without a key it quoted passages and refused an
out-of-scope question. It used about 490 MB of memory.
**Not tested:** the deployment on Community Cloud itself.

To run it locally:

```bash
pip install -r deploy/streamlit/requirements.txt
streamlit run deploy/streamlit/app.py
```

## Any other host

Anything that runs a container with 1 GB of RAM works:

```bash
docker build -t sentinel-rag .
docker run -p 7860:7860 -e LLM_PROVIDER=gemini -e GEMINI_API_KEY=your-key sentinel-rag
```

## Getting a free LLM key

Without a key the app still works: it quotes the best passages instead of
writing an answer. For generated answers:

- **Gemini** (the default): create a key in [Google AI Studio](https://aistudio.google.com)
  with a Google account; the free tier needs no card. Newer keys start with
  `AQ.` rather than `AIza`; both work.
  - The default model is `gemini-3.5-flash-lite`, with `gemini-3.1-flash-lite`
    as a fallback. Both are on the free tier. In a test on 19 November 2026,
    `gemini-3.5-flash-lite` answered in about a second, while
    `gemini-3.8-flash` (and the `gemini-flash-latest` alias) repeatedly timed
    out or returned 503 "unavailable" on the free tier.
  - If a model is rate limited, overloaded or slow, the app tries the next
    one and skips the failed model for a minute, so visitors don't each wait
    out the same timeout. If every model is down, it quotes passages instead.
  - Google may use the content of free-tier requests to improve its products.
    The documents are public, but visitors' questions are sent too, which is
    why the UI tells them so.
  - Google does not publish the free-tier request limits; AI Studio shows the
    ones for your project.
  - Change models with `GEMINI_MODEL` and `GEMINI_FALLBACK_MODELS`
    (comma-separated). AI Studio lists the models your key can use.
- **Groq**: not to be confused with xAI's Grok, which is paid. Create a key at
  console.groq.com and set `LLM_PROVIDER=groq`. The default model is
  `llama-3.3-70b-versatile`.

Gemini 3 models always "think" before answering. The app asks for low thinking
effort and gives each reply room for it, so short calls do not come back empty;
if one ever does, it is treated as a failure and handled like any other.

## Settings that matter in production

| Variable | Why |
|---|---|
| `TRUST_PROXY_HEADERS=true` | Rate-limit real visitors rather than the platform's proxy. |
| `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_PER_DAY`, `GLOBAL_DAILY_LIMIT` | Protect the free LLM quota. |
| `ADMIN_TOKEN` | Leave unset. The ingest endpoints then do not exist at all. |
| `ALLOW_PROVIDER_OVERRIDE` | Leave false so a visitor cannot switch to a paid model. |
| `TRACING_ENABLED` | Leave false unless you point `OTEL_EXPORTER_OTLP_ENDPOINT` at a collector. |
