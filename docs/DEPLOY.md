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
| **Streamlit Community Cloud** | Fits, with a simpler UI. Free, no card. Reported memory limit is between 1 GB and 2.7 GB depending on the source. |
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
  --set-env-vars TRUST_PROXY_HEADERS=true,LLM_PROVIDER=groq,GROQ_API_KEY=your-key
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
  `--set-secrets GROQ_API_KEY=groq-key:latest` instead of `--set-env-vars`.

I have not run this deployment. The image itself is tested locally under the
same CPU and memory limits (see the numbers above).

## Option B: Streamlit Community Cloud (no card at all)

`deploy/streamlit/app.py` runs the same pipeline inside a Streamlit app: same
index, same retrieval, same guardrails, a plainer interface. Community Cloud
hosts it straight from the GitHub repo.

1. Push the repo to GitHub.
2. On share.streamlit.io choose **New app**, pick the repo, and set the main
   file to `deploy/streamlit/app.py`.
3. Under **Advanced settings → Secrets**, add:

   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "your-key"
   ```

It installs from `deploy/streamlit/requirements.txt`. To try it locally:

```bash
pip install -r deploy/streamlit/requirements.txt
streamlit run deploy/streamlit/app.py
```

I ran this locally and it answers correctly. I have not deployed it to
Community Cloud, so the install step there is untested.

## Any other host

Anything that runs a container with 1 GB of RAM works:

```bash
docker build -t sentinel-rag .
docker run -p 7860:7860 -e LLM_PROVIDER=groq -e GROQ_API_KEY=your-key sentinel-rag
```

## Getting a free LLM key

Without a key the app still works: it quotes the best passages instead of
writing an answer. For generated answers:

- **Groq**: create a key at console.groq.com. The default model is
  `llama-3.3-70b-versatile`. Free limits are tight on tokens per minute (a few
  thousand), which is about four or five questions a minute. That is why the
  API rate-limits to 8 questions a minute per visitor, and falls back to quoted
  passages if the provider returns a rate-limit error.
- **Gemini**: create a key in Google AI Studio and set `LLM_PROVIDER=gemini`.
  Check AI Studio for the models your free tier includes and set
  `GEMINI_MODEL` to one of them.

## Settings that matter in production

| Variable | Why |
|---|---|
| `TRUST_PROXY_HEADERS=true` | Rate-limit real visitors rather than the platform's proxy. |
| `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_PER_DAY`, `GLOBAL_DAILY_LIMIT` | Protect the free LLM quota. |
| `ADMIN_TOKEN` | Leave unset. The ingest endpoints then do not exist at all. |
| `ALLOW_PROVIDER_OVERRIDE` | Leave false so a visitor cannot switch to a paid model. |
| `TRACING_ENABLED` | Leave false unless you point `OTEL_EXPORTER_OTLP_ENDPOINT` at a collector. |
