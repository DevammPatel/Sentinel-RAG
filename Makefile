.PHONY: help install ingest-sample serve test eval up down fmt smoke

help:
	@echo "Targets:"
	@echo "  install         pip install -r requirements.txt + editable package"
	@echo "  ingest-sample   load the offline SEBI/RBI sample corpus into the index"
	@echo "  serve           run the FastAPI app locally (uvicorn, reload)"
	@echo "  test            run the pytest suite"
	@echo "  smoke           end-to-end wiring check with no API keys"
	@echo "  eval            run the RAGAS + custom eval harness"
	@echo "  up / down       start / stop the full docker stack (api+pgvector+phoenix)"

install:
	pip install -r requirements.txt && pip install -e .

ingest-sample:
	python scripts/ingest_cli.py --sample

serve:
	uvicorn ragsys.api.app:app --reload --port 8000

test:
	pytest -q

smoke:
	python scripts/smoke_test.py

eval:
	python -m ragsys.eval.run_eval --dataset data/golden/qa.jsonl

up:
	docker compose up --build -d

down:
	docker compose down

fmt:
	ruff check --fix src tests scripts || true
