.PHONY: help install fetch ingest serve test smoke eval calibrate lint

help:
	@echo "install    create deps (pip install -r requirements-dev.txt + editable package)"
	@echo "fetch      download the official SEBI/RBI documents listed in data/sources.json"
	@echo "ingest     build the index in data/index from data/raw"
	@echo "serve      run the API + chat UI on http://localhost:8000"
	@echo "test       run the offline test suite"
	@echo "smoke      end-to-end check with real local models, no keys"
	@echo "eval       retrieval ablation over data/golden/qa.jsonl"
	@echo "calibrate  sweep the refusal threshold"

install:
	pip install -r requirements-dev.txt && pip install -e . --no-deps

fetch:
	python scripts/fetch_circulars.py

ingest:
	python scripts/ingest_cli.py --reset

serve:
	uvicorn ragsys.api.app:app --port 8000

test:
	pytest -q

smoke:
	python scripts/smoke_test.py

eval:
	python -m ragsys.eval.run_eval

calibrate:
	python -m ragsys.eval.calibrate

lint:
	ruff check src tests scripts
