.PHONY: dev lint test up ingest eval

TEST_DSN ?= postgresql://research:research@localhost:5432/research

dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	MAR_TEST_DSN=$(TEST_DSN) python -m pytest -q

up:
	docker compose up --build

ingest:
	research ingest

eval:
	research evaluate --judge
