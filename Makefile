.PHONY: docs docs-clean  help install lock sync test test-fast validate-manifest lint format run clean

help:
	@echo "Targets: install lock sync test test-fast validate-manifest lint format run clean"

install:
	uv sync --dev

lock:
	uv lock

sync:
	uv sync --dev

test:
	uv sync --dev
	uv run --dev python -m pytest

test-fast:
	uv sync --dev
	uv run --dev python -m pytest -q tests/test_manifest.py tests/test_contract.py tests/test_agent_smoke.py tests/test_examples.py

validate-manifest:
	uv sync --dev
	uv run --dev python -m pytest -q tests/test_manifest.py

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

run:
	uv run python -m agent_email_extract_non_nlp.api

clean:
	rm -rf .pytest_cache .ruff_cache


docs:
	uv sync --dev
	uv run --dev sphinx-build -b html docs docs/_build/html

docs-clean:
	rm -rf docs/_build
