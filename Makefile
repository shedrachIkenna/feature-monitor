.PHONY: test test-fast lint format

test:
	uv run pytest

test-fast:
	uv run pytest -m "not requires_model"

lint:
	uv run ruff check .

format:
	uv run ruff format .