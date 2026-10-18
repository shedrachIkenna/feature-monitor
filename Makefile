.PHONY: test test-fast lint format results

test:
	uv run pytest

test-fast:
	uv run pytest -m "not requires_model"

lint:
	uv run ruff check .

format:
	uv run ruff format .

# Regenerate every number and figure of eval/02_results.ipynb from the extraction cache.
results:
	HF_HUB_OFFLINE=1 uv run jupyter nbconvert --to notebook --execute --inplace eval/02_results.ipynb
