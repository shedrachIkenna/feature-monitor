# Feature Monitor

## Problem Statement

When a deployed language model starts receiving different requests, or the model itself changes, operators often find out late. Feature Monitor tests whether shifts in one layer's GPT-2 Small activations, encoded as sparse autoencoder features, can flag changes, and compares those alerts with monitoring prompt sentence embeddings using the same drift method. This is an experiment: neither approach is assumed to work better, and results will include null or negative findings.

**Status: in progress**

## Quick Start

```sh
uv sync
uv run pytest
```

The first test run downloads GPT-2 Small and its SAE. `make test-fast` skips the tests that need them.

Run the API (loads the model once at startup):

```sh
uv run uvicorn backend.app:app
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"prompt": "Hello"}'
```

Endpoints: `GET /health`, `POST /chat`, `GET /features/recent?limit=50`, `GET /features/stats?minutes=60`, `GET /alerts`, `GET /dashboard/timeline`, `GET /dashboard/drifting`, `GET /dashboard/feature/{id}`.

### Open the dashboard

The same server serves the dashboard at `/`. From the project folder:

```sh
uv run uvicorn backend.app:app --port 8000
```

Wait for `Uvicorn running on http://127.0.0.1:8000` (loading GPT-2 and the SAE takes about half a minute), then open <http://localhost:8000/>. Stop it with Ctrl+C.

To look at a saved scenario run instead of live traffic, point the app at that run's database:

```sh
FEATURE_MONITOR_DB=data/dashboard_check/medical_ramp_5_2.db uv run uvicorn backend.app:app --port 8000
```

The charts show time up to now, so for a run from earlier in the day set **Time range** to **Last 24 hours**. "address already in use" means another copy of the server is already running on port 8000; stop that one first.

GPT-2 Small is not a chat model, so `/chat` replies are poor continuations of the prompt. That is expected: reply quality is not what we measure. The replies use a fixed seed (40 new tokens at most), so the same prompt always gets the same reply.

See [docs/design.md](docs/design.md) for the design.

## Folder Layout

- `backend/` - API and storage
- `monitor/` - feature extraction and drift detection
- `traffic/` - traffic generator and scenario configs
- `eval/` - experiments and results
- `frontend/` - dashboard
- `docs/` - design notes and build log
- `tests/` - tests mirroring the project folders
