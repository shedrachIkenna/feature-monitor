# Feature Monitor

## Problem Statement

When a deployed language model starts receiving different requests, or the model itself changes, operators often find out late. Feature Monitor tests whether shifts in one layer's GPT-2 Small activations, encoded as sparse autoencoder features, can flag changes, and compares those alerts with monitoring sentence embeddings of the prompts, calibrated and evaluated the same way. This is an experiment: neither approach is assumed to work better, and results will include null or negative findings.

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

Endpoints: `GET /health`, `POST /chat`, `GET /features/recent?limit=50`, `GET /features/stats?minutes=60`, `GET /alerts`, `GET /dashboard/timeline`, `GET /dashboard/drifting`, `GET /dashboard/feature/{id}`, `GET /dashboard/labels?ids=1,2,3`.

`POST /chat` stores every request as monitored traffic. Send `"record": false` to get the reply and features without storing anything; the Playground page does this, so trying prompts by hand never moves the drift scores, raises alerts, or shows up as example prompts.

### Open the dashboard

The same server serves the dashboard at `/`. From the project folder:

```sh
uv run uvicorn backend.app:app --port 8000
```

Wait for `Uvicorn running on http://127.0.0.1:8000` (loading GPT-2 and the SAE takes about half a minute), then open <http://localhost:8000/>. Stop it with Ctrl+C.

To look at a saved scenario run instead of live traffic, point the app at that run's database:

```sh
FEATURE_MONITOR_DB=data/dashboard_check/medical_ramp_mean.db uv run uvicorn backend.app:app --port 8000
```

The charts show time up to now, so for a run from earlier in the day set **Time range** to **Last 24 hours**. "address already in use" means another copy of the server is already running on port 8000; stop that one first.

### What it looks like

These screenshots come from one `medical_ramp` run (seed 42; 2,000 requests in about 14 minutes, medical questions ramped into normal traffic), not from live traffic, viewed on the 1-hour range.

The overview: each detector's drift score divided by its own threshold, so the dashed line at 1 is the alarm level for both. In this run the SAE detector first alerted at request 780 and the embedding detector at request 1061. How far a line rises above 1 is not comparable between detectors.

![Overview with drift alerts](docs/images/overview-alert.png)

The Features page: SAE features that became more and less common in the latest 500 requests, and the rate over time of the top one, #13992 ("What" questions). Most of the top rising features are about question wording, not medicine, because the medical prompts are templated; medical features such as #18171 (inflammation terms) rank lower.

![Features page with feature 13992 selected](docs/images/features-drift.png)

The Playground: the top 20 features for one prompt, typed by hand and not stored.

![Playground with a medical prompt](docs/images/playground.png)

GPT-2 Small is not a chat model, so `/chat` replies are poor continuations of the prompt. That is expected: reply quality is not what we measure. The replies use a fixed seed (40 new tokens at most), so the same prompt always gets the same reply.

### Run in Docker

The image runs on CPU only and needs no network to start: GPT-2, the SAE, the MiniLM embedder
and the Neuronpedia labels of the 2,488 baseline features are downloaded while the image is
built. Labels of other features are fetched on first use when the network is reachable, and
show "unlabelled" otherwise.

```sh
docker compose up --build
```

The first build takes several minutes. Then open <http://localhost:7860/>;
`curl localhost:7860/health` reports `"device": "cpu"`. The database is
`/data/feature_monitor.db` on the `feature-monitor-data` volume, so it survives
`docker compose down` and rebuilds (`docker compose down -v` deletes it). The app runs as a
non-root user with uid 1000, as Hugging Face Spaces requires.

On Linux the lock file installs the CPU-only build of torch from the PyTorch index
(`[tool.uv.sources]` in `pyproject.toml`); macOS keeps PyPI's torch. PyPI's Linux torch
brings several GB of NVIDIA CUDA libraries that a CPU container cannot use. On this Mac the
image went from 11.8 GB to 4.06 GB on disk (4.22 GB to 1.22 GB compressed).

**Apple Silicon and amd64.** A Mac with an M-series chip is `arm64` (also called aarch64), so
Docker builds `linux/arm64` images by default. Most servers, including Hugging Face Spaces,
are `linux/amd64` (x86_64). The two run different machine code, so an image built for one
does not run natively on the other. To build for the host:

```sh
docker buildx build --platform linux/amd64 -t feature-monitor:amd64 --load .
docker buildx build --platform linux/amd64,linux/arm64 -t <registry>/feature-monitor --push .
```

The first command builds an amd64 image on the Mac by emulating an x86 CPU, which is slower
than a native build: here about 32 minutes against about 6 for arm64 (not a clean comparison:
the arm64 build reused cached downloads), and the emulated container took 31 seconds to start
against 6 to 9 natively. The second builds both in one multi-platform image
and pushes it to a registry; whoever pulls it gets the variant that matches their machine.
A Hugging Face Docker Space builds the `Dockerfile` itself on amd64, so pushing an image is
not needed there.

See [docs/design.md](docs/design.md) for the design.

## Folder Layout

- `backend/` - API and storage
- `monitor/` - feature extraction and drift detection
- `traffic/` - traffic generator and scenario configs
- `eval/` - experiments and results
- `frontend/` - dashboard
- `docs/` - design notes and build log
- `tests/` - tests mirroring the project folders
