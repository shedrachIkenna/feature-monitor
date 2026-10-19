---
title: Feature Monitor
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
short_description: Drift alerts from GPT-2 sparse autoencoder features
pinned: false
---

# Feature Monitor

[![CI](https://github.com/shedrachIkenna/feature-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/shedrachIkenna/feature-monitor/actions/workflows/ci.yml)

Drift alerts for a deployed language model from its sparse autoencoder (SAE) features, compared against the same drift test on prompt embeddings.

**Live demo:** [Shedrach/feature-monitor](https://huggingface.co/spaces/Shedrach/feature-monitor) on a Hugging Face Space (GPT-2 Small, CPU). Its data resets whenever the Space restarts. **Writeup:** [docs/writeup.md](docs/writeup.md).

## How it works

Every request's activations at one layer are encoded by an SAE and its 20 strongest features are stored. Every 60 seconds a job compares the latest 500 requests with a fixed baseline, once with SAE features and once with sentence embeddings of the prompts, and alerts when either score crosses its own calibrated threshold.

```mermaid
flowchart LR
    U[Request] --> API[FastAPI /chat]
    API --> M[Language model]
    M -->|reply| U
    M -->|activations at one layer| SAE[SAE encoder]
    SAE -->|top 20 features| DB[(SQLite)]
    API -->|prompt text| DB
    subgraph Drift job, every 60 s
        DB --> W[Latest 500 requests]
        W --> S1[SAE score: JSD vs baseline]
        W --> S2[Embedding score: MiniLM mean distance]
        S1 --> T{Above threshold?}
        S2 --> T
    end
    T -->|yes| A[Alert + changed features]
    A --> D[Dashboard]
    DB --> D
```

## Results

Mean delay in requests from the start of a gradual shift to the first alert, with the range over 10 seeds, at each detector's saved threshold (99th percentile of normal windows). Every shift was detected in all 10 seeds. From `eval/results/experiment_runs.csv`, via [`eval/02_results.ipynb`](eval/02_results.ipynb).

| Scenario | GPT-2 Small SAE | Gemma 2 2B SAE | Embeddings |
|---|---|---|---|
| `medical_ramp` | 204.9 (133–324) | 263.2 (87–342) | 404.0 (163–502) |
| `medical_nowhat_ramp` | 190.6 (18–331) | 284.6 (87–364) | 404.8 (165–501) |
| `new_topic_ramp` | 286.0 (193–407) | 349.8 (87–501) | 448.9 (174–544) |
| `injection_ramp` | 158.0 (18–242) | 223.5 (87–308) | 413.4 (174–534) |
| False alarms per 1,000 normal requests (runs with any) | 0.35 (2/10) | 0.15 (2/10) | 0.55 (4/10) |

**Finding.** The SAE detector alerted before mean embedding distance in 38 of 40 runs on GPT-2 Small and 36 of 40 on Gemma 2 2B, and stayed faster at equal false-alarm rates.
But part of GPT-2's lead is normal traffic already sitting near its threshold (25 of 40 first alerts), and Gemma's SAE detector was slower than GPT-2's in every scenario.
These are 10 seeds of templated customer-support traffic with gradual shifts only; see the [writeup](docs/writeup.md) for what did not work and the limitations.

![Drift chart: both detectors' scores over their thresholds during a medical_ramp run](docs/images/overview-alert.png)

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and about 1 GB of disk for GPT-2 Small, its SAE and the embedding model.

```sh
git clone https://github.com/shedrachIkenna/feature-monitor.git
cd feature-monitor
uv sync
make test-fast                          # offline tests
uv run pytest                           # all tests; the first run downloads GPT-2 Small and its SAE
uv run uvicorn backend.app:app --port 8000
```

Wait for `Uvicorn running on http://127.0.0.1:8000`, then open <http://localhost:8000/> and type a prompt in **Try it**, or:

```sh
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"prompt": "Hello"}'
```

To reproduce the results: `uv run python -m traffic.prepare_data` builds the prompt pools, `uv run python -m eval.run_experiment extract calibrate replay` replays every scenario and seed, and `make results` rebuilds the notebook. Details in [docs/usage.md](docs/usage.md).

## Documentation

- [docs/writeup.md](docs/writeup.md): technical writeup, results and limitations
- [docs/design.md](docs/design.md): design, drift test and scenarios
- [docs/usage.md](docs/usage.md): API, configuration, dashboard and offline experiments
- [docs/deployment.md](docs/deployment.md): Docker, the Hugging Face Space and rollbacks
- [docs/datasets.md](docs/datasets.md): prompt sources, licences and processing
- [docs/schema.md](docs/schema.md): database schema
- [docs/log.md](docs/log.md): build log
- [docs/study-report.md](docs/study-report.md): historical study report (Milestones 0–1)

## Folder layout

- `backend/` - API and storage
- `monitor/` - feature extraction and drift detection
- `traffic/` - traffic generator and scenario configs
- `eval/` - experiments and results
- `frontend/` - dashboard
- `docs/` - design notes and build log
- `tests/` - tests mirroring the project folders

## Licence and citation

Code under the [MIT licence](LICENSE). The datasets and models keep their own licences ([docs/datasets.md](docs/datasets.md); Gemma 2 2B is under Google's Gemma terms). To cite, see [CITATION.cff](CITATION.cff).
