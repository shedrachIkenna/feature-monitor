Feature Monitor

What this project is
A production monitoring system for a deployed language model. On every request, it
reads the model's internal activations at one layer, encodes them with a sparse
autoencoder (SAE), and stores the top 20 active features. A scheduled job compares
recent feature frequencies against a baseline and raises alerts when they drift.
We compare this against a simpler baseline: the same drift method applied to
sentence embeddings of the prompts. We report results honestly, including null or
negative results.

Stack
Python, managed with uv only. Never use pip. Add packages with uv add.
Model stage 1: GPT 2 small via TransformerLens, SAE release gpt2-small-res-jb.
Model stage 2 (Milestone 7 only): Gemma 2 2B with a Gemma Scope SAE.
SAE loading: sae-lens. Feature labels: Neuronpedia.
API: FastAPI. Storage: SQLite. Scheduler: APScheduler.
Frontend: plain HTML, CSS and JS served by FastAPI, with a small charting library.
Tests: pytest. Lint and format: ruff.
Containers: Docker. CI/CD: GitHub Actions. Hosting: Hugging Face Docker Space.
Machine
MacBook Pro, Apple M5, 16 GB memory. Use CPU for TransformerLens by default; use MPS only after its installed PyTorch version is confirmed safe.
Docker containers on this Mac cannot use the GPU, so container code must run on CPU.

Folder layout
backend/ API and storage
monitor/ feature extraction and drift detection
traffic/ traffic generator and scenario configs
eval/ experiments and results
frontend/ dashboard
docs/ design notes and the build log
tests/ tests mirror the folders above

How to work with me
Before editing any file, explain your plan in a few lines and wait for my go ahead.
Explain new concepts in simple English, building from the basics. Define each term
before using it.

Before using any library API, check the installed version and its actual function
signatures. Do not rely on memory for sae-lens, TransformerLens or Hugging Face APIs.

Small steps. One task per session. Run the tests after every change.
Never invent numbers or results. If something did not work, say so plainly.
Fix random seeds in anything experimental, so results can be regenerated.
Type hints on public functions. Short docstrings. No dead code.
When you finish a task, add 3 to 5 lines to docs/log.md: what was done, what
surprised us, what is still open.