# Feature Monitor Design

## Problem
When a deployed language model starts receiving different requests, or the model itself
changes, operators often find out late. This project tests whether watching the model's
internal features can flag such changes, and compares it with a simpler method: watching
sentence embeddings of the prompts. It is an experiment; neither method is assumed to win.

## Users
- Today: a researcher (us) running controlled experiments to compare the two methods.
- Intended: an on-call ML engineer, who checks whether a model update or new client caused
  an alert and may roll back; and a trust and safety reviewer, who checks whether users are
  pushing the model somewhere risky.

## How it works
1. For each request, read GPT-2 Small's activations at layer 8 (`blocks.8.hook_resid_pre`).
2. Encode them with a sparse autoencoder (SAE, `gpt2-small-res-jb`, 24,576 features).
3. Store the request's 20 most active features (`backend/storage.py`, schema in
   [schema.md](schema.md)).
4. Every 60 seconds, a drift job scores the latest 500 requests against a fixed baseline
   and stores the score; a score above the threshold raises an alert (`monitor/job.py`).
5. The same job scores the same window from prompt sentence embeddings, as the comparison baseline.
6. A dashboard served at `/` shows scores, alerts and drifting features; its Playground page
   runs single prompts without storing them (`POST /chat` with `record: false`). At most 4
   `/chat` requests run or wait at once; the rest get 429 (`FEATURE_MONITOR_MAX_CHAT_QUEUE`).
   Recorded prompts need the `X-Record-Token` header when `FEATURE_MONITOR_RECORD_TOKEN` is
   set, and are refused (403) when it is not and `FEATURE_MONITOR_REQUIRE_RECORD_TOKEN=1`, as
   in the Docker image; so on the public Space only the token holder can add monitored traffic.
   Unrecorded prompts are rate limited per client address (`FEATURE_MONITOR_CHAT_RATE_LIMIT`,
   default 10 per minute, with the `limits` library); on a Space the address is the last
   `X-Forwarded-For` entry, the one Hugging Face's proxy appends.
7. It ships as one CPU-only Docker image on port 7860 (`Dockerfile`), with the models and the
   baseline features' labels downloaded at build time (`monitor/prefetch.py`), the traffic
   pools built at build time (`traffic/prepare_data.py`), and the database under `/data`.
   With `FEATURE_MONITOR_DEMO_TRAFFIC=1`, as in the image, `backend/demo.py` sends normal
   prompts (`normal_test`, tagged `demo_normal`) straight to the model from a background
   thread, only while no visitor's `/chat` is running or waiting: at full speed until 500
   requests are stored, then one every 20 seconds. `POST /admin/scenario`, behind
   `FEATURE_MONITOR_ADMIN_TOKEN`, runs a scenario from `traffic/scenarios/` through the same
   thread; the dashboard shows the warm-up and any running scenario as staged. A daily job
   deletes requests, drift checks and alerts older than 7 days, and each request is logged
   as one JSON line without prompt text or client address.
8. GitHub Actions CI (`.github/workflows/ci.yml`) runs on every push and pull request: ruff
   check, ruff format check and the tests without `requires_model` (offline), plus a Docker
   build that is not pushed. When CI passes on a push to `main`, `.github/workflows/deploy.yml`
   uploads that commit to the Hugging Face Space with `huggingface/hub-sync`, and the Space
   rebuilds the image (`docs/deployment.md`). The Space's disk is wiped on every restart, so
   `/health` reports `storage_resets` (true when Hugging Face's `SPACE_ID` is set) and the
   dashboard shows "Demo data resets on restart".

## Drift test (decided)
- **Baseline:** 3,000 prompts sampled (seed 0) from `normal_train`; per feature, how many had
  it in their top 20 (`monitor/baseline.py`).
- **Window score:** Jensen-Shannon divergence, in bits, between the baseline's and the window's
  feature distributions (`monitor/drift.py`).
- **Which features changed:** a two-proportion z-test per feature, with Benjamini-Hochberg
  adjustment for testing thousands at once. This is shown to the user only; it does not raise alerts.
- **Threshold:** the 99th percentile of scores of 1,000 random 500-prompt windows from the
  ~11,000 `normal_train` prompts not in the baseline (`monitor/calibrate.py`, settings in
  `monitor/config.yaml`).
- **Embedding detector** (`embedding_mean`): `all-MiniLM-L6-v2` embeddings; the window score is
  the squared distance between the window's mean embedding and the mean of the same 3,000
  baseline prompts, with its own 99th-percentile threshold (`monitor/embed_mean.py`). Each prompt
  is also put in the nearest of 50 k-means clusters (`monitor/embed_drift.py`), used only to list
  which topics changed in an alert. Until 2026-10-16 the clusters themselves were scored (kind
  `embedding`); offline, mean distance, MMD and 200 clusters were compared and mean distance was
  the simplest of the two best (`eval/embed_baselines.py`).

## Drift scenarios
Each scenario has 2,000 requests from `normal_test`. Shifted prompts are mixed in **gradually**:
0% for the first 500 requests, rising linearly to 30% by request 1,500, then held
(`traffic/scenarios/`, pools described in [datasets.md](datasets.md)).

| Scenario | Shift pool | Status |
|---|---|---|
| `normal_only` | none: the false-alarm control | Built |
| `medical_ramp` | MedQuAD health questions: topic **and** style change | Built |
| `medical_nowhat_ramp` | the same without "What…" openers, to separate topic from one wording cue | Built (offline evaluation only) |
| `new_topic_ramp` | held-out Bitext category (ACCOUNT): topic change, same style | Built |
| `injection_ramp` | deepset prompt injections: framing change | Built |
| Style change | the same requests rewritten casually (slang, typos) | Planned, not built |
| Model change | same prompts, a steering vector added before layer 8 at several strengths | Planned, not built |

In the model-change scenario, prompt embeddings cannot see the change by design. We will
report it as a test of what only internal monitoring can see, not as a win over embeddings.

## Success measures
- Detection rate, delay (requests after the shift starts until the first alert), and false
  alarms (alerts on `normal_only`), both per window and per run.
- Fair comparison: thresholds are swept over percentiles of each detector's calibration
  scores, so the detectors are compared at equal false-alarm rates (`eval/multiseed.py`).
- Every scenario is replayed offline with 10 fixed seeds; we report the spread.
- The project succeeds if it gives a fair, reproducible answer for each scenario, whichever
  method it favours. Null and negative results are reported.

## Out of scope
- Explaining the cause of drift. Changed features are shown with Neuronpedia labels as a
  hint only.
- Automatic actions such as rollback or blocking; the system only alerts.
- Embedding model responses.
- Load testing at production scale.
- Not yet: sudden switches (all scenarios are ramps), and models other than GPT-2 Small
  (Gemma 2 2B in Milestone 7).

## Known gaps
Things we know are missing, with when each must be fixed. Remove an item once it is fixed.
- Model versions are not pinned: GPT-2, the SAE and MiniLM are downloaded from the latest
  Hugging Face upload at build time, so a rebuild could get different weights. Optional; pin
  revisions for reproducible builds.
- The Space keeps no data: its database is wiped on every restart, redeploy and wake from
  sleep (stated on the dashboard). Optional; a Storage Bucket mounted at `/data` would keep
  it, and the `storage_resets` notice would then have to change.
- A Space rebuild downloads the models and the 2,488 baseline labels while building the
  image (unless Hugging Face's builder reuses cached layers, which we have not checked), so a
  deploy can fail if Hugging Face or Neuronpedia is down at that moment. Optional.
- The rate limit's counters live in the app's memory: they reset on restart and would not be
  shared between replicas (the Space runs one). Optional.
- On CPU the demo is slow: on the Space the warm-up sent about 47 prompts a minute and
  reached 500 in about 11 minutes, so a 2,000-request scenario would take about 43 minutes
  at that rate (not measured). Optional; a shorter demo scenario would show an alert sooner.

## Open questions
- Is one threshold crossing enough to alert, or should alerts need several in a row? At the
  saved thresholds, 2 of 10 normal 2,000-request runs raise a false alarm for each detector.
- Which steering direction and strengths, and at which layer?
- Should operators treat style-only changes as real alerts?
