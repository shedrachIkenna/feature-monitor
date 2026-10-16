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
5. The same job runs the same test on prompt sentence embeddings, as the comparison baseline.
6. A dashboard served at `/` shows scores, alerts and drifting features; its Playground page
   runs single prompts without storing them (`POST /chat` with `record: false`).

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
- **Embedding detector:** `all-MiniLM-L6-v2` embeddings, each prompt assigned to the nearest of
  50 k-means clusters fitted on the same 3,000 prompts; the cluster is scored exactly like a
  feature list (`monitor/embed_drift.py`). Three stronger embedding detectors (200 clusters,
  mean-embedding distance, MMD) were tested offline (`eval/embed_baselines.py`); the live app
  still runs the 50-cluster one.

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

## Open questions
- Should the live embedding detector switch to mean distance or MMD? Both alert sooner than
  50 clusters offline, though still later than the SAE detector (see [log.md](log.md)).
- Is one threshold crossing enough to alert, or should alerts need several in a row? At the
  saved thresholds, 2 of 10 normal 2,000-request runs raise a false alarm for each detector.
- Which steering direction and strengths, and at which layer?
- Should operators treat style-only changes as real alerts?
