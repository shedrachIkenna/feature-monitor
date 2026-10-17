# Database schema

SQLite, defined in `backend/storage.py`. The file path comes from `FEATURE_MONITOR_DB`;
the default is `data/feature_monitor.db` at the repo root, whatever the working directory.

## Tables

**requests**: one row per monitored model call: UTC `created_at`, `prompt`, `reply`, `model_name`,
an optional `scenario` tag, and `gen_ms`/`extract_ms` timings. Indexed on `created_at`.

**feature_hits**: one row per top-k SAE feature of a request: `feature_id`, `activation`, `rank`
(1 = strongest). Keyed by `(request_id, rank)`; `request_id` must exist; indexed on `feature_id`.

**alerts**: one row per drift alert: `created_at`, `kind`, optional `feature_id`, a numeric `score`,
and `details` as JSON text. Written by the drift job (`monitor/job.py`) with `save_alert`;
read by `GET /alerts` through `list_alerts`.

**drift_checks**: one row per detector per drift check, alert or not: `created_at`, `kind`
(`sae` or `embedding_mean`; databases from before 2026-10-16 hold `embedding`, the old
cluster score), `score`, the `threshold` it was judged against, `window_size`, and the
window's `first_request_id`/`last_request_id`. Written by the drift job with `save_drift_check`, so
`GET /dashboard/timeline` is a cheap read of stored scores instead of rescoring windows. Indexed on
`created_at`.

## Why hits have their own table

The drift job asks "in how many requests did feature X appear in this window?". With one
row per hit and an index on `feature_id`, SQLite answers by lookup; a JSON column on
`requests` would have to be parsed for every request on every query. Separate rows also
let the database enforce one hit per rank per request.

## Conventions

- Timestamps are stored as fixed-width UTC strings (`2026-10-15T12:00:00.000000Z`), so
  comparing them as text gives the same order as comparing them as times.
- Dashboard buckets are 1- or 5-minute slots (`bucket_minutes`) aligned to the UTC clock;
  SQLite rounds each `created_at` down with `strftime('%s', ...)`.
- Time windows are half-open, `[since, until)`, so back-to-back windows never count a
  request twice. `request_count` gives the denominator for feature frequencies.
- Deployment: a Hugging Face Space's disk is wiped on every restart, so the database there
  starts empty each time (the dashboard says so); a Storage Bucket mounted at `/data` would
  keep it, and none is attached. The Docker image sets
  `FEATURE_MONITOR_DB=/data/feature_monitor.db`; `docker-compose.yml` mounts a volume there.
