# Build Log

Initialized the Python 3.12 uv project and added pytest and Ruff as development tools.
Created the backend, monitor, traffic, eval, frontend, docs, and mirrored test skeletons.
No unexpected issues; the configured Python version and development tools are available.
Application code and ML dependencies remain intentionally out of scope.

Drafted the design brief for model operators and ML researchers, including the three drift scenarios and comparison measures.
The brief separates evaluation goals from results; no monitoring outcomes have been measured yet.
Still open: reproducible scenario definitions, baseline and alert settings, and evaluation periods.

## 2026-10-08
Added a GitHub-ready README with the project scope, status, quick-start commands, and folder layout.
The design document contains no explicit M0-M8 milestone sequence, so any issue titles are inferred.
Open: define and confirm the M0-M8 sequence against the implementation plan.

## 2026-10-08
Added GPT-2 Small and block-8 SAE loading, cached once per process with MPS/CPU device selection.
Confirmed the installed SAE release metadata maps the selected hook to `blocks.8.hook_resid_pre`.
The dimension integration test passes; first model/SAE download requires network access.
Still open: TransformerLens deprecates the requested loader and warns MPS may be numerically unreliable with this PyTorch version.

### MPS correctness follow-up
Changed GPT-2 and SAE loading to CPU because the installed TransformerLens warns that MPS may silently produce incorrect results.
Updated the machine guidance and test to make the CPU choice explicit.
Still open: revisit MPS after TransformerLens confirms a safe PyTorch version; the deprecated loader migration remains separate.

## 2026-10-08
Added `extract_features` to cache only the SAE hook, encode without gradients, drop BOS, and return positive per-feature maxima.
On the password-reset prompt, BOS was `<|endoftext|>` and its maximum SAE activation was about 502 versus about 52 for the strongest remaining token.
The request returned 20 features in 0.330 seconds on the first measured call and 0.043 seconds on subsequent warm calls; model loading was excluded.
Still open: max pooling loses token frequency, and the installed TransformerLens loader emits deprecation warnings.

## 2026-10-11
Added `monitor/labels.py` with a single-feature Neuronpedia lookup, local JSON cache, one retry on timeout/429-style failures, and a CLI that prints the top prompt features with labels.
The live `GET /api/feature/gpt2-small/8-res-jb/{id}` call returned HTTP 200 and exposed nested `jsonDetails` entries whose `text` field holds the human-readable explanation.
The script now prints a label for each top feature, and the tests keep the network mocked so they can run deterministically offline.
Still open: confirm whether Neuronpedia’s explanation quality is consistent enough for production alerting and whether the cached file should be persisted in a dedicated app-data directory.

## 2026-10-11
Added 20 hand-written prompts and a runnable notebook comparing top-20 layer-8 SAE feature frequencies, overlap, labels, and per-prompt extraction times.
This run found 2 medical-enriched and 4 coding-enriched features, 3 shared common features (Jaccard 0.214), and a weak separation verdict.
With the model loaded before timing, median extraction was 0.0426 seconds and the slowest prompt took 0.1740 seconds.
The existing label cache held a generic release title; label parsing now prefers feature-specific explanations and refreshes that stale value.
Still open: test layers 4 and 11 after confirming their residual-pre SAE IDs and hook availability.

## 2026-10-11
Audited Milestone 1 (repo skeleton): all six items and both done-when checks hold.
Pinned Python with `.python-version` (3.12) and ran `ruff format .`; 8 files changed, layout only (mostly missing final newlines, one wrapped line).
Surprise: `ruff check` passed while `ruff format --check` failed, because the linter and the formatter are separate checks.
Still open: add `ruff format --check .` to the CI workflow once GitHub Actions is set up.

## 2026-10-11
Rewrote `docs/design.md` after a 6-question interview: four sudden scenarios (topic, style, framing, steering-vector model change), matched false-alarm comparison, 5 seeds.
Added `docs/study-report.md`, a full walkthrough of concepts, code, tests, the sanity check and review findings.
Surprise: five of the six "enriched" sanity-check features describe question phrasing, not topic; and the SAE loading warning is harmless (FVU 0.1845 both ways on one prompt).
Still open: label failures are cached permanently, retry is broader than documented, and the M0-M8 milestone list is not written down.

## 2026-10-11
Fixed review issue 1: `get_label` no longer caches "unlabelled" after both network attempts fail, so a brief outage is retried on the next call.
Successful lookups with no explanation are still cached, since that is a real answer.
The extended test failed on the old code (cache file written) and passes on the fix.
Still open: issues 2-10 from the study report.

## 2026-10-11
Fixed review issue 2: `get_label` now retries once only on transient failures (timeouts, connection errors, HTTP 429 and 5xx), and gives up immediately on other HTTP errors such as 404 or on invalid JSON.
Only the network call sits inside the `try` now, and the redundant exception list is gone (`OSError` already covers timeouts and URL errors).
The new 404 test failed on the old code; the 503 test passes on both and guards the retry path.
Still open: issues 3-10 from the study report.

## 2026-10-11
Fixed review issue 3: the label parser now reads only `explanations[].description`; the 45-line recursive search and the generic-release-title workaround are gone.
Correction to an earlier entry: `jsonDetails[].text` is not the explanation. It lives under `explanations[].scores[]` and holds example text snippets used to score explanations; the generic title came from `sourceSet.description`.
8 of 14 local cache entries held the generic title; they were removed and re-fetched live, and the 6 sanity-check labels still match the notebook.
Still open: issues 4-10 from the study report.

## 2026-10-11
Addressed review issue 4: added a phrasing-matched prompt set (10 templates, each filled with one medical and one coding subject), moved the separation rule into `eval/sanity.py` with tests, and ran both sets through the same rule in the notebook.
Matched result: 1 medical-enriched feature (18171, medical terms, 7/10 vs 0/10), 0 coding-enriched, Jaccard 0.429, verdict "weakly". The original set's five phrasing features all disappeared.
Surprise: removing the phrasing difference left only one topical feature, so most of the original separation was phrasing, and the topic signal in common top-20 features at layer 8 is thin.
Added ipykernel (previously installed outside uv) and nbconvert as dev tools so the notebook can be executed with `uv run jupyter nbconvert --execute`.
Still open: whether a drift test over the full feature distribution, not just common features, separates topics; layers 4 and 11; issues 5-10.

## 2026-10-11
Addressed review issue 5: the two tests that load GPT-2 and the SAE carry a registered `requires_model` marker, and `make test-fast` runs the other 10 without them.
`make test` still runs all 12; `make test-fast` took 2.6 s, not the under-a-second I estimated, because importing `monitor.labels` pulls in torch and TransformerLens.
Still open: GitHub Actions should cache `~/.cache/huggingface` so the model tests can run in CI; issues 6-10.

## 2026-10-11
Addressed review issue 6: removed the `_top_prompt_features` pass-through so `main` calls `extract_features` directly, and dropped a redundant `len(hits) <= 20` assertion.
The label test now imports `FeatureHit` from `monitor.extract`, where it is defined. No behaviour change; all 12 tests pass and the CLI still prints labels.
Still open: issues 7-10 from the study report.
