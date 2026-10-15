# Build Log

## 2026-10-08
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

## 2026-10-11
Addressed review issue 7: `load_model_and_sae` now uses `TransformerBridge.boot_transformers("gpt2")` with `enable_compatibility_mode()` instead of the deprecated `HookedTransformer.from_pretrained`.
On all 40 sanity prompts the layer-8 activations were bit-for-bit identical (max difference 0.0) and all 40 top-20 sets matched; a regression test now checks this on two prompts.
Surprise: in one 20-prompt measurement the bridge was faster, 0.0179 s versus 0.0425 s per prompt. Test warnings dropped from 4 to 2.
Still open: the 2 remaining warnings come from inside sae-lens (documented for a possible upstream contribution); delete the regression test when TransformerLens 4.0 removes HookedTransformer.

## 2026-10-11
Added `docs/upstream-warnings.pdf` on the two remaining sae-lens warnings, for a possible open-source contribution (both libraries are MIT-licensed).
Warning 1 (deprecated `transformer_lens.utils` import) is already fixed in open, unmerged SAELens PR #739 (issue #738); the useful contribution is testing that PR with our setup.
Warning 2 (`model_from_pretrained_kwargs`) is unchanged on main with no existing issue; it names a non-existent attribute, recommends a deprecated API, fires even when loading is correct, and points at the wrong line.
Surprise: transformer-lens 4.2.0 is released, but sae-lens 6.54.0 pins `<4.0.0`, so we stay on 3.9.0 for now.

## 2026-10-11
Addressed review issue 8: removed the repository-local git identity ("Feature Monitor <feature-monitor@local>") so commits use the global one, and rewrote all 13 existing commits to credit Shedrach Nwali.
Checked: author name and email updated, original dates kept, final file contents identical to the pre-rewrite backup. Nothing had been pushed, so no one else's copy is affected.
Every commit hash changed; the two old hashes in `docs/study-report.md` were updated to the new ones.
Still open: issues 9-10 from the study report.

## 2026-10-11
Addressed review issue 9: deleted `docs/setup-report.pdf`, which described the 8 October skeleton ("No ML libraries were installed") and was now misleading; the study report supersedes it.
It stays recoverable from history with `git show c5b023b:docs/setup-report.pdf`.
Still open: issue 10 (milestone list) from the study report.

## 2026-10-11
Closed review issue 10 by decision: the milestone sequence will be supplied one milestone at a time rather than written down in advance, so there is no `docs/milestones.md` yet.
All 10 issues from the study report's review are now resolved; 13 tests pass, and the only remaining warnings come from inside sae-lens (see `docs/upstream-warnings.pdf`).
Still open: the sae-lens upstream contributions, the drift-test choice, and layers 4 and 11 for the sanity check.

## 2026-10-11
Verified Milestone 0.3 (README and project board): README had the required sections; its first sentence was rewritten to match the design, with a note on the first-run model download, `make test-fast`, and a link to the design.
Added the missing 2026-10-08 heading to the first two log entries without changing their wording.
Surprise: the earlier M0-M8 issue list was inferred, not taken from the design. The milestone sequence now comes from the user: Milestone 0 is Scope and setup (0.1 repository and environment, 0.2 design document, 0.3 README and project board).
Still open: which milestone number covers the model loading, extraction, labels and sanity-check work already done.

## 2026-10-11
Verified Milestone 1.1 (load the model and SAE): all items hold, with two deliberate changes from the original wording (TransformerBridge instead of HookedTransformer, and CPU instead of MPS-if-available). Added the missing tensor-shapes explanation to the study report and brought its section 3.2 up to date.
Added `eval/mps_check.py`. Pass rule fixed before running: identical top-20 sets on all 40 sanity prompts and a relative activation difference of at most 0.1%. Result: PASS (40/40, relative difference 2.4e-7).
Surprise: MPS was faster but not dramatically so, median 10.8 ms versus 17.2 ms on CPU over 120 calls.
Still open: whether to switch the default device to MPS; the check covers layer-8 extraction only, not generation or steering.

## 2026-10-11
Kept CPU as the default device after the MPS check: the speed gain (about 6 ms per request) does not justify differing from the CPU-only Docker environment yet.
Refreshed `docs/study-report.md` to match the code: milestone status table, TransformerBridge, the strict label parser and its error handling, all 13 tests, both sanity-check prompt sets with results, and the 10 resolved review issues.
Surprise: much of the report had drifted within a single day of fixes, so it should be refreshed after each milestone, not left to accumulate.
Still open: the remaining Milestone 1 items, still to be supplied.

## 2026-10-11
Verified Milestone 1.2 (prompt to top features): the code meets every item. "How do I reset my password?" returns 20 features in 331 ms on the first call and 28 ms after that, on CPU, with model loading excluded.
Added the two missing explanations to the study report: why BOS activations are large, and max versus mean pooling.
Surprise: at layer 8, the BOS residual vector is about 25 times larger than any other token's (median 24.8x over 40 prompts) and exactly identical in every prompt, so it carries no prompt information.
Still open: the attention-sink explanation for the large BOS norm comes from published work and is untested here.

## 2026-10-13
Verified Milestone 1.3 (feature labels): a live request to the endpoint returned HTTP 200, and the script prints the top 10 labelled features. One gap: the retry fired instantly, which is not polite under a rate limit.
`get_label` now pauses before its single retry: 1 s by default, or a numeric `Retry-After` header on 429/5xx, capped at 10 s. Permanent errors still return at once without pausing.
Tests replace `time.sleep` with a recorder; the 4 pause tests failed on the old logic and pass now (15 tests in total).
Surprise: Neuronpedia sent no rate-limit headers on a successful request, so the 1 s default is our own choice.

## 2026-10-15
Labels now prefer the highest-scored Neuronpedia explanation, but only if its mean score is at least 0.7; otherwise the first explanation is used. The 0.7 cut-off is our own judgement (scores near 0.5 are close to coin-flip).
On every feature checked, the first (gpt-3.5) explanation is unscored and the second (gpt-4o-mini) has two eleuther_fuzz scores. Pure "highest-scored" would have relabelled feature 23838 as "societal issues" (score 0.49), although its top output tokens are all question marks; the minimum keeps the question-mark label.
Surprise: the two explanation sources disagreed on 13 of 14 cached features. The cache was cleared (old copy kept outside the repo) and the notebook re-run: 12 of 14 labels changed, counts and verdicts are unchanged, and the phrasing conclusion still holds because it rests on feature behaviour, not labels.
Still open: whether the cache should record which rule chose each label, so a future rule change is detected automatically.

## 2026-10-15
Completed Milestone 1.4 (sanity check notebook): added a plain-words verdict cell to `eval/01_sanity_check.ipynb`. Layer 8 separates medical from coding prompts weakly (original set 2/4 enriched, Jaccard 0.214; matched set 1/0, Jaccard 0.429), and the cell proposes layers 4 and 11 with the change needed (make the layer a parameter in `monitor/model.py` and `monitor/labels.py`, and key the label cache by layer).
Ran the notebook twice from a clean kernel: feature counts and overlap were identical both times; the study report's status table, section 5.4 and Part 9 now match.
Surprise: the slowest extraction varied a lot between runs (0.3307 s, then 0.0503 s; medians 0.0179 s and 0.0174 s), so the verdict cell does not copy timings.
Still open: the layer 4 and 11 runs; whether the timing loop should make one untimed warm-up call first.

## 2026-10-15
Deferred the layer side quest by decision: layer 8 stays the monitored layer for now, and the next milestone takes priority.
Parked, in order: (1) make the layer a parameter in `monitor/model.py` and `monitor/labels.py`, with the label cache keyed by layer; (2) add an untimed warm-up call to the notebook's timing loop; (3) re-run the sanity check at layers 4 and 11 with a "better" criterion fixed beforehand (a stronger verdict on the matched set).
Still open: these three items, and the drift-test choice, which the layer results would inform.

## 2026-10-15
Completed Milestone 2.1 (storage layer): `backend/storage.py` stores requests, ranked feature hits and alerts in SQLite (standard-library `sqlite3`, no new dependency), with indexes on `requests.created_at` and `feature_hits.feature_id`. `save_request` writes a request and its hits in one transaction; `feature_counts` and the extra `request_count` use half-open `[since, until)` windows. 10 new tests on temporary database files; 32 tests pass in total. Table descriptions are in `docs/schema.md`.
Surprise: importing `FeatureHit` from `monitor.extract` also imports torch, so storage imports it for type checking only.
Still open: alert read/write functions (with the drift job); whether the Hugging Face Space gets persistent storage, without which `/data` is wiped on restart.

## 2026-10-15
Completed Milestone 2.2 (FastAPI app): `backend/app.py` serves `/health`, `/chat`, `/features/recent` and `/features/stats`. The model and SAE load once in the lifespan; model work runs in a threadpool behind a lock, because the fixed seed (`torch.manual_seed(0)`) is process-wide. Added `storage.hits_for_requests` for `/features/recent`. Tests inject a fake engine through `create_app`: 41 tests pass, the backend ones in about 3 s. A real `uvicorn` + `curl` run answered every endpoint; empty prompts get 422.
Surprise: generation dominates the cost. On one curl request: 586 ms to generate 40 tokens, 23 ms to extract features. The seeded reply was identical across two separate processes.
Still open: the lock serialises all `/chat` requests (about 0.6 s each in the one request measured), so throughput is not yet measured and the traffic generator should be paced with this in mind.

## 2026-10-15
Completed Milestone 2.3 (load test): `eval/latency.py` sends 100 varied prompts (`eval/data/latency_prompts.json`, 10 topics) one at a time, once with extraction on and once with the new `FEATURE_MONITOR_EXTRACT=0` flag, each against its own uvicorn server and temporary database, after one untimed warm-up. All 200 requests returned 200 and stored a row (20 hits each with extraction on, 0 off). On CPU: total p50 408.3 ms on vs 389.7 ms off, an extraction overhead of +4.8% at p50 and +5.9% at p95; `extract_ms` p50 17.6 ms, p95 20.3 ms. Results are in `eval/results/latency_m2.json`. 52 tests pass.
Surprise: generation, not extraction, sets the latency, and it varies a lot by prompt (80 ms to 816 ms) because some replies end early. The 586 ms from the single cold curl request in 2.2 was not typical: the warm p50 is 389 ms.
Still open: this is one run with the modes in a fixed order (on, then off), so run-to-run variation and order effects are unmeasured; one 816 ms generation outlier in the "on" run is unexplained.

## 2026-10-15
Deferred: limits of the Milestone 2.3 latency numbers (+4.8% p50, +5.9% p95), to address later.
(1) One run, modes always in the same order (on, then off): run-to-run variation and order effects are unmeasured. Recommendation: repeat 5 runs, alternating which mode goes first, and report the median overhead with its min–max range.
(2) Generation time depends on the prompt (80–816 ms, because some replies end early), which blurs a comparison of medians. Recommendation: a paired comparison, since the fixed seed gives each prompt the same reply in both modes. Per prompt, take total on minus total off, report the median and p95 of those differences, and also record reply token count so time per token can be reported.
(3) One unexplained 816 ms generation in the "on" run. Recommendation: with the per-request token counts from (2), check whether it recurs on the same prompt in repeated runs. If it does, the cause is a long reply; if not, record it as machine noise.

## 2026-10-15
Completed Milestone 3.1 (datasets): `traffic/prepare_data.py` builds five pools from Bitext (normal, with ACCOUNT held out as shift_new_topic), MedQuAD (shift_medical) and deepset (shift_injection). Final sizes: normal_train 14,017, normal_test 6,008, shift_new_topic 5,454, shift_medical 43,448, shift_injection 125. Two runs give byte-identical files. Sources, licences and choices are in `docs/datasets.md`; the data files are gitignored. 62 tests pass (the 3 model tests were not run).
Surprise: 25% of Bitext messages contain unfilled `{{Slot}}` placeholders, which would have made shift detection easy for the wrong reason, so they are filled from seeded value lists. 98 of the 223 deepset injections under 300 characters (44%) are not fully English (German, mixed German/English, plus Spanish, French, Russian, Croatian, Khmer).
Still open: shift_injection is small (125) and the English check is rough both ways (it drops at least 3 English items with German names, and keeps a few English items that start with German topic words such as "Datenschutz Europa"); Bitext and MedQuAD are synthetic or templated, not real user traffic; MedQuAD questions are templated, so a medical alert may reflect style as well as topic; some deepset "injections" look benign and were kept as labelled.

## 2026-10-15
Completed Milestone 3.2 (traffic generator): `traffic/generate.py` builds a seeded prompt sequence from a scenario in `traffic/scenarios/` (normal_only, medical_ramp, injection_ramp, new_topic_ramp). It then either writes the sequence to `traffic/runs/` (`--dry-run`, gitignored) or POSTs each prompt to `/chat` with the scenario tag. Every row records `is_shift` (ground truth) and, when sent, the HTTP status and `request_id`. 77 tests pass (the 3 model tests were not run). Two dry runs of medical_ramp were byte-identical; every 100-request block is within 0.5 points of its schedule.
Decisions (agreed with the user): (1) Each block of 100 gets exactly round(sum of scheduled fractions) shift requests at seeded random positions, instead of a coin flip per request, which would vary by about ±4.6 points per block at 30%. (2) All three ramps share one schedule (0% to request 500, linear to 30% by 1,500, held at 30% to 2,000) so they can be compared. The 500-request hold lets us see whether an alert persists. (3) The injection pool (125) is reused: it is drawn without replacement and reshuffled when empty, giving 295 shift requests, each injection repeated 2–3 times. (4) Normal traffic comes from `normal_test`; `normal_train` is kept for the baseline.
Surprise: with exact block counts, the per-block mix during the ramp sits slightly below the schedule (for example 1.485 expected rounds to 1), which is expected rounding, not bias in the generator.
Still open: repeated injection prompts may make injection drift easier to detect than with unique prompts; a live run takes about 14 min per scenario at the measured ~0.4 s per request and has not been done yet; positions are random only within each block, so the shift order is less random than independent draws.

## 2026-10-15
Deferred: live testing of the Milestone 3.2 traffic generator, to address later.
(1) Send mode has only been tested against a fake server (`httpx.MockTransport`), never the real API. Recommendation: start uvicorn with a temporary database, send a short scenario (for example the first 100 requests of medical_ramp), and check that every request returns 200 and that the stored `scenario` tags and `request_id`s match the output file.
(2) No full live scenario run yet (about 14 min each at ~0.4 s per request). Recommendation: run all four before the drift-evaluation milestone and record the failure count and wall time.

## 2026-10-15
Completed Milestone 3.3 (baseline): `monitor/baseline.py` samples 3,000 prompts from `normal_train` (seed 0), runs `extract_features` directly and saves each feature's top-20 count and rate per 1,000 prompts to `monitor/baselines/gpt2_8.json` with its metadata. 2,488 distinct features appear; 732 of them only once. Reloading in a fresh process gives the same numbers, and a full rebuild gives identical counts. The build takes about 75 s on CPU. 6 new tests; 86 pass in total, including the 3 model tests.
Surprise: only one feature (13586, "end-of-document marker", 522/1000) is in more than half of the top-20 lists. Two hashtag features (170.7 and 149.0 per 1,000) track the `#48213`-style order numbers we filled into the slots (512 of 3,000 prompts contain `#`), so some baseline features reflect our data preparation, not user intent.
Still open: feature 1017 ("Apple products", 183/1000) has no obvious cause in the prompts; features seen once or twice give noisy rates, so the drift test needs a minimum count or smoothing; the baseline is a single 3,000-prompt sample, so its sampling error is not yet measured.

## 2026-10-15
Deferred: open issues from the Milestone 3.3 baseline, to address later.
(1) Hashtag features 20921 and 6678 (and probably number features 15219 and 10536) track the `#48213`-style order numbers filled in by `prepare_data.py`. Normal scenario traffic (`normal_test`) has them too, so they should not cause false alarms, but shift pools lack them, so their drop may help detection for the wrong reason. Recommendation: leave the data as is; in drift evaluation, record the features behind each alert and add an ablation run without these features.
(2) 732 of the 2,488 baseline features appear only once, so their rates are noisy. Recommendation: handle in the drift test design (a minimum count, smoothing or a small-count-aware test); the baseline already stores raw counts.
(3) Baseline sampling error is unmeasured. Recommendation: before setting thresholds, build a second baseline from 3,000 disjoint `normal_train` prompts and compare the two (pure noise), then measure the false-alarm rate on `normal_only`.
(4) Feature 1017 ("Apple products", 183/1000) is unexplained. Recommendation: a quick look at its top-activating prompts in `normal_train`; low priority.

## 2026-10-15
Completed Milestone 4.1 (per-feature drift test): `monitor/drift.py` tests every feature seen at least `min_count` (default 5) times in baseline or window with a two-sided pooled two-proportion z-test on raw counts, adjusts p-values with Benjamini-Hochberg (`scipy.stats.false_discovery_control`) and returns features ranked by adjusted p, with add-one smoothed rates per 1,000 and their ratio. scipy is now a direct dependency. 4 new tests on synthetic counts (2,000 features, 3,000 requests each side); 90 pass in total.
Seed sweep (200 seeds, same synthetic setup): pure noise gave about 84 uncorrected p < 0.05 hits per run out of about 1,585 tested features, while BH flagged anything in only 4 of 200 runs (2%, within the 5% it allows). The injected 10x feature (1% to 10%) was ranked first and flagged in 200 of 200 runs.
Surprise: without correction, every noise run would raise roughly 84 false alerts, which shows the multiple comparisons problem is not theoretical at this scale.
Still open: the z-test is approximate for small counts (min_count only partly guards this); synthetic counts are independent per feature, but real top-20 features are correlated, which BH tolerates for positive correlation but is untested here; the real false-alarm rate on `normal_only` traffic is not yet measured.

## 2026-10-15
Deferred: open issues from the Milestone 4.1 per-feature drift test, to address later.
(1) The pooled z-test is an approximation that gets worse with small counts (a feature at 5 vs 0 is near its limit). min_count=5 only partly guards this. Recommendation: when measuring false alarms on real traffic, compare against Fisher's exact test (scipy.stats.fisher_exact) on the flagged features; switch if they disagree often.
(2) The synthetic tests draw each feature independently, but real top-20 features are correlated (they compete for the same 20 slots and co-occur by topic). BH is proven for independent or positively correlated tests; negative correlation from slot competition is not covered. Recommendation: measure the real false-alarm rate (item 3) instead of relying on the theory; if it is above 5%, try the Benjamini-Yekutieli method (method="by"), which holds under any correlation but is stricter.
(3) The false-alarm rate on real normal traffic is unmeasured. Recommendation: build windows from the `normal_only` scenario (and from the second disjoint baseline suggested in 3.3), run feature_drift over many windows and record the fraction with any flag.
(4) min_count=5 and alpha=0.05 are untuned defaults, and the window size N needed to detect a 10x shift on real features is unknown. Recommendation: sweep min_count and window size on the four drift scenarios during drift evaluation and report detection delay against the false-alarm rate.

## 2026-10-15
Completed Milestone 4.2 (overall drift score and threshold): `monitor/drift.py` now turns feature counts into distributions (each feature's share of all top-k slots) and scores a window with Jensen-Shannon divergence in bits (scipy's `jensenshannon(base=2)` squared, since scipy returns the square root). Window size and calibration settings live in `monitor/config.yaml`. `monitor/calibrate.py` extracts features for all 6,008 `normal_test` prompts once (about 2 min on CPU, cached and not committed), splits them in half with seed 0, sets the threshold at the 99th percentile of 1,000 random windows from the calibration half, and measures the exceed rate on 1,000 windows from the held-out check half. Saved to `monitor/baselines/gpt2_8_threshold.json`: threshold 0.0591 for window 500; check half exceeds it in 1.1% of windows (11 of 1,000). A rerun gives the same numbers. Plot: `eval/results/normal_jsd.png`. 15 new tests; 105 pass.
Window sweep (threshold / check-half exceed rate): 100 → 0.1646 / 0.8%; 250 → 0.0908 / 2.6%; 500 → 0.0591 / 1.1%; 1000 → 0.0389 / 0.8%.
Surprise: normal windows never score near 0 (median 0.054 at window 500), and the median falls as the window grows (0.145 at 100, 0.036 at 1000). A small window simply cannot contain the baseline's many rare features, so the score has a size-dependent floor; the threshold is only valid for its window size, and `load_threshold` refuses a mismatch.
Still open: window 250's 2.6% is well above 1% (about 1 in 1,000 windows of noise would be ±0.3%), which suggests the 99th percentile estimated from 1,000 overlapping windows of only ~3,000 prompts is unstable; seed variation of the threshold is unmeasured. Random windows are not consecutive traffic, though normal_test is already shuffled. Detection speed on the drift scenarios is not yet measured.

## 2026-10-15
Threshold stability check (`eval/threshold_stability.py`, results in `eval/results/threshold_stability.json`): reran the 4.2 calibration with seeds 0-19 (each seed is a different half-split of normal_test and different windows) at window sizes 100/250/500/1000. Thresholds barely move (window 500: 0.0581-0.0602), but held-out exceed rates do: mean 1.2% / 1.4% / 1.8% / 4.2%, worst seed 3.2% / 3.9% / 8.2% / 30.2%, and 3 / 5 / 7 / 9 of 20 seeds above 2%. Swapping the halves at seed 0 gives 0.4% / 0.8% / 0.6% / 0.9%. So the 1.1% reported for window 500 in 4.2 was a favourable split, not a stable property.
Cause (checked): each half of ~3,000 prompts sits at its own small distance from the baseline, and all its windows share that offset. The gap between the two halves' median scores spans -0.0023 to 0.0018, as large as the spread of windows within a half (sd 0.0020 at window 500, 0.0011 at 1000); the gap correlates with the exceed rate (r = 0.81 at 500, 0.72 at 1000). Large windows overlap more, hide more of the real variation and are hit hardest.
Surprise: the windows within one half underestimate how much fresh normal traffic varies, so a 99th percentile from them is too low; the 1% target is not met on average for any window size above 100.
Still open: the saved threshold (0.0591) is unchanged and likely gives about 2% false alarms at window 500, with a bad split up to 8%. Recommendation: calibrate on a larger pool the baseline never saw (the ~11,000 normal_train prompts outside the baseline sample), check on normal_test across seeds, and report the false-alarm rate as a range.

## 2026-10-15
Deferred: random versus consecutive windows. Calibration draws random windows from normal_test; production windows are consecutive requests. normal_test is already shuffled, so they should behave the same. Recommendation: verify during drift evaluation by running the normal_only scenario through the real pipeline and counting alerts over consecutive windows.

## 2026-10-15
Recalibrated the 4.2 threshold on a bigger pool: `monitor/calibrate.py` now sets the threshold on windows from the 11,017 normal_train prompts left out of the baseline sample and checks it on all 6,008 normal_test prompts (both feature caches uncommitted; extraction about 4 min). Saved threshold for window 500: 0.0595 (was 0.0591), seed 0 exceed rate on normal_test 0.6%. `eval/threshold_stability.py` now runs 20 seeds and also checks each threshold on a random 3,000-prompt subset of normal_test, which stands in for a fresh batch with its own offset. Window 500 exceed rate: mean 0.7%, range 0.1-1.6% on full normal_test (was mean 1.8%, worst 8.2% with half-splits); 0.6%, range 0.0-2.4% on 3,000-prompt subsets. Other sizes (full normal_test mean, range): 100 → 1.2%, 0.3-3.1%; 250 → 0.9%, 0.3-1.8%; 1000 → 0.3%, 0.0-0.9%. 105 tests pass.
Surprise: the threshold itself moved by only 0.0004; almost all of the old problem was in the check, where a small half-split shared one offset. Large windows now land below 1% (window 1000: 0.3%), so the threshold errs on the quiet side there; seed 0 happens to be the worst seed for window 100 (3.1%).
Still open: the false-alarm rate should be quoted as a range (window 500: about 0.1-2.4% depending on the batch), not as "1%". The 4.3 embedding detector must use these same two pools to stay comparable.

## 2026-10-15
Completed Milestone 4.3 (embedding baseline detector): `monitor/embed_drift.py` embeds prompts with all-MiniLM-L6-v2 (CPU, unit length), fits k-means (k=50, seed 0, n_init 10) on the same 3,000 normal_train prompts the SAE baseline used, and stores each prompt's cluster as a one-item ID list. That has the same shape as an SAE top-k list, so it goes through the same shared code: `score_window` (new in `drift.py`), `calibrate_sizes`, `pool_prompts` and `print_sweep` (pulled out of `calibrate.py`), with the same pools, window sizes and seeds. The SAE calibration reproduces its committed threshold exactly after the refactor. Embedding threshold for window 500: 0.0338 (median normal 0.0215), check-pool exceed rate 1.0% at seed 0 (100 → 1.5%, 250 → 1.0%, 1000 → 0.5%). On the same window (the first 500 normal_test prompts), SAE scores 0.0532 against 0.0595 and embeddings score 0.0236 against 0.0338; neither alerts. 111 tests pass.
Surprise: embedding all ~20,000 prompts on CPU took about 15 s, far faster than SAE extraction (about 4 min). k-means centroids differ by about 2e-15 between runs with the same seed (parallel summation), but cluster assignments match, so the test checks with a tolerance.
Still open: raw JSD values are not comparable across detectors, because a window holds 10,000 SAE counts but only 500 cluster counts (one cluster per prompt), so each detector has to be judged against its own threshold. The embedding threshold's seed stability is unmeasured (20-seed study not yet run for it), and k=50 is untuned. Neither detector has been run on the drift scenarios yet.

## 2026-10-15
Completed Milestone 4.4 (scheduler, alerts, first run): the app runs `monitor/job.py` every 60 s with APScheduler. It scores the latest 500 requests with both detectors through the shared `score_window`, and on a crossing stores an alert with the top 5 drifting IDs from `feature_drift` (SAE: Neuronpedia labels; clusters: example prompts). `GET /alerts` lists alerts. `eval/first_run.py` sends each scenario through the real app on a fresh database; results are in `eval/results/first_run.json`. 122 tests pass. This is a rough first look from one run per scenario, not the final result.
First run: normal_only is clean, with 0 alerts from either detector in 10 checks (highest SAE 0.0585 vs threshold 0.0595, embedding 0.0289 vs 0.0338). First alert by request number (SAE / embedding): medical_ramp 749 / 1172, new_topic_ramp 769 / 1550, injection_ramp 758 / 1717. The ramp starts at request 500, so the embedding detector alerted once 12-28% of the window was shifted.
Surprise: the early SAE alerts (750-770, with only 1-2% of the window shifted) are probably not real detections. All scenarios use seed 42, so they share the same normal prompt order. That normal stretch already scored 0.0582 in normal_only, and 7-11 shifted prompts were enough to tip it over. The top features at those alerts ("knock", currency, pricing) are the same across scenarios and unrelated to the shift. After that, the SAE score does rise steadily with the ramp and stays ahead of the embedding score in all three.
Surprise: APScheduler skipped 20 checks across the run ("missed by" up to 54 s). By default it drops a run that starts more than 1 s late, which happens while the server is busy with a request. injection_ramp got only 5 checks, with a gap from request 932 to 1717, so its embedding first alert (1717) is late by an unknown amount.
Still open: set `misfire_grace_time=None` so late checks run instead of being skipped. Then rerun each scenario with several seeds and measure detection delay against the false-alarm rate offline, at every request, instead of at wall-clock ticks, before drawing any conclusion about which detector is faster.
