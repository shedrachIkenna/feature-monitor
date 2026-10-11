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
3. Store the request's 20 most active features.
4. On a schedule, compare recent feature frequencies with a baseline window; alert on drift.
5. Run the same drift test on prompt sentence embeddings, as the comparison baseline.

## Drift scenarios
All changes are sudden: traffic switches at a known request, so delay can be measured.
1. Topic change: traffic switches between clearly different subjects (e.g. coding to
   medical). Closer topics come later.
2. Style change: the same requests, rewritten casually (slang, typos).
3. Framing change: the same requests, wrapped in jailbreak-style framing
   ("pretend you are...").
4. Model change: prompts stay the same, but a steering vector (a fixed direction added to
   the activations) is injected before layer 8, at several strengths.

In scenario 4, prompt embeddings cannot see the change by design. We report it as a test
of what only internal monitoring can see, not as a win over embeddings.

## Success measures
- Detection rate, delay (in requests after the switch), and false alarms (alerts on
  unchanged traffic).
- Fair comparison: each method's threshold is tuned to the same false alarm rate on
  unchanged traffic, then detection and delay are compared at that setting.
- Every scenario is run with 5 fixed random seeds; we report the spread.
- The project succeeds if it gives a fair, reproducible answer for each scenario, whichever
  method it favours. Null and negative results are reported.

## Out of scope
- Explaining the cause of drift. Changed features are shown with Neuronpedia labels as a
  hint only.
- Automatic actions such as rollback or blocking; the system only alerts.
- Embedding model responses.
- Load testing at production scale.
- Not yet: gradual drift, and models other than GPT-2 Small (Gemma 2 2B in Milestone 7).

## Open questions
- Which drift test (for example chi-squared on feature counts, or a distance between
  frequency distributions), and what baseline and recent window sizes?
- What false alarm rate to match the methods at?
- Which steering direction and strengths, and at which layer?
- Should operators treat style-only changes (scenarios 2 and 3) as real alerts?
- Which sentence embedding model to use for the baseline?
