# Feature Monitor Design

## Problem
Changes in incoming requests or model behavior may alter what a deployed language model does. This project tests whether monitoring internal model features can flag such changes, and compares that approach with monitoring prompt sentence embeddings. The comparison is an experiment; neither method is assumed to work better.

## Users
Model operators who need to see whether monitored behavior has shifted, and ML researchers who want to inspect feature changes and evaluate the monitoring method.

## How it works
1. For each request, read activations from one layer of GPT-2 Small.
2. Use a sparse autoencoder (SAE) to represent those activations as features.
3. Store the 20 most active features for the request.
4. On a schedule, compare recent feature frequencies with a baseline and raise alerts for detected drift.
5. Apply the same drift method to prompt sentence embeddings as a baseline for comparison.

## Drift scenarios
- Topic change: prompts shift to a different subject.
- Wording change: prompts express similar intent in different language.
- Model-behavior change: the model's responses or internal feature activity shift.

These are scenarios to evaluate; detection has not yet been established.

## Success measures
Evaluate whether drift is detected in each scenario, how often alerts occur without a scenario change (false alarms), and how long detection takes. Compare these measures for SAE features and prompt embeddings. No outcome or target threshold is claimed here.

## Out of scope
Identifying or explaining the cause of detected drift.

## Open questions
- How will each drift scenario be defined and generated reproducibly?
- What baseline window, alert threshold, and evaluation period will be used?
- What evidence and comparison results will be reported for each scenario?