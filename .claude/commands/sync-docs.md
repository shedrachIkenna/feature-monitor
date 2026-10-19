---
description: Check the project docs against the code and recent changes, and update what no longer matches
---

Check that the project docs still describe the code. Run this before committing a finished task.

1. See what changed: `git status`, `git diff` and, for committed work, `git log --stat` since
   the docs were last touched (`git log -1 --format=%h -- docs/`).
2. Check each doc against the code (read the code, do not rely on memory):
   - `docs/design.md`: how it works, drift test settings, scenarios and their status (built or
     planned), success measures, out of scope, open questions. Answered questions move out
     of "Open questions".
   - `docs/schema.md`: every table and column in `backend/storage.py`, and the conventions.
   - `docs/datasets.md`: pools, sizes and processing in `traffic/prepare_data.py`.
   - `README.md`: summary, results table and finding (must match `docs/writeup.md`), quick start, links.
   - `docs/usage.md`: endpoints, settings, how to run the dashboard, screenshots, offline experiments.
   - `docs/deployment.md`: Docker, the Space, deploys and rollbacks.
   - `docs/study-report.md`: the milestone table and the "out of date" notice at the top.
     If the task added something the report does not cover, extend the notice; a full
     refresh is its own task.
3. Every number written into a doc must come from code output or a results file; check it.
4. Fix what no longer matches, in the docs' existing style. Do not rewrite text that is
   still correct.
5. Report which docs were changed and why, or that none needed changes.

$ARGUMENTS
