# Deployment

The live demo is the Hugging Face Docker Space
[Shedrach/feature-monitor](https://huggingface.co/spaces/Shedrach/feature-monitor)
(app at <https://shedrach-feature-monitor.hf.space>). Every commit on `main` that passes CI
is deployed to it automatically.

## Run locally

Without Docker (model on CPU, database in `data/`):

```sh
uv sync
uv run uvicorn backend.app:app --port 8000   # open http://localhost:8000/
```

In Docker, the same image the Space runs (database on a named volume, so it survives restarts):

```sh
FEATURE_MONITOR_RECORD_TOKEN=some-secret docker compose up --build   # open http://localhost:7860/
```

Without the token the container still serves the dashboard and the Playground, but refuses
recorded prompts (`403`), as on the Space. See the README for details.

## How a deploy works

1. A pull request is merged into `main`, which is a push to `main`.
2. CI (`.github/workflows/ci.yml`) runs on that push: lint, tests and a Docker build.
3. When CI finishes, `.github/workflows/deploy.yml` starts. It does nothing unless CI
   succeeded on a push. It checks out the exact commit CI tested (not whatever `main` points
   to by then) and runs the official
   [`huggingface/hub-sync`](https://github.com/huggingface/hub-sync) action, pinned to a commit.
4. hub-sync uploads the repository's files to the Space with the `hf` CLI (pinned to 2.2.0).
   It skips `.git*` files and `.github/`, and deletes Space files that are no longer in the
   repository, so the Space is a mirror of that commit. It is a file upload, not a `git push`;
   a `git push` would be refused because the Space wants binary files such as the PNGs in
   `docs/images/` stored with Xet.
5. The new commit on the Space makes Hugging Face build the `Dockerfile` (on amd64) and
   restart the container. The `README.md` header (`sdk: docker`, `app_port: 7860`) tells it
   to build a Docker image and send traffic to port 7860.

Only one deploy runs at a time; a newer one waits for the running one to finish. The Deploy
run's log prints the commit it deployed, and GitHub lists every deploy under the repository's
**Deployments** (environment `huggingface-space`). The Space's own commit history only says
"Sync from shedrachIkenna/feature-monitor via hub-sync".

A deploy finishing in GitHub Actions only means the files were uploaded. The image is built
afterwards on Hugging Face; watch the Space's **Logs** tab (Build, then Container) to see it
come up, and check `https://shedrach-feature-monitor.hf.space/health`.

### One-time setup (already done)

- **Space:** Docker SDK, CPU basic hardware, public. Creating a Docker Space needs a paid
  Hugging Face plan (PRO for a personal account); the CPU basic hardware itself is free.
- **Space secret** `FEATURE_MONITOR_RECORD_TOKEN`: the token that recorded prompts must send.
  The image refuses all recorded prompts without it.
- **Space secret** `FEATURE_MONITOR_ADMIN_TOKEN`: the token that starts demo scenarios.
  Without it `POST /admin/scenario` refuses everything.
- **GitHub secret** `HF_TOKEN`: a fine-grained Hugging Face token with write access to this
  Space only. It reaches hub-sync as an input and is passed to the `hf` CLI as an
  environment variable; GitHub masks it in logs and nothing in the workflow prints it.
- **GitHub variable** `HF_SPACE_ID`: `Shedrach/feature-monitor`.

To rotate the deploy token, make a new one with the same scope, replace the `HF_TOKEN`
secret, then delete the old token on Hugging Face.

## Running the demo

The image sets `FEATURE_MONITOR_DEMO_TRAFFIC=1`, so after every start the Space fills its own
database: normal prompts as fast as the model allows until 500 are stored (the dashboard shows
"Warming up: n / 500"), then one every 20 seconds. Drift checks start once 500 are stored.

To show an alert, start a scenario (2,000 requests; it pauses the trickle while it runs):

```sh
curl -X POST https://shedrach-feature-monitor.hf.space/admin/scenario \
  -H 'content-type: application/json' -H "X-Admin-Token: $FEATURE_MONITOR_ADMIN_TOKEN" \
  -d '{"name": "medical_ramp"}'
```

The names are the files in `traffic/scenarios/`: `normal_only`, `medical_ramp`,
`new_topic_ramp`, `injection_ramp`. The dashboard says which one is running and how far it
has got, so visitors can tell staged drift from real drift. A second start while one runs
gets `409`. A restart stops a scenario, and the data with it.

The Space's **Logs** tab (Container) shows one JSON line per request (`"event": "http"`) and
per demo prompt (`"event": "demo"`), plus the daily cleanup (`"event": "retention"`).

## What the free hardware means

- **Data resets.** The Space's disk is not persistent: the SQLite database in `/data` is
  wiped on every restart, redeploy and wake from sleep. Hugging Face sets `SPACE_ID` inside
  every Space; when it is set, `/health` returns `"storage_resets": true` and every dashboard
  page shows "Demo data resets on restart"; the demo traffic then refills it. Keeping data
  would mean attaching a Storage
  Bucket at `/data`, after which that notice would be wrong and would need changing.
- **Sleep.** An unused Space on free hardware goes to sleep and starts again (empty) on the
  next visit. The Hub API reports this Space's sleep timeout as 172,800 s (48 hours).
- **Size.** CPU basic is 2 vCPU and 16 GB of RAM.

## Roll back

**Usual way: revert on `main`.** Revert the bad commit in a pull request (on GitHub, the
merged PR's **Revert** button, or `git revert <sha>` on a branch), and merge it. CI and the
deploy run as for any change, and `main` still matches what is live.

**Faster: redeploy an older commit.** In GitHub, **Actions → Deploy → Run workflow**, and set
`ref` to the SHA of the last good commit (or a tag). This skips CI, so pick a commit that
passed it before. The Space then runs that commit while `main` still has the bad one, and the
next merge to `main` deploys `main` again, so follow up with a revert.

**If GitHub Actions is down.** Either pause the Space (**Settings → Pause**), or upload a
good commit yourself, from a clean checkout so that local files such as `.venv/` and `data/`
are not uploaded:

```sh
git worktree add ../fm-rollback <good-sha>
cd ../fm-rollback
uvx hf@2.2.0 auth login        # paste a token with write access to the Space
uvx hf@2.2.0 upload Shedrach/feature-monitor . --type space \
  --exclude ".git*" --exclude "*/.git/" --exclude "*/.github/" --delete "*"
cd - && git worktree remove ../fm-rollback
```

These are the same commands hub-sync runs.

A rollback does not bring back data: the restart wipes the database either way.
