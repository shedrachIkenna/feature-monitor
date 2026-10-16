# Feature Monitor: Study Report

Prepared 2026-10-11, last refreshed 2026-10-15 after Milestone 1.4. Covers Milestone 0
(scope and setup), Milestones 1.1 to 1.4 (model and SAE, extraction, labels, sanity check),
the review fixes, and the supporting experiments.

> **Out of date beyond Milestone 1.4 (checked 2026-10-16).** Everything after that (traffic
> pools, baseline, drift test, calibration, the embedding detector, storage, API, drift job,
> dashboard, first run, multi-seed evaluation, the stronger embedding baselines, the Docker
> image, CI and the `/chat` queue cap) is not
> covered here yet; see
> [log.md](log.md). Sections 1.8 and 1.9 ("not chosen yet") and Part 6 are superseded by
> [design.md](design.md). A full refresh is pending.

Every number in this report was either printed by code in this repository or measured
again while writing it. Where something has not been tested, the report says so.

**Where the project stands.**

| Milestone | Item | Status |
|---|---|---|
| 0 Scope and setup | 0.1 Repository and environment | Done |
| | 0.2 Design document | Done |
| | 0.3 README and project board | Done |
| 1 Feature extraction | 1.1 Load the model and SAE | Done |
| | 1.2 Prompt to top features | Done |
| | 1.3 Feature labels | Done |
| | 1.4 Sanity check notebook | Done (verdict: weak separation at layer 8) |
| | Further items | Not yet given |

**How to read this.** Part 1 builds up the ideas from scratch. Part 2 explains the tools.
Parts 3 to 6 walk through every file. Part 7 lists the problems found in the review and
how each was fixed.
Read in order the first time; later, use it as a reference.

---

## Part 1. The ideas, from the basics

### 1.1 What the project is trying to find out
A company runs a language model in production. Over time, two kinds of things can change
without anyone noticing quickly:
- **The traffic**: users start asking about different things, or asking in a different way.
- **The model**: someone updates it, and it starts computing differently.

The project asks: *can we notice these changes by watching what happens inside the model,
and is that any better than watching the prompts themselves?* It is an experiment. A
result of "no, the simple method is just as good" is a valid and reportable outcome.

### 1.2 Language model, tokens and BOS
- A **language model** is a program that reads text and predicts which piece of text
  comes next. We use **GPT-2 Small**, a small, old, freely available model from OpenAI.
- Models do not read letters or words; they read **tokens**, which are chunks of text
  (often a word or part of a word). GPT-2 has 50,257 different tokens.
  Example, measured on this machine:
  `"How do I reset my password?"` becomes
  `['<|endoftext|>', 'How', ' do', ' I', ' reset', ' my', ' password', '?']`.
- The first token, `<|endoftext|>`, is the **BOS token** ("beginning of sequence"). It is
  added automatically so the model has a fixed starting point. It is not part of what the
  user wrote. This matters later (section 3.3).

### 1.3 Layers, the residual stream and activations
- GPT-2 Small is built as a stack of 12 **layers** (also called *blocks*), numbered 0 to 11.
  Each layer takes the model's current "working notes" and refines them.
- Those working notes are called the **residual stream**. For each token, it is a list of
  768 numbers (the model's width, `d_model = 768`). Every layer reads from it and adds to it.
- The numbers inside the model at any point are called **activations**. "Reading the
  activations at layer 8" means: copy the 768 numbers per token from the residual stream at
  that point.
- Our exact read point is `blocks.8.hook_resid_pre`: the residual stream *just before*
  layer 8 runs (`pre` = before). A **hook** is a named point in the model where code can
  look at, or change, the activations. TransformerLens (Part 2) provides these hooks.

### 1.4 Why not just look at the 768 numbers?
The 768 numbers are hard for humans to interpret. Any single number mixes many ideas
together. You cannot point at number 412 and say "that is the medical number".

### 1.5 Sparse autoencoders (SAEs) and features
- An **autoencoder** is a small neural network trained to take an input, convert it to
  another form, and then rebuild the original input from that form. Converting is called
  **encoding**; rebuilding is called **decoding**.
- A **sparse autoencoder (SAE)** encodes the 768 numbers into a much longer list, here
  24,576 numbers, with one rule: almost all of them must be zero. **Sparse** means "mostly
  zeros". Measured on the password prompt, each token had only 23 to 79 non-zero entries
  out of 24,576.
- Each of the 24,576 positions is called a **feature**. The number of features is the
  **dictionary size** (`d_sae = 24,576`). Because only a few are active at once, each
  feature tends to stand for one recognisable thing ("medical vocabulary", "question
  mark", "Python code"). That is what makes features easier to interpret than raw
  activations.
- A feature's **activation** is its number for a given token. Zero means "not present";
  bigger means "more strongly present".
- We use a pre-trained SAE called `gpt2-small-res-jb` (trained by Joseph Bloom on GPT-2's
  residual stream), loaded with the `sae-lens` library.

### 1.6 How good is the SAE's rebuild?
An SAE is not perfect. A standard way to measure it is **FVU, the fraction of variance
unexplained**: how much of the original activations' variation the rebuild misses.
0 means perfect; 1 means it captured nothing. On one test prompt measured during this
review, FVU was **0.18**: the SAE rebuilt about 82% of the variation. That is a single
prompt, not a general measurement.

### 1.7 Feature labels and Neuronpedia
A feature is just a number like `18171`. To know what it means, people run it over lots
of text, see when it fires, and write a short description. **Neuronpedia** is a public
website that stores such descriptions. Many of them were written automatically by
another language model, so they can be wrong or vague. Example from our data:
feature 18171 is labelled *"medical terms related to various types of "itis" and
inflammation conditions"*.

A feature can have several explanations, written by different models. Some are
**scored**: another model checks how well the explanation predicts where the feature
fires, giving a number from 0 to 1. Labels are **hints, not explanations**: on our 14
cached features, the two explanation sources disagreed on 13, sometimes completely
(section 3.4).

### 1.8 Drift, baselines and alerts
- **Drift** means the statistics of something change over time. Here: which features
  appear, and how often.
- A **baseline** is a reference period of "normal" traffic. A **baseline window** is that
  period, for example the first 2,000 requests.
- A **recent window** is the latest stretch of traffic, for example the last 500 requests.
- A **drift test** compares the two windows and produces a score. If the score passes a
  **threshold**, the system raises an **alert**. The drift test itself is not chosen yet.
- A **false alarm** is an alert when nothing actually changed.
- **Detection delay** is how many requests pass between the real change and the first
  alert.
- Setting the threshold is a trade-off. Lower thresholds catch more real changes but also
  raise more false alarms. That is why the design compares methods *at the same false alarm
  rate* (section 6.2).

### 1.9 The comparison method: sentence embeddings
A **sentence embedding** is a fixed-length list of numbers that summarises what a
sentence means, produced by a model trained for that purpose. Sentences with similar
meanings get similar lists. The project applies the same drift test to embeddings of the
prompts, to see whether the complicated SAE method adds anything. The embedding model is
not chosen yet.

### 1.10 Steering vectors
A **steering vector** is a fixed list of 768 numbers that is *added* to the residual stream
at some layer, on every request. It pushes the model's computation in one direction. We
will use it to simulate a model change while keeping the prompts identical.

### 1.11 Random seeds
Programs that use randomness (shuffling, sampling) give different results each run. A
**seed** is a starting number for the random number generator. Fixing it makes results
repeatable. The sanity check (Part 5) uses no randomness, so it needs no seed. The drift
experiments will.

---

## Part 2. The tools

| Tool | What it is | How this project uses it |
|---|---|---|
| **Python 3.12** | The programming language | Pinned in `.python-version` |
| **uv** | Installs Python packages and manages project environments | All installs go through `uv add`, never `pip` |
| **.venv** | A **virtual environment**: a private folder of installed packages for this project only | Created by uv; not committed |
| **pyproject.toml** | The project's settings file: name, Python version, dependencies, tool settings | See 2.1 |
| **uv.lock** | The **lock file**: the exact version of every package, including indirect ones | Lets anyone recreate the identical environment with `uv sync` |
| **PyTorch (`torch`)** | The maths library neural networks run on | Runs GPT-2 and the SAE |
| **TransformerLens** | A library that loads models like GPT-2 with hooks everywhere | `TransformerBridge`, `run_with_cache` |
| **sae-lens** | A library that downloads and runs pre-trained SAEs | `SAE.from_pretrained`, `sae.encode` |
| **pytest** | Runs tests: small programs that check the code does what it should | `uv run pytest` |
| **ipykernel, nbconvert** | Run notebook code; execute a notebook from the terminal | `uv run jupyter nbconvert --execute` |
| **ruff** | Two tools in one: a **linter** (finds code problems) and a **formatter** (fixes layout) | `ruff check`, `ruff format` |
| **make / Makefile** | Short names for longer commands | `make test`, `make test-fast`, `make lint`, `make format` |
| **git** | Saves snapshots (**commits**) of the project's history | One focused commit per step, credited to Shedrach Nwali; no remote yet |

Planned but **not installed yet**: FastAPI (web API), SQLite (database), APScheduler
(scheduled jobs), a sentence-embedding library, Docker, GitHub Actions.

### 2.1 `pyproject.toml`, explained
```toml
[project]
name = "feature-monitor"
requires-python = ">=3.12"          # Python 3.12 or newer (.python-version pins 3.12)
dependencies = ["sae-lens>=6.54.0", "torch>=2.14.1", "transformer-lens>=3.9.0"]

[dependency-groups]
dev = ["ipykernel>=7.4.0", "nbconvert>=7.17.1", "pytest>=9.1.1", "ruff>=0.16.10"]
# dev = needed to develop, not to run

[tool.ruff]
line-length = 100                    # lines longer than 100 characters are flagged

[tool.ruff.lint]
extend-select = ["I"]                # also check import ordering ("I" = isort rules)

[tool.pytest.ini_options]
markers = ["requires_model: loads GPT-2 and the SAE, downloading them on first run"]
```
A **marker** is a tag on a test. `make test-fast` uses this one to skip the slow tests
(Part 4).
Installed versions right now: sae-lens 6.54.0, transformer-lens 3.9.0, torch 2.14.1.

### 2.2 Linter versus formatter
This caught us once, so it is worth spelling out:
- `ruff check .` looks for *mistakes and bad practice*: unused imports, wrong import order,
  and so on.
- `ruff format --check .` checks *layout only*: spacing, line wrapping, final newlines.

They are independent. Earlier today `ruff check` passed while `ruff format --check`
wanted to change 8 files, all of which lacked a newline at the end. CI should run both.

### 2.3 `.gitignore`
This file lists what git should never save: Python's compiled caches (`__pycache__/`),
tool caches, `.venv/`, macOS junk (`.DS_Store`), model download folders, SQLite databases
(`*.db` plus its side files `*.db-shm` and `*.db-wal`), and `monitor/labels_cache.json`.

Models are not stored in the project. They download once to
`~/.cache/huggingface/` in your home folder, outside the repository.

---

## Part 3. The code, file by file

### 3.1 Repository map
```
Feature_monitor/
├── CLAUDE.md                 Rules for AI assistants working on the project
├── README.md                 Short public description and quick start
├── Makefile                  test / lint / format shortcuts
├── pyproject.toml, uv.lock   Dependencies and settings
├── .python-version           Pins Python 3.12
├── .gitignore
├── monitor/                  ← the only package with real code so far
│   ├── model.py              Loads GPT-2 + SAE
│   ├── extract.py            Prompt → top 20 features
│   ├── labels.py             Feature ID → human-readable label (Neuronpedia)
│   └── labels_cache.json     Saved labels (not committed)
├── eval/
│   ├── sanity.py             The topic-separation rule (counts, enrichment, verdict)
│   ├── mps_check.py          MPS-versus-CPU check (section 3.5)
│   ├── 01_sanity_check.ipynb The medical-vs-coding experiment, two prompt sets
│   └── data/
│       ├── sanity_prompts.json          20 hand-written prompts
│       └── sanity_prompts_matched.json  20 phrasing-matched prompts
├── docs/
│   ├── design.md             One-page design
│   ├── log.md                Build log
│   ├── study-report.md       This file
│   └── upstream-warnings.pdf The two sae-lens warnings, for an upstream contribution
├── backend/  traffic/  frontend/   Empty placeholders (not started)
└── tests/                    Mirrors the folders above
```
An `__init__.py` file, even an empty one, tells Python that a folder is a **package**, so
code can write `from monitor.model import ...`. `frontend/` is HTML and JavaScript, not
Python, so it has a `.gitkeep` instead. Git does not save empty folders, so a placeholder
file keeps it in the history.

### 3.2 `monitor/model.py`: loading the model and SAE
```python
MODEL_NAME = "gpt2"  # Hugging Face name for GPT-2 Small
SAE_RELEASE = "gpt2-small-res-jb"  # which family of SAEs
SAE_ID = "blocks.8.hook_resid_pre"  # which one: layer 8, residual-pre


@lru_cache(maxsize=1)
def load_model_and_sae() -> tuple[TransformerBridge, SAE]:
    device = "cpu"
    model = TransformerBridge.boot_transformers(MODEL_NAME, device=device)
    model.enable_compatibility_mode()
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=device)
    return model, sae
```
- **`TransformerBridge`** wraps the standard Hugging Face GPT-2 and adds TransformerLens
  hooks. It replaced the deprecated `HookedTransformer.from_pretrained` on 2026-10-11.
  **Deprecated** means "still works, but will be removed in a future version".
  `enable_compatibility_mode()` applies the same weight processing as the old loader,
  including `center_writing_weights=True`, which the SAE was trained with. On all 40 sanity
  prompts the two loaders gave bit-for-bit identical layer-8 activations, and
  `test_bridge_matches_deprecated_loader` keeps checking this.
- **`@lru_cache(maxsize=1)`** is a **decorator**: a line starting with `@` that wraps a
  function with extra behaviour. This one remembers the result. The first call loads the
  model, which takes seconds; every later call returns the same objects instantly. Without
  it, every request would reload GPT-2.
- **`device = "cpu"`**: a **device** is the hardware that does the maths. The Mac has a GPU
  reachable through **MPS** (Apple's GPU interface). The installed TransformerLens treats
  no PyTorch version as MPS-safe yet, so CPU is the cautious default; see section 3.5 for
  our own MPS check. Docker on a Mac has no GPU anyway.
- `-> tuple[TransformerBridge, SAE]` is a **type hint**: it documents what the function
  returns. CLAUDE.md requires these on public functions.
- `main()` prints the sizes, so you can run `uv run python -m monitor.model` as a quick
  check. Output: `Model: gpt2`, `Device: cpu`, `d_model: 768`, `d_sae: 24576`.

**Warnings you will see when this runs.** Both come from inside sae-lens, not our code, and
are documented in `docs/upstream-warnings.pdf`:
1. *"The 'utils' module has been deprecated …"*. sae-lens imports from an old TransformerLens
   location. A fix is waiting upstream (SAELens PR #739).
2. *"This SAE has non-empty model_from_pretrained_kwargs …"*. The SAE was trained on GPT-2
   loaded with `center_writing_weights=True`. We checked that our loading matches: FVU was
   **0.1845** both ways on the same prompt. The warning is harmless for us.

### 3.3 `monitor/extract.py`: from prompt to top 20 features
```python
@dataclass(frozen=True)
class FeatureHit:
    feature_id: int  # which feature, 0 to 24,575
    activation: float  # how strongly it fired
```
A **dataclass** is a quick way to define a small record type. `frozen=True` means it
cannot be changed after creation, which makes it safe to compare and share.

`extract_features(prompt, k=20)`, step by step:

| Line | What it does | Why |
|---|---|---|
| `if k < 0: raise …` / `if k == 0: return []` | Rejects nonsense input | Defensive edge cases |
| `hook_point = sae.cfg.metadata.hook_name` | Asks the SAE where it was trained to read | Avoids hard-coding the hook name twice |
| `tokens = model.to_tokens(prompt, prepend_bos=True)` | Text → token IDs, adding BOS at the start | The SAE was trained with BOS present |
| `with torch.no_grad():` | Turns off gradient tracking | **Gradients** are only needed for training. Turning them off saves memory and time |
| `model.run_with_cache(tokens, names_filter=[hook_point])` | Runs GPT-2 and saves activations, but only at our hook | Saving all 12 layers would waste memory |
| `sae.encode(activations)[0, 1:, :]` | Encodes to 24,576 features per token, then keeps batch item 0 and **drops token 0 (BOS)** | See below |
| `.amax(dim=0)` | For each feature, takes its **maximum** over all tokens | Turns "features per token" into "features per prompt" |
| `torch.nonzero(max_activations > 0)` | Keeps only features that fired at all | Avoids returning zeros as "top" features |
| `torch.topk(..., count)` | Picks the `k` largest, sorted high to low | The "top 20" |
| Returns a list of `FeatureHit` | | Plain Python objects, easy to store |

**Tensor shapes.** A **tensor** is a grid of numbers; its **shape** lists the size of
each direction, so `(1, 8, 768)` means 1 × 8 × 768 numbers. Measured on
"How do I reset my password?":

| Step | Tensor | Shape | What each position means |
|---|---|---|---|
| Text → tokens | `tokens` | (1, 8) | 1 prompt (the **batch** size: prompts processed at once) × 8 tokens (BOS + 7 real) |
| Read at `blocks.8.hook_resid_pre` | `activations` | (1, 8, 768) | per token, the 768 residual-stream numbers (`d_model`) |
| `sae.encode` | `features` | (1, 8, 24576) | per token, 24,576 feature strengths (`d_sae`), almost all zero |
| SAE encoder weights | `W_enc` | (768, 24576) | the table that turns 768 numbers into 24,576 |
| SAE decoder weights | `W_dec` | (24576, 768) | the table that turns them back |
| `[0, 1:, :]` | | (7, 24576) | prompt 0, BOS dropped, all features |
| `.amax(dim=0)` | `max_activations` | (24576,) | each feature's maximum over the 7 tokens |
| `topk` | | (20,) | the 20 strongest features |

The SAE's input size (the 768 in `W_enc`) must equal the model's `d_model` (768). If they
differed, the SAE would have been built for a different model. `test_model.py` checks this.

**Why drop BOS?** Measured on all 40 sanity prompts at layer 8:
- BOS's residual-stream vector is about **25 times larger** than any other token's
  (median 24.8×, range 22 to 28). Its strongest SAE feature is about **10 times**
  stronger than any real token's. On the password prompt that is **502** versus **51.6**.
- BOS's vector is **exactly identical in every prompt** (difference 0.0).

Why this happens:
1. **BOS cannot see the prompt.** GPT-2 reads left to right: each token can only look at
   itself and the tokens *before* it. BOS comes first, so it only ever sees itself. Its
   numbers are the same whatever the user writes, so **it carries no information about
   the prompt**. Our measurement proves this part.
2. **The model uses it as a "parking spot".** Each token decides how much to **attend to**
   (look at) each earlier token, and those attention weights must add up to 100%. When
   nothing earlier is useful, models learn to send the spare attention to the first token,
   which is called an **attention sink**. In that role the first position develops a very
   large vector, called **massive activations**. This explanation comes from published
   research (attention sinks, Xiao et al. 2023; massive activations, Sun et al. 2024); we
   have not tested it ourselves.

If BOS were kept, the same 8 BOS features would fire hugely in every request. They would
take top-20 slots while carrying no information, because they are identical everywhere.

**Max versus mean.** These are two ways to combine a feature's per-token values into one
number per prompt (called **pooling**):
- **Max** asks "did this feature fire strongly *anywhere*?", so a concept mentioned once
  (one medical word in a long question) still counts fully.
- **Mean** averages over all tokens, so that single mention is diluted by every other word,
  and the longer the prompt, the weaker it gets.
- **What max loses:** *how much* of the prompt the feature covers. A feature firing on one
  token looks the same as one firing on every token, and one unusually high token can
  dominate.

**Speed.** Once the model is loaded, extraction takes a median of **17.2 ms** per prompt on
CPU (120 calls, measured with `TransformerBridge`). The old `HookedTransformer` loader took
about 43 ms. The first call after loading is slower because of one-time warm-up work.

### 3.4 `monitor/labels.py`: feature ID to label
Goal: `get_label(18171)` → `"medical terms related to inflammation or clinical conditions"`.

How `get_label` works:
1. **Check the cache.** A **cache** is a saved copy of earlier answers, so we don't ask
   the website again. Here it is `monitor/labels_cache.json`, a dictionary from feature ID
   to label. If the ID is there, return it.
2. **Ask Neuronpedia.** Call the web **API** (an address that returns data for programs,
   not people):
   `https://www.neuronpedia.org/api/feature/gpt2-small/8-res-jb/{feature_id}`.
   The response is **JSON**, a standard text format for nested data.
3. **Handle failures by kind.** A **transient** error is temporary and might succeed a
   moment later: a timeout, a dropped connection, HTTP 429 ("too many requests") or a 5xx
   code (the server broke). These are retried once, after a **pause**: 1 second, or the
   server's `Retry-After` header (how long it asks us to wait), capped at 10 seconds.
   Retrying instantly would ignore what a rate limit is asking for. A **permanent** error won't change on
   retry: other HTTP errors such as 404 ("no such feature"), or a response that isn't
   valid JSON. These give up straight away. Either way the result is `"unlabelled"`, and
   **nothing is saved**, so a brief outage doesn't become a permanently missing label.
4. **Choose the label** with `_explanation_from_payload`. It reads only
   `explanations[].description`, plus each explanation's `scores`:
   - An explanation's score is the **average** of its score values (`_mean_score`).
   - The highest-scoring explanation wins, but **only if it scores at least 0.7**
     (`MIN_EXPLANATION_SCORE`). Otherwise the first explanation is used. On a tie, the
     earlier one wins.
   - The 0.7 cut-off is our own judgement, not a Neuronpedia standard. Scores near 0.5 are
     close to coin-flip agreement.
5. **Save** the label to the cache. A successful response with no explanation saves
   `"unlabelled"`, because that is a real answer.

**What we learned building this.** An earlier version searched the whole response for any
text field. It picked up `sourceSet.description`, the SAE release's title (*"Open Source
Sparse Autoencoders for all Residual Stream Layers of GPT2-Small"*), and 8 of the 14
cached labels were that title. It could also have picked up `jsonDetails[].text`. An
earlier log entry called that the explanation, but it is actually example text used to
score explanations. The strict parser fixed both problems, and the 8 bad entries were
re-fetched.

**Choosing between explanations (2026-10-15).** Each feature we checked had two
explanations. The first, by gpt-3.5-turbo, is never scored. The second, by gpt-4o-mini,
has two `eleuther_fuzz` scores. Switching to "highest-scored" changed 13 of the 14 cached
labels, and for feature 23838 it would have picked *"questions or discussions about
societal issues"*, scoring only **0.49**. The unscored alternative, *"interrogative
sentences ending with a question mark"*, fits the evidence: the tokens this feature makes
the model more likely to output next are all question marks. The 0.7 minimum keeps that
one, so 12 of 14 labels changed. The old labels are backed up outside the repository.
The cache stores only label text, so it was cleared to apply the new rule everywhere.

The `urllib` used here is part of Python itself, so no extra package is needed.

Running `uv run python -m monitor.labels "your prompt"` prints the top 10 features with
their labels, for example `3602: references to passwords and related security terminology`.

### 3.5 `eval/mps_check.py`: is MPS safe for our use?
TransformerLens treats no PyTorch version as MPS-safe, but the only *specific* bug it lists
is in PyTorch 2.8; we have 2.14.1. So we tested our own path directly.
`uv run python -m eval.mps_check` loads the model and SAE on CPU and on MPS and runs all 40
sanity prompts on both. The pass rule was fixed before the first run: identical top-20
feature sets on every prompt, and a largest activation difference of at most 0.1% of the
largest activation.

Result on 2026-10-11 (torch 2.14.1): **PASS**. Top-20 sets matched on 40/40 prompts. The
largest difference was 0.00073 on activations up to 3,010 (relative 2.4 × 10<sup>-7</sup>).
Speed over 120 timed calls after warm-up: median **17.2 ms on CPU** and **10.8 ms on MPS**.

The check covers one forward pass to layer 8 plus the SAE encoder. It does not cover text
generation or steering, which later milestones add.

---

## Part 4. The tests

Run all of them with `make test` (`uv run pytest`): **22 tests, all passing.** Two warnings
appear, both from inside sae-lens (section 3.2). Three tests are tagged `requires_model`
because they load GPT-2 and the SAE, which downloads them on the first run. `make
test-fast` skips those three and runs the other 19 in about 2.8 s, without any network
access.

| Test | What it checks | Loads the model? |
|---|---|---|
| `test_smoke.py::test_smoke` | `assert True`: proves pytest itself runs | No |
| `test_model.py::test_sae_input_size_matches_model_width` | Model is on CPU; the SAE's input size (768) equals the model's width (768) | Yes |
| `test_model.py::test_bridge_matches_deprecated_loader` | `TransformerBridge` gives the same layer-8 activations as the old `HookedTransformer` on two prompts. Delete it once TransformerLens 4.0 removes the old loader | Yes |
| `test_extract.py::…returns_top_valid_deterministic_hits` | Password prompt gives exactly 20 hits; `k=5` gives ≤5; IDs are in range; all activations > 0; sorted high to low; two runs give identical results (**deterministic**) | Yes |
| `test_labels.py::…fetches_and_caches_label` | A real-shaped response gives its first explanation, fetched once and then served from the cache | No (mocked) |
| `test_labels.py::…ignores_text_outside_explanations` | A response with only the release title and a `jsonDetails` snippet gives `"unlabelled"` | No (mocked) |
| `test_labels.py::…retries_once_then_returns_unlabelled` | Timeout, then HTTP 429: exactly 2 tries with one pause, nothing cached, and the next call tries again | No (mocked) |
| `test_labels.py::…does_not_retry_permanent_http_error` | HTTP 404: exactly 1 try and no pause | No (mocked) |
| `test_labels.py::…retries_server_error_once` | HTTP 503: exactly 2 tries, with a 1 s pause between them | No (mocked) |
| `test_labels.py::…honours_capped_retry_after` (2 cases) | HTTP 429 with `Retry-After: 3` pauses 3 s; `Retry-After: 120` is capped at 10 s | No (mocked) |
| `test_labels.py::…prefers_highest_mean_score` (7 cases) | Scored beats unscored; higher average wins; ties and no scores keep the first; text scores are ignored; 0.49 is rejected and 0.7 is accepted | No |
| `test_labels.py::…prints_top_prompt_features_with_labels` | The command-line output contains the IDs and labels | No (mocked) |
| `test_sanity.py::…counts_each_prompt_once` | A feature repeated within one prompt counts once | No |
| `test_sanity.py::…finds_enriched_features_and_jaccard` | Common, enriched and Jaccard values on a small hand-made example | No |
| `test_sanity.py::…clear_and_none_verdicts` | The "clearly" and "not at all" verdicts | No |

**Mocking.** The label tests never touch the real internet, and never really wait: an
automatic fixture replaces the pause with a recorder. They use pytest's
**`monkeypatch`**, which temporarily swaps a real function for a fake one during a test.
For example, the fake `urlopen` returns a prepared response instead of calling
Neuronpedia. This makes the tests fast, and they give the same result every time. A fake
like this is called a **mock**. `tmp_path` gives each test a fresh temporary folder, so
tests never touch the real cache file.

**Checking that a test can fail.** For each bug fix, the new test was first run against
the *old* code to confirm it failed there. A test that passes on broken code proves
nothing.

---

## Part 5. The sanity-check experiment (`eval/01_sanity_check.ipynb`)

A **notebook** is a document mixing code, its output and notes, run one **cell** at a time.
The counting rule lives in `eval/sanity.py`, so it can be tested; the notebook just calls
it. Re-run everything with
`uv run jupyter nbconvert --to notebook --execute --inplace eval/01_sanity_check.ipynb`.

### 5.1 Question
Before building a drift monitor on SAE features, check the most basic thing: *do medical
prompts and coding prompts even produce different features at layer 8?* If not, the whole
idea fails early.

### 5.2 Setup
- **Original set:** 10 medical prompts ("Can ibuprofen upset the stomach?") and 10 coding
  prompts ("How do I reverse a list in Python?"), hand-written.
- **Matched set:** 10 sentence templates, each filled once with a medical and once with a
  coding subject ("What causes a migraine?" / "What causes a memory leak?"). The only
  difference within each pair is the topic.
- Top 20 features per prompt.
- A feature is **common** in a group if it is in the top 20 of at least 5 of that group's
  10 prompts, and **rare** if in at most 1.
- A feature is **enriched** for a group if it is common there and rare in the other.
- **Jaccard overlap** measures how much two sets share: *(items in both) ÷ (items in
  either)*. 0 = nothing shared, 1 = identical.
- The rule was written **before** looking at results: "clear separation" needs at least 3
  enriched features in each group and Jaccard ≤ 0.25; "weak" means some enriched features
  but not clear.
- The prediction for the matched set was also written down before it was run: if
  phrasing drove the original separation, the remaining enriched features should be
  topical.

### 5.3 Results (from the notebook's saved output)

| | Original prompts | Matched prompts |
|---|---|---|
| Common features (medical / coding / shared) | 7 / 10 / 3 | 5 / 5 / 3 |
| Jaccard overlap | 0.214 | 0.429 |
| Medical-enriched | 2 | 1 |
| Coding-enriched | 4 | 0 |
| Verdict | weakly | weakly |

Enriched features in the **original** set:

| Enriched for | Feature | Medical | Coding | Neuronpedia label |
|---|---:|---:|---:|---|
| Medical | 18171 | 5/10 | 0/10 | medical terms related to various types of "itis" and inflammation conditions |
| Medical | 23838 | 5/10 | 1/10 | interrogative sentences ending with a question mark |
| Coding | 10590 | 1/10 | 5/10 | instances of the first-person pronoun "I" used in various contexts |
| Coding | 18200 | 1/10 | 6/10 | instances of the article "a" and its variations |
| Coding | 21629 | 1/10 | 5/10 | phrases that start with "How to." |
| Coding | 21981 | 1/10 | 5/10 | query-based phrasing or questions regarding knowledge and understanding |

In the **matched** set, only one feature remains: 18171 (medical terms), in 7/10 medical
prompts and 0/10 coding prompts.

### 5.4 What this actually tells us (read carefully)
- **Most of the original separation came from phrasing, not topic.** Five of the six
  original enriched features describe how the question is phrased. Once phrasing was held
  constant, all five disappeared, and only the topical feature 18171 remained.
- **The topic signal in common top-20 features at layer 8 is thin:** one medical feature,
  and no coding feature at all.
- **That does not show topic drift is undetectable.** This rule only counts features
  shared by 5 or more of 10 prompts. Coding subjects such as "recursion" and "deadlock" may
  each trigger their own specific features, which a drift test over the *whole* feature
  distribution could still pick up. Keep this in mind when choosing the drift test.
- **For the style and framing scenarios,** layer 8's sensitivity to phrasing suggests SAE
  features may react to wording changes that embeddings are designed to ignore.
  Untested; that is what the real experiment is for.
- **This conclusion does not depend on labels.** It rests on what the features *did*:
  five disappeared when only phrasing was held constant. The labels were re-chosen on
  2026-10-15 (section 3.4) and 5 of the 6 changed wording; all still describe phrasing
  except 18171.
- **Limits:** 20 prompts per set, one layer, hand-written prompts, labels that may be
  wrong. This is a quick check, not evidence of anything general.
- **Next step, proposed in the notebook's final cell:** repeat at layers 4 and 11. This
  needs the layer, now fixed at 8 in `monitor/model.py` (`SAE_ID`) and `monitor/labels.py`
  (`SOURCE_ID`), to become a parameter, with the label cache keyed by layer. Deferred on
  2026-10-15 in favour of the next milestone; layer 8 stays for now.

---

## Part 6. The design (`docs/design.md`) and why each choice was made

The design was rewritten after a 6-question interview. Summary of the decisions:

### 6.1 Users
Today the user is **us, the researchers**. The system is ultimately meant for an **on-call
ML engineer** (did an update break something? roll back?) and a **trust and safety
reviewer** (are users pushing the model somewhere risky?).

### 6.2 Four drift scenarios, all **sudden**
Sudden means the traffic switches at one known request. That makes delay easy to
measure: alert request number minus switch request number.

| # | Scenario | How it is created | Why |
|---|---|---|---|
| 1 | Topic change | Traffic switches between clearly different subjects (coding → medical) | Easiest case first; closer topics later |
| 2 | Style change | Same requests, rewritten casually (slang, typos) | Same meaning, different wording |
| 3 | Framing change | Same requests, wrapped in jailbreak-style framing | Relevant to the safety user |
| 4 | Model change | Same prompts; a steering vector is added **before** layer 8 at several strengths | A true model change, controllable and repeatable |

Why steering rather than the alternatives:
- A changed system prompt is really an *input* change, which blurs into scenarios 2 and 3.
- A fine-tuned model is realistic but much more work. It can come later.
- Steering must happen *before* layer 8, or the monitor cannot see it by construction.
- Testing several strengths finds the *smallest* change each method can catch, which
  says more than a single yes/no.

**The asymmetry, stated honestly.** The embedding baseline only sees prompts. In scenario 4
the prompts don't change, so embeddings *cannot* detect it. That is not a win for SAE
features. It shows what only internal monitoring can see. The design says this explicitly.

### 6.3 Success measures
- **Detection rate**, **delay** (counted in requests, because clock time depends on how
  fast our generator sends traffic), **false alarms**.
- **Matched false alarm rate.** Each method's threshold is tuned so both have the same
  false alarm rate on unchanged traffic; detection and delay are compared at that setting.
  Without this, a method could "win" simply by being more trigger-happy.
- **5 fixed seeds** per scenario; report the spread, not one lucky run.
- **Project success is about answering the question fairly, not about which method wins.**
  No result was chosen in advance as "the success".

### 6.4 Out of scope
Explaining *why* drift happened (labels are a hint only); automatic actions such as rollback;
embedding model responses; production-scale load testing. **Not yet:** gradual drift,
and models other than GPT-2 Small (Gemma 2 2B is planned for Milestone 7).

### 6.5 Still open
Which drift test and window sizes; which false alarm rate to match at; steering
direction, strengths and layer; whether style-only alerts count for operators; which
embedding model.

---

## Part 7. Issues found in the review, and how each was fixed

All 10 are resolved.

| # | Problem | What was done |
|---|---|---|
| 1 | Network failures were saved permanently as `"unlabelled"` | Failures are no longer cached; the next call retries |
| 2 | The retry fired on every error, unlike what the log described | Only transient errors (timeouts, connection errors, 429, 5xx) are retried; 404 and invalid JSON give up at once |
| 3 | The label search picked up any text in the response | It reads only `explanations[].description`. The generic-title workaround was removed, and 8 bad cache entries were re-fetched |
| 4 | Topic and phrasing were mixed together in the sanity check | Added the matched prompt set and moved the rule into `eval/sanity.py` with tests (Part 5) |
| 5 | Model tests needed the internet on the first run | Tagged `requires_model`; `make test-fast` skips them. CI should cache `~/.cache/huggingface` |
| 6 | A pass-through helper and a redundant assertion | Both removed |
| 7 | Deprecated `HookedTransformer` loader | Switched to `TransformerBridge`: bit-identical activations, warnings down from 4 to 2 |
| 8 | Commits credited to "Feature Monitor <feature-monitor@local>" | Local identity removed; all commits rewritten to credit Shedrach Nwali, with dates and files unchanged |
| 9 | `docs/setup-report.pdf` described the empty skeleton | Deleted. Recover it with `git show c5b023b:docs/setup-report.pdf` |
| 10 | No milestone list | Closed by decision: milestones are given one at a time |

---

## Part 8. What was done, in order

1. **Milestone 0.1 audit:** the skeleton checks all held. Pinned Python 3.12 and ran
   `ruff format` (8 files, layout only).
2. **Milestone 0.2:** reviewed the design doc, ran the 6-question interview and rewrote
   `docs/design.md`.
3. **First study report.**
4. **Review fixes 1 to 10** (Part 7), one commit each, with a log entry each time.
5. **Upstream warnings:** `docs/upstream-warnings.pdf`. Warning 1 is already being fixed
   in SAELens PR #739. Warning 2 has no issue yet and five concrete problems.
6. **Milestone 0.3:** README synced with the design; the first log entries dated.
7. **Milestone 1.1 audit:** all items hold, with two deliberate changes (TransformerBridge
   and CPU). Added the tensor-shapes table (section 3.3) and the MPS check (section 3.5).
   The default stays CPU: the gain is about 6 ms per request, and CPU keeps local results
   consistent with Docker.
8. **This refresh** of the study report.

---

## Part 9. What comes next

- **The next items of Milestone 1** (or whichever milestone you give next). Each will be
  checked against what already exists, because extraction and labels are already built.
- **Open questions** (from the log): which drift test to use, and whether the
  whole-distribution topic signal is stronger than the common-feature check suggests;
  layers 4 and 11 (the change needed is in section 5.4).
- **Optional upstream contributions:** test SAELens PR #739; open an issue about warning 2.
- **When CI is set up:** run `ruff format --check` as well as `ruff check`, and cache the
  model downloads.

---

## Appendix: command cheat sheet

| Command | What it does |
|---|---|
| `uv sync` | Install exactly what `uv.lock` lists |
| `uv add <pkg>` / `uv add --dev <pkg>` | Add a dependency (never `pip install`) |
| `make test` = `uv run pytest` | Run all 22 tests |
| `make test-fast` | Run the 19 tests that don't load the model |
| `uv run pytest tests/monitor/test_labels.py -v` | Run one test file, listing each test |
| `make lint` = `uv run ruff check .` | Look for code problems |
| `make format` = `uv run ruff format .` | Fix layout |
| `uv run ruff format --check .` | Check layout without changing anything |
| `uv run python -m monitor.model` | Print model and SAE sizes |
| `uv run python -m monitor.labels "a prompt"` | Top 10 features with labels |
| `uv run python -m eval.mps_check` | Compare MPS with CPU on the 40 sanity prompts |
| `uv run jupyter nbconvert --to notebook --execute --inplace eval/01_sanity_check.ipynb` | Re-run the sanity notebook |
| `git log --oneline` | List commits |
| `git status` | Show uncommitted changes |
