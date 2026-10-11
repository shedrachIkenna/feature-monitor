# Feature Monitor: Study Report

Prepared 2026-10-11. Covers everything in the repository up to and including commit
`92766a2`, plus the rewritten `docs/design.md`.

Every number in this report was either printed by code in this repository or measured
again while writing it. Where something has not been tested, the report says so.

**How to read this.** Part 1 builds up the ideas from scratch. Part 2 explains the tools.
Parts 3 to 6 walk through every file. Part 7 lists problems found during the review.
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
feature 18171 is labelled *"medical terms related to inflammation or clinical conditions"*.

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
| **TransformerLens** | A library that loads models like GPT-2 with hooks everywhere | `HookedTransformer`, `run_with_cache` |
| **sae-lens** | A library that downloads and runs pre-trained SAEs | `SAE.from_pretrained`, `sae.encode` |
| **pytest** | Runs tests: small programs that check the code does what it should | `uv run pytest` |
| **ruff** | Two tools in one: a **linter** (finds code problems) and a **formatter** (fixes layout) | `ruff check`, `ruff format` |
| **make / Makefile** | Short names for longer commands | `make test`, `make lint`, `make format` |
| **git** | Saves snapshots (**commits**) of the project's history | 4 commits so far |

Planned but **not installed yet**: FastAPI (web API), SQLite (database), APScheduler
(scheduled jobs), a sentence-embedding library, Docker, GitHub Actions.

### 2.1 `pyproject.toml`, explained
```toml
[project]
name = "feature-monitor"
requires-python = ">=3.12"          # Python 3.12 or newer (.python-version pins 3.12)
dependencies = ["sae-lens>=6.54.0", "torch>=2.14.1", "transformer-lens>=3.9.0"]

[dependency-groups]
dev = ["pytest>=9.1.1", "ruff>=0.16.10"]  # needed to develop, not to run

[tool.ruff]
line-length = 100                    # lines longer than 100 characters are flagged

[tool.ruff.lint]
extend-select = ["I"]                # also check import ordering ("I" = isort rules)
```
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
│   ├── 01_sanity_check.ipynb The medical-vs-coding experiment
│   └── data/sanity_prompts.json  Its 20 prompts
├── docs/
│   ├── design.md             One-page design (rewritten today)
│   ├── log.md                Build log
│   ├── setup-report.pdf      Report on the original skeleton (Oct 8; now out of date)
│   └── study-report.md       This file
├── backend/  traffic/  frontend/   Empty placeholders (not started)
└── tests/                    Mirrors the folders above
```
An `__init__.py` file, even an empty one, tells Python that a folder is a **package**, so
code can write `from monitor.model import ...`. `frontend/` is HTML and JavaScript, not
Python, so it has a `.gitkeep` instead. Git does not save empty folders, so a placeholder
file keeps it in the history.

### 3.2 `monitor/model.py`: loading the model and SAE
```python
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"  # which family of SAEs
SAE_ID = "blocks.8.hook_resid_pre"  # which one: layer 8, residual-pre


@lru_cache(maxsize=1)
def load_model_and_sae() -> tuple[HookedTransformer, SAE]:
    device = "cpu"
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=device)
    return model, sae
```
- **`@lru_cache(maxsize=1)`** is a **decorator**: a line starting with `@` that wraps a
  function with extra behaviour. This one remembers the result. The first call loads the
  model, which takes seconds; every later call returns the same objects instantly. Without
  it, every request would reload GPT-2.
- **`device = "cpu"`**: a **device** is the hardware that does the maths. The Mac has a GPU
  reachable through **MPS** (Apple's GPU interface), which is faster. The installed
  TransformerLens warns that MPS can *silently produce wrong numbers* with this PyTorch
  version, so the code uses the CPU. CLAUDE.md was updated to match. Docker on a Mac has
  no GPU anyway.
- `-> tuple[HookedTransformer, SAE]` is a **type hint**: it documents what the function
  returns. CLAUDE.md requires these on public functions.
- `main()` prints the sizes, so you can run `uv run python -m monitor.model` as a quick
  check. Output: `d_model: 768`, `d_sae: 24576`.

**Warnings you will see when this runs:**
1. *"HookedTransformer.from_pretrained is deprecated … use TransformerBridge"*.
   **Deprecated** means "still works, but will be removed in a future version". It will
   need migrating before TransformerLens 4.0. Not urgent.
2. *"This SAE has non-empty model_from_pretrained_kwargs …"*. The SAE was trained on GPT-2
   loaded with the setting `center_writing_weights=True`. I checked whether our loading
   matches by measuring FVU both ways on the same prompt. Both gave **0.1845**, so our
   default loading already matches. This warning is harmless for us.

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

**Why drop BOS?** It is a fixed token that appears in every prompt, and its activations
are huge. On the password prompt, the strongest feature on BOS was **502** versus **51.6**
for the strongest real token. If it were kept, the same BOS features would take the top
spots in *every* request and drown out the signal from what the user actually wrote.

**Max pooling.** Taking the maximum across tokens is called **max pooling**. It answers
"did this feature fire strongly *anywhere* in the prompt?" It loses *how many* tokens it
fired on. A feature that fires once and one that fires on every word look the same. This
is a known, deliberate simplification (it is listed as open in the log).

**Speed.** Once loaded, extraction takes about **0.043 s** per prompt on CPU. The first
prompt after loading is slower (0.17 to 0.18 s) because of one-time warm-up work.

### 3.4 `monitor/labels.py`: feature ID to label
Goal: `get_label(18171)` → `"medical terms related to inflammation or clinical conditions"`.

How `get_label` works:
1. **Check the cache.** A **cache** is a saved copy of earlier answers, so we don't ask
   the website again. Here it is `monitor/labels_cache.json`, a dictionary from feature ID
   to label. If the ID is there (and isn't the bad generic label, see step 4), return it.
2. **Ask Neuronpedia.** Call the web **API** (an address that returns data for programs,
   not people):
   `https://www.neuronpedia.org/api/feature/gpt2-small/8-res-jb/{feature_id}`.
   The response is **JSON**, a standard text format for nested data.
3. **Find the label inside the response** with `_extract_explanation_text`. The response
   is deeply nested, so this function searches it **recursively** (it calls itself on
   each nested piece). It looks first in `explanations[].description`, then in fields
   named `explanation`, `label`, `text` or `description`, then inside `jsonDetails`, and
   finally in every other value. It even handles JSON stored inside a text string.
4. **Reject the generic label.** An early version picked up the SAE release's title
   (*"Open Source Sparse Autoencoders for all Residual Stream Layers of GPT2-Small"*)
   instead of the feature's own description. That bad value had been cached, so the code
   now rejects it explicitly and fetches again.
5. **Retry once.** If the request fails (timeout, network error, rate limit), it tries a
   second time. If that also fails, it returns `"unlabelled"`.
6. **Save** the result to the cache file.

The `urllib` used here is part of Python itself, so no extra package is needed.

Running `uv run python -m monitor.labels "your prompt"` prints the top 10 features with
their labels.

---

## Part 4. The tests

Run with `uv run pytest`. There are **7 tests and all pass.** Four warnings appear; they
are the deprecation and SAE messages from section 3.2.

| Test | What it checks | Needs internet / model? |
|---|---|---|
| `test_smoke.py::test_smoke` | `assert True`: proves pytest itself runs | No |
| `test_model.py::test_sae_input_size_matches_model_width` | Model is on CPU; the SAE's input size (768) equals the model's width (768). If they differed, the SAE was built for a different model | Yes, the first time (downloads) |
| `test_extract.py::…returns_top_valid_deterministic_hits` | Password prompt gives exactly 20 hits; `k=5` gives ≤5; IDs are in range; all activations > 0; sorted high to low; two runs give identical results (**deterministic**) | Yes, the first time |
| `test_labels.py::…fetches_and_caches_label` | Label is fetched once, then served from the cache | No (mocked) |
| `test_labels.py::…refreshes_generic_release_cache` | A cached generic label gets replaced by the real one | No (mocked) |
| `test_labels.py::…retries_once_then_returns_unlabelled` | Timeout, then rate-limit error (HTTP 429), gives `"unlabelled"` after exactly 2 tries | No (mocked) |
| `test_labels.py::…prints_top_prompt_features_with_labels` | The command-line output contains the IDs and labels | No (mocked) |

**Mocking.** The label tests never touch the real internet. They use pytest's
**`monkeypatch`**, which temporarily swaps a real function for a fake one during a test.
For example, the fake `urlopen` returns a prepared response instead of calling
Neuronpedia. This makes the tests fast, and they give the same result every time. A fake
like this is called a **mock**. `tmp_path` gives each test a fresh temporary folder, so
tests never touch the real cache file.

---

## Part 5. The sanity-check experiment (`eval/01_sanity_check.ipynb`)

A **notebook** is a document mixing code, its output and notes, run one **cell** at a time.

### 5.1 Question
Before building a drift monitor on SAE features, check the most basic thing: *do medical
prompts and coding prompts even produce different features at layer 8?* If not, the whole
idea fails early.

### 5.2 Setup
- 10 medical prompts ("Can ibuprofen upset the stomach?") and 10 coding prompts ("How do I
  reverse a list in Python?"), hand-written, in `eval/data/sanity_prompts.json`.
- Top 20 features per prompt.
- A feature is **common** in a group if it is in the top 20 of at least 5 of that group's
  10 prompts, and **rare** if in at most 1.
- A feature is **enriched** for a group if it is common there and rare in the other.
- **Jaccard overlap** measures how much two sets share: *(items in both) ÷ (items in
  either)*. 0 = nothing shared, 1 = identical.
- The rule was written **before** looking at results: "clear separation" needs at least 3
  enriched features in each group and Jaccard ≤ 0.25; "weak" means some enriched features
  but not clear.

### 5.3 Results (from the notebook's saved output)
- Common features: 7 medical, 10 coding, 3 shared → **Jaccard = 3/14 = 0.214**.
- Enriched: **2 medical, 4 coding**. The rule needs 3 in each group, so the verdict is **weak**.

| Enriched for | Feature | Medical | Coding | Neuronpedia label |
|---|---:|---:|---:|---|
| Medical | 18171 | 5/10 | 0/10 | medical terms related to inflammation or clinical conditions |
| Medical | 23838 | 5/10 | 1/10 | interrogative sentences ending with a question mark |
| Coding | 10590 | 1/10 | 5/10 | questions starting with "Can I" or "How can I" |
| Coding | 18200 | 1/10 | 6/10 | instances of the article "a" before a noun |
| Coding | 21629 | 1/10 | 5/10 | phrases related to instructing or providing guidance |
| Coding | 21981 | 1/10 | 5/10 | questions starting with "How do you"/"How do we" … |

Timing: median **0.0429 s** per prompt, slowest 0.1838 s (the first prompt). The log
records 0.0426 s and 0.1740 s from an earlier run of the same notebook. Small
run-to-run differences in timing are normal.

### 5.4 What this actually tells us (read carefully)
- **Only one of the six enriched features is about the topic** (18171, medical terms).
  The other five describe **how the question is phrased**: question marks, "How can I",
  the word "a", "How do you". Our coding prompts happen to use "How do I / How can I"
  more, and the features picked that up.
- So "weak separation" may even overstate *topic* separation. Layer 8 features seem at
  least as sensitive to **wording** as to **subject**.
- This cuts both ways for the project. It is a warning for the topic-change scenario
  (some "detections" might really be phrasing artefacts). It is also a hint that SAE
  features could catch the **style and framing scenarios** that embeddings, which are
  designed to ignore wording, may miss. Untested; that is what the real experiment is for.
- **Limits:** 20 prompts, one layer, hand-written prompts, labels that may be wrong. This
  is a quick check, not evidence of anything general.
- **Suggested next check** (already in the log): repeat at layers 4 and 11, and with
  prompts whose phrasing is matched across topics, so that phrasing and topic are not
  mixed together.

---

## Part 6. The design (`docs/design.md`) and why each choice was made

The design was rewritten today after a 6-question interview. Summary of the decisions:

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

## Part 7. Issues found during the review

Ranked by how much they matter. None breaks the current tests.

1. **Network failures are cached forever as "unlabelled"** (`monitor/labels.py:130`).
   If Neuronpedia is briefly down, `"unlabelled"` is saved to the cache and never retried.
   A brief outage becomes a permanent missing label. Fix: don't cache failures.
2. **The retry fires on every error, not only on timeouts and rate limits**
   (`labels.py:127`). The log describes "one retry on timeout/429-style failures", but the
   code retries on *any* error, including a 404 for a feature that doesn't exist. Also,
   `OSError` already covers the other error types listed, so the list repeats itself.
   Harmless, but the code doesn't do what the log says.
3. **The label search is very permissive.** `_extract_explanation_text` falls back to *any*
   text field anywhere in the response. That is exactly how the generic release title got
   picked up. Blocking that one string fixes the known case, but another unrelated field
   could be picked up the same way. A stricter parser that reads only the known
   explanation field would be safer.
4. **Topic and phrasing are mixed together in the sanity check** (Part 5.4). This isn't a
   bug, but it affects how we design the scenario prompts.
5. **Model tests need the internet the first time.** `test_model.py` and `test_extract.py`
   download GPT-2 and the SAE on the first run. They will be slow or fail in CI unless the
   download is cached there.
6. **Small code-quality items:** `_top_prompt_features` in `labels.py` only calls
   `extract_features` and adds nothing (CLAUDE.md asks for no dead code).
   `test_extract.py` asserts both `len(hits) == 20` and `len(hits) <= 20`; the second is
   redundant.
7. **Deprecated TransformerLens loader** (section 3.2). Migrate before TransformerLens 4.0.
8. **Commit author is "Feature Monitor <feature-monitor@local>"**, not your name. This
   repository has its own local git identity. That's fine if intended. If you want your
   GitHub profile credited, set `git config user.name` and `user.email` in this repository.
9. **`docs/setup-report.pdf` is out of date.** It says "No ML libraries were installed",
   which was true on 8 October. Delete it or mark it historical.
10. **Milestone numbering isn't written down anywhere.** CLAUDE.md mentions Milestone 7;
    nothing lists M0 to M8.

---

## Part 8. What was done in this session

1. **Audited milestone 1** (the repository skeleton). All six items and both "done when"
   checks hold.
2. **Pinned Python 3.12** (`.python-version`) and **ran `ruff format`**: 8 files changed,
   layout only.
3. **Committed** in two focused commits, keeping Copilot's work separate from the cleanup:
   - `a9082b0` Add Neuronpedia label parsing fix and sanity-check notebook
   - `92766a2` Pin Python 3.12 and apply ruff format
4. **Reviewed the design doc** against its task. The structure was complete, but the
   content was thin. Ran the 6-question interview and **rewrote `docs/design.md`**.
5. **Checked the SAE loading warning.** It's harmless: FVU is identical both ways.
6. **Wrote this report.**

---

## Part 9. What comes next (suggested order)

1. Write the milestone list (M0 to M8) into `docs/design.md` or a `docs/milestones.md`.
2. Fix issues 1 to 3 in `labels.py` (small, contained).
3. Optional quick experiment: the sanity check at layers 4 and 11, with phrasing matched
   across topics.
4. Start the **storage layer**: SQLite table for requests and their top 20 features.
5. Then the **traffic generator**, starting with the topic-change scenario.
6. Then the **drift test** itself, with false alarm calibration on unchanged traffic.

---

## Appendix: command cheat sheet

| Command | What it does |
|---|---|
| `uv sync` | Install exactly what `uv.lock` lists |
| `uv add <pkg>` / `uv add --dev <pkg>` | Add a dependency (never `pip install`) |
| `make test` = `uv run pytest` | Run all tests |
| `uv run pytest tests/monitor/test_labels.py -v` | Run one test file, listing each test |
| `make lint` = `uv run ruff check .` | Look for code problems |
| `make format` = `uv run ruff format .` | Fix layout |
| `uv run ruff format --check .` | Check layout without changing anything |
| `uv run python -m monitor.model` | Print model and SAE sizes |
| `uv run python -m monitor.labels "a prompt"` | Top 10 features with labels |
| `git log --oneline` | List commits |
| `git status` | Show uncommitted changes |
