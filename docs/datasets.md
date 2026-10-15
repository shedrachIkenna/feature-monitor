# Traffic datasets

`traffic/prepare_data.py` builds five prompt pools from three Hugging Face datasets and
writes them to `traffic/data/<pool>.jsonl` (one `{"text": ...}` per line). The files are
not committed: run `uv run python -m traffic.prepare_data` to recreate them. With seed 0
the output is byte-identical between runs.

## Sources and licences

| Dataset | Used for | Licence | Attribution |
|---|---|---|---|
| [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | normal, shift_new_topic | CDLA-Sharing-1.0 (changed versions we share must keep this licence) | Bitext Innovations |
| [lavita/MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD), a conversion of [abachaa/MedQuAD](https://github.com/abachaa/MedQuAD) | shift_medical | CC BY 4.0 (stated upstream; the HF copy has no licence tag) | Ben Abacha & Demner-Fushman, "A question-entailment approach to question answering", BMC Bioinformatics 2019 |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | shift_injection | Tagged Apache-2.0; the card metadata also says CC BY 4.0. We give attribution to satisfy both | deepset |

Rejected: `curaihealth/medical_questions_pairs` (more natural questions, licence unknown) and
`lavita/ChatDoctor-HealthCareMagic-100k` (scraped, no licence, long messages).

## Processing

Every pool is processed in the same order:

1. Keep the user message only.
2. Strip whitespace.
3. Drop empty items and items over 300 characters.
4. Remove duplicates, ignoring case and spacing.
5. Shuffle with seed 0.

| Pool | Source rows | Final | Notes |
|---|---|---|---|
| normal_train | 20,886 (Bitext minus ACCOUNT) | 14,017 | 70% of the 20,025 left after removing duplicates; builds the baseline |
| normal_test | (same) | 6,008 | 30%; used in experiments only; no overlap with train (checked in code) |
| shift_new_topic | 5,986 (Bitext ACCOUNT) | 5,454 | |
| shift_medical | 47,441 | 43,448 | " ?" rewritten as "?" |
| shift_injection | 263 with label 1 | 125 | English only (see below) |

**Bitext template slots.** 6,670 of the 26,872 Bitext messages contain unfilled slots such as
`{{Order Number}}` (9 slot types). Only the support pools have them. Left in, they would let
the monitor tell the pools apart by the braces alone, which would make our results look too
good. Each slot is filled with a value drawn by a seeded random generator from a short list
(`SLOT_VALUES`).

**Why hold out a Bitext category for shift_new_topic.** A *confound* is a second difference
that could explain a result instead of the one being tested. MedQuAD differs from Bitext in
topic, but also in style, length, punctuation, typos and how it was generated. A held-out
Bitext category (ACCOUNT: account changes, passwords, registration) differs only in topic,
so an alert on it can be put down to the topic. The medical and injection pools test
"something changed", not "the topic changed".

**English filter for injections.** Of the 223 deepset injections under 300 characters,
98 (44%) are not fully English: most are German or mixed German/English, and a few are in
Spanish, French, Russian, Croatian or Khmer.
GPT-2 small is an English model, so those would test a language change. `is_english` drops
a message if it has a non-ASCII letter or any listed German, Spanish, French or Croatian
function word. This rule is rough:
- It drops at least three English injections because each holds a German name or quoted
  phrase ("Der Spiegel", "Höcke", `'SPD ist doof'`).
- It keeps a few English injections that start with German topic keywords, e.g.
  "Datenschutz Europa".

## Known limits

- Bitext and MedQuAD are synthetic or templated, not real user traffic.
- MedQuAD questions follow a few fixed templates ("What are the treatments for X?").
- shift_injection has only 125 prompts, which limits experiment window sizes.
- Some deepset label-1 items do not look like injections ("generate c++", "Be a DJ. Give me
  some music recommendation"). We kept the labels as published.
