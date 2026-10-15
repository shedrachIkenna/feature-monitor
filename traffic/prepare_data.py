"""Download prompt datasets from Hugging Face and save them as traffic pools.

Pools written to traffic/data/<pool>.jsonl, one {"text": ...} object per line:
  normal_train     Bitext support questions, minus the held-out category; builds the baseline
  normal_test      the rest of the same questions; used in experiments, never in the baseline
  shift_new_topic  Bitext questions from the held-out category only
  shift_medical    MedQuAD health questions
  shift_injection  deepset prompt injections (label 1), English only (rough check)

Run with: uv run python -m traffic.prepare_data
Sources, licences and the reasons for each choice are in docs/datasets.md.
"""

import json
import random
import re
from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset

SEED = 0
MAX_CHARS = 300
TEST_FRACTION = 0.3
HELD_OUT_CATEGORY = "ACCOUNT"
OUT_DIR = Path(__file__).parent / "data"

BITEXT = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
MEDQUAD = "lavita/MedQuAD"
INJECTIONS = "deepset/prompt-injections"

# Bitext leaves template slots such as {{Order Number}} unfilled. Left in, they would be a
# token pattern unique to the support pools, so we fill them with plausible values.
SLOT_VALUES: dict[str, list[str]] = {
    "Order Number": ["#48213", "#10957", "#77342", "#30518", "#92604"],
    "Account Type": ["Premium", "Basic", "Business", "Pro", "Family"],
    "Person Name": ["Alex Morgan", "Maria Lopez", "Sam Chen", "Priya Patel", "John Okafor"],
    "Account Category": ["Gold", "Standard", "Platinum", "Student", "Free"],
    "Refund Amount": ["49.99", "120", "15.50", "300", "75"],
    "Currency Symbol": ["$", "£", "€"],
    "Delivery City": ["Chicago", "Manchester", "Lagos", "Toronto", "Sydney"],
    "Delivery Country": ["Canada", "Germany", "Nigeria", "Australia", "Spain"],
    "Invoice Number": ["INV-20431", "INV-88702", "INV-51296"],
}
SLOT_PATTERN = re.compile(r"\{\{([^}]+)\}\}")

# Function words from the other languages in the injection set (German, Spanish, French,
# Croatian). Words that are also English, such as "was", "die" or "yo", are left out.
FOREIGN_WORDS = re.compile(
    r"\b(und|nicht|ich|der|das|ist|sind|du|dich|dir|mir|mich|ein|eine|zu|mit|auf|wie|wer|"
    r"bitte|vergiss|ignoriere|alle|alles|jetzt|deine|sie|wir|auch|oder|aber|schreibe|"
    r"que|todo|todos|todas|olvida|olvide|dices|dije|digo|tienes|"
    r"oubliez|toutes|dites|les|zaboravi|sve|budi|ako)\b",
    re.IGNORECASE,
)


def fill_slots(text: str, rng: random.Random) -> str:
    """Replace each {{Slot}} with a value drawn from SLOT_VALUES; unknown slots raise KeyError."""
    return SLOT_PATTERN.sub(lambda m: rng.choice(SLOT_VALUES[m.group(1)]), text)


def is_english(text: str) -> bool:
    """Rough check: no non-ASCII letters and none of FOREIGN_WORDS. Mixed-language text fails."""
    has_foreign_letter = any(c.isalpha() and not c.isascii() for c in text)
    return not has_foreign_letter and not FOREIGN_WORDS.search(text)


def clean(texts: Iterable[str], seed: int = SEED) -> list[str]:
    """Strip, drop empty or over-long texts, deduplicate (case and spacing ignored), shuffle."""
    seen: set[str] = set()
    kept: list[str] = []
    for text in texts:
        text = text.strip()
        key = " ".join(text.casefold().split())
        if not text or len(text) > MAX_CHARS or key in seen:
            continue
        seen.add(key)
        kept.append(text)
    random.Random(seed).shuffle(kept)
    return kept


def split(texts: list[str], test_fraction: float = TEST_FRACTION) -> tuple[list[str], list[str]]:
    """Split an already shuffled list into (train, test)."""
    n_test = round(len(texts) * test_fraction)
    return texts[n_test:], texts[:n_test]


def write_jsonl(path: Path, texts: list[str]) -> None:
    """Write one {"text": ...} object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def build_pools() -> dict[str, list[str]]:
    """Download the three datasets and return the five cleaned pools."""
    rng = random.Random(SEED)
    bitext = load_dataset(BITEXT, split="train")
    normal, new_topic = [], []
    for row in bitext:
        text = fill_slots(row["instruction"], rng)
        (new_topic if row["category"] == HELD_OUT_CATEGORY else normal).append(text)
    normal_train, normal_test = split(clean(normal))

    # MedQuAD writes "Is X inherited ?"; drop the space before "?" so it reads like a user.
    medquad = load_dataset(MEDQUAD, split="train")
    medical = [re.sub(r"\s+\?", "?", q) for q in medquad["question"]]

    injections = load_dataset(INJECTIONS)
    attacks = [
        row["text"] for name in ("train", "test") for row in injections[name] if row["label"] == 1
    ]
    dropped = [t for t in attacks if len(t.strip()) <= MAX_CHARS and not is_english(t)]
    print(f"Dropped {len(dropped)} non-English injections, e.g.:")
    for text in dropped[:5]:
        print(f"  {text[:100]!r}")

    return {
        "normal_train": normal_train,
        "normal_test": normal_test,
        "shift_new_topic": clean(new_topic),
        "shift_medical": clean(medical),
        "shift_injection": clean(t for t in attacks if is_english(t)),
    }


def main() -> None:
    """Build every pool, save it, and print its size and three examples."""
    pools = build_pools()
    overlap = set(pools["normal_train"]) & set(pools["normal_test"])
    assert not overlap, f"{len(overlap)} prompts are in both normal splits"
    for name, texts in pools.items():
        write_jsonl(OUT_DIR / f"{name}.jsonl", texts)
        print(f"\n{name}: {len(texts)} prompts -> {OUT_DIR / name}.jsonl")
        for text in texts[:3]:
            print(f"  - {text!r}")


if __name__ == "__main__":
    main()
