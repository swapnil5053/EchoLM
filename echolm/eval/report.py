import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

COLUMNS = [
    ("p_me", "P(me) ↑", "authorship classifier: average probability that you wrote the reply"),
    ("style_gap", "style gap ↓", "mean standardized difference from your real replies on 9 style features"),
    ("reply_ppl", "reply ppl ↓", "perplexity of your real test replies under the model"),
    ("chatbot_rate", "chatbot ↓", "share of replies with assistant phrases (\"I'm sorry, but\" ...)"),
    ("exact_copy", "exact copy ↓", "share of 4+ word replies identical to a training reply"),
    ("ngram_copy", "6-gram copy ↓", "share of 6+ word replies sharing a 6-word run with a training reply"),
    ("distinct", "distinct ↑", "share of replies that are unique"),
    ("median_words", "median words", "reply length"),
]


def fmt(value) -> str:
    if value is None:
        return "–"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def load_runs(root: Path) -> dict[str, dict]:
    runs = {}
    for path in sorted(root.glob("*/metrics.json")):
        runs[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    if not runs:
        raise ValueError(f"no metrics.json under {root}; run `echolm eval run` first")
    return runs


def order(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: (n != "base", n))


def build_report(root: Path) -> str:
    runs = load_runs(root)
    names = order(list(runs))
    reference = runs[names[0]]["reference"]
    head = "| model | " + " | ".join(c[1] for c in COLUMNS) + " |"
    lines = [head, "|" + "---|" * (len(COLUMNS) + 1)]
    for name in names:
        lines.append(f"| {name} | " + " | ".join(fmt(runs[name].get(c[0])) for c in COLUMNS) + " |")
    lines.append("| real replies | " + " | ".join(fmt(reference.get(c[0])) for c in COLUMNS) + " |")
    n = runs[names[0]]["per_seed"]
    first = n[next(iter(n))]["n"]
    acc = runs[names[0]].get("classifier_accuracy")
    notes = [f"- {label}: {desc}" for _, label, desc in COLUMNS]
    return "\n".join([
        "# EchoLM evaluation", "",
        f"{first} held-out test replies, {len(n)} sampled replies each. "
        f"Authorship classifier balanced accuracy on held-out real messages: {fmt(acc)}.", "",
        *lines, "", *notes, "",
    ])


def write_report(root: Path) -> Path:
    path = root / "report.md"
    path.write_text(build_report(root), encoding="utf-8")
    log.info("report written to %s", path)
    return path
