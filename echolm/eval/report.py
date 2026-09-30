import json
import logging
import re
import statistics
from pathlib import Path

from echolm.eval.human import summary

log = logging.getLogger(__name__)

# (key, header, description); † marks metrics that are also GRPO rewards on the training data
COLUMNS = [
    ("detect_auc", "detect AUC ↓", "cross-validated classifier telling your real replies from the model's; "
     "0.5 = cannot tell them apart. Not optimized by any reward"),
    ("chrf", "chrF ↑ †", "character n-gram F-score against what you actually replied to the same message "
     "(95% bootstrap interval over prompts)"),
    ("style_gap", "style gap ↓ †", "mean standardized difference from your real replies on 9 style features"),
    ("reply_ppl", "reply ppl ↓", "perplexity of your real test replies under the model"),
    ("distinct", "distinct ↑", "share of unique replies; low means it falls back on stock replies"),
    ("ngram_copy", "6-gram copy ↓ †", "share of 6+ word replies sharing a 6-word run with a training reply"),
    ("chatbot_rate", "chatbot ↓ †", "share of replies with assistant phrases (\"I'm sorry, but\" ...)"),
    ("median_words", "median words", "reply length"),
]
STAGES = ("base", "sft", "grpo")
START, END = "<!-- results:start -->", "<!-- results:end -->"
SEEDED = re.compile(r"^(.+)-s\d+$")


def fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def cell(metrics: dict, key: str) -> str:
    if f"{key}_seed_sd" in metrics:
        return f"{metrics[key]:.3f} ±{metrics[f'{key}_seed_sd']:.3f}"
    text = fmt(metrics.get(key))
    if key == "median_words" and metrics.get(key) is not None:
        text = f"{round(metrics[key], 1):g}"
    if key == "reply_ppl" and metrics.get(key) is not None:
        text = f"{metrics[key]:.1f}"
    if key == "chrf" and metrics.get("chrf_ci"):
        lo, hi = metrics["chrf_ci"]
        text += f" ({lo:.3f}–{hi:.3f})"
    if key == "detect_auc" and metrics.get("detect_auc_sd"):
        text += f" ±{metrics['detect_auc_sd']:.3f}"
    return text


def load_runs(root: Path) -> dict[str, dict]:
    paths = sorted(root.glob("*/metrics.json"))
    runs = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in paths}
    if not runs:
        raise ValueError(f"no metrics.json under {root}; run `echolm eval run` first")
    return runs


def order(names: list[str]) -> list[str]:
    def key(name: str) -> tuple:
        stage = next((i for i, s in enumerate(STAGES) if name.startswith(s)), len(STAGES))
        return stage, name

    return sorted(names, key=key)


def seed_means(runs: dict[str, dict]) -> dict[str, dict]:
    """One extra row per model trained with several seeds (names like grpo-s1, grpo-s2): mean ± sd."""
    groups = {}
    for name in runs:
        m = SEEDED.match(name)
        if m:
            groups.setdefault(m.group(1), []).append(runs[name])
    out = {}
    for prefix, rows in groups.items():
        if len(rows) < 2:
            continue
        row = {"reference": rows[0]["reference"]}
        for key, _, _ in COLUMNS:
            values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if len(values) == len(rows):
                row[key], row[f"{key}_seed_sd"] = statistics.fmean(values), statistics.stdev(values)
        out[f"{prefix} (mean of {len(rows)} seeds)"] = row
    return out


def table(runs: dict[str, dict]) -> str:
    runs = {**runs, **seed_means(runs)}
    names = order(list(runs))
    reference = runs[names[0]]["reference"]
    lines = ["| model | " + " | ".join(c[1] for c in COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 1)]
    lines += [f"| {n} | " + " | ".join(cell(runs[n], c[0]) for c in COLUMNS) + " |" for n in names]
    lines.append("| your real replies | " + " | ".join(cell(reference, c[0]) for c in COLUMNS) + " |")
    return "\n".join(lines)


def build_report(root: Path, notes: bool = True) -> str:
    """The results table; `notes` adds one line per metric (the README explains them in its own words)."""
    runs = load_runs(root)
    first = runs[order(list(runs))[0]]
    seeds = first["per_seed"]
    n = seeds[next(iter(seeds))]["n"]
    legend = [f"- **{label}**: {desc}" for _, label, desc in COLUMNS] + [""] if notes else []
    return "\n".join([
        f"{n} held-out test replies (later in time than all training data), {len(seeds)} sampled "
        "replies each at temperature 0.8. † = also a GRPO reward on the training split.", "",
        table(runs), "", *human_lines(root), *legend,
    ])


def human_lines(root: Path) -> list[str]:
    scores = summary(root)
    if not scores:
        return []
    parts = [f"{m}: real reply spotted in {s['correct']} of {s['rounds']} rounds ({s['accuracy']:.0%})"
             for m, s in sorted(scores.items(), key=lambda kv: order(list(scores)).index(kv[0]))]
    return ["Human judge, blind A/B in the demo's \"Real or model?\" tab (50% = cannot tell): "
            + "; ".join(parts) + ".", ""]


def update_readme(readme: Path, body: str) -> None:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"{readme} has no {START} / {END} markers")
    pattern = re.compile(re.escape(START) + ".*?" + re.escape(END), re.DOTALL)
    readme.write_text(pattern.sub(lambda _: f"{START}\n{body}\n{END}", text), encoding="utf-8")


def write_report(root: Path, readme: Path | None = None) -> Path:
    path = root / "report.md"
    path.write_text("# EchoLM evaluation\n\n" + build_report(root), encoding="utf-8")
    (root / "report.json").write_text(json.dumps(load_runs(root), indent=2), encoding="utf-8")
    if readme is not None:
        update_readme(readme, build_report(root, notes=False))
        log.info("results table updated in %s", readme)
    log.info("report written to %s", path)
    return path
