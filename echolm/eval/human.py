import logging
from pathlib import Path

from echolm.data.io import read_jsonl

log = logging.getLogger(__name__)

# written by the "Real or model?" tab of the demo, one line per answered round
FILE = "guesses.jsonl"


def summary(eval_dir: Path) -> dict[str, dict]:
    """Share of rounds where a person spotted the real reply, per model (0.5 = could not tell)."""
    path = eval_dir / FILE
    if not path.exists():
        return {}
    out = {}
    for row in read_jsonl(path):
        s = out.setdefault(row["model"], {"rounds": 0, "correct": 0})
        s["rounds"] += 1
        s["correct"] += row["correct"]
    return {m: {**s, "accuracy": round(s["correct"] / s["rounds"], 3)} for m, s in out.items()}
