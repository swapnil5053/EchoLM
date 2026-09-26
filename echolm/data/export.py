import json
import logging
import statistics
from collections import Counter
from pathlib import Path

from echolm.data.io import write_jsonl
from echolm.data.models import Window

log = logging.getLogger(__name__)

OPENER = "[new conversation]"
ROLES = {"them": "user", "me": "assistant"}


def to_prompt(w: Window, system: str) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": ROLES[c["role"]], "content": c["text"]} for c in w.context]
    if len(msgs) == 1:
        msgs.append({"role": "user", "content": f"> {w.quoted}" if w.quoted else OPENER})
    elif w.quoted:
        last = msgs[-1]
        last["content"] = f"> {w.quoted}\n{last['content']}"
    return msgs


def sft_row(w: Window, system: str) -> dict:
    return {
        "id": w.id,
        "prompt": to_prompt(w, system),
        "completion": [{"role": "assistant", "content": w.target}],
    }


def ref_row(w: Window, system: str) -> dict:
    return {"id": w.id, "chat_id": w.chat_id, "prompt": to_prompt(w, system), "reference": w.target}


def stats(wins: list[Window]) -> dict:
    lengths = [len(w.target) for w in wins]
    return {
        "windows": len(wins),
        "splits": dict(Counter(w.split for w in wins)),
        "per_chat": dict(Counter(w.chat_id for w in wins)),
        "openers": sum(not w.context for w in wins),
        "with_quote": sum(w.quoted is not None for w in wins),
        "target_chars_median": statistics.median(lengths),
        "context_turns_mean": round(statistics.mean(len(w.context) for w in wins), 2),
    }


def export_all(wins: list[Window], out_dir: Path, system: str) -> dict:
    by_split = {s: [w for w in wins if w.split == s] for s in ("train", "val", "test")}
    if not by_split["train"]:
        raise ValueError("train split is empty; export more history or lower val_frac/test_frac")
    write_jsonl(out_dir / "windows.jsonl", [w.to_dict() for w in wins])
    write_jsonl(out_dir / "train_sft.jsonl", [sft_row(w, system) for w in by_split["train"]])
    write_jsonl(out_dir / "val_sft.jsonl", [sft_row(w, system) for w in by_split["val"]])
    write_jsonl(out_dir / "grpo.jsonl", [ref_row(w, system) for w in by_split["train"]])
    write_jsonl(out_dir / "test.jsonl", [ref_row(w, system) for w in by_split["test"]])
    info = stats(wins)
    (out_dir / "stats.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
