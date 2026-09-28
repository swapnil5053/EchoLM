import hashlib
import logging
import statistics
from pathlib import Path

from echolm.data.io import read_jsonl

log = logging.getLogger(__name__)

IGNORE = -100


def render(row: dict, tok) -> dict:
    prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=False)
    if not full.startswith(prompt):
        raise ValueError(f"chat template renders the prompt differently in the full chat (row {row['id']})")
    # the reply keeps its <|im_end|> so the model learns to stop; the newline after it is template padding
    return {"id": row["id"], "prompt": prompt, "completion": full[len(prompt):].rstrip("\n")}


def tokenize(row: dict, tok) -> dict:
    p = tok(row["prompt"], add_special_tokens=False)["input_ids"]
    c = tok(row["completion"], add_special_tokens=False)["input_ids"]
    return {**row, "input_ids": p + c, "attention_mask": [1] * (len(p) + len(c)),
            "labels": [IGNORE] * len(p) + c}


def load_split(path: Path, tok) -> list[dict]:
    return [tokenize(render(row, tok), tok) for row in read_jsonl(path)]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def drop_too_long(rows: list[dict], max_len: int) -> list[dict]:
    # truncating would cut off the reply we train on, so long windows are dropped instead
    kept = [r for r in rows if len(r["input_ids"]) <= max_len]
    if len(kept) < len(rows):
        log.warning("dropped %d of %d windows longer than max_seq_len=%d",
                    len(rows) - len(kept), len(rows), max_len)
    return kept


def length_report(rows: list[dict], max_len: int) -> dict:
    lengths = sorted(len(r["input_ids"]) for r in rows)
    return {
        "tokens_median": statistics.median(lengths),
        "tokens_p95": lengths[int(0.95 * (len(lengths) - 1))],
        "tokens_max": lengths[-1],
        "over_max_seq_len": sum(n > max_len for n in lengths),
    }


def model_columns(rows: list[dict]) -> list[dict]:
    return [{k: r[k] for k in ("input_ids", "attention_mask", "labels")} for r in rows]


def pad_batch(batch: list[dict], pad_id: int, multiple: int = 8) -> dict:
    longest = max(len(b["input_ids"]) for b in batch)
    n = -(-longest // multiple) * multiple
    return {
        "input_ids": [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch],
        "attention_mask": [b["attention_mask"] + [0] * (n - len(b["attention_mask"])) for b in batch],
        "labels": [b["labels"] + [IGNORE] * (n - len(b["labels"])) for b in batch],
    }
