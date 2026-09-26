import hashlib
import logging
import statistics
from pathlib import Path

from echolm.data.io import read_jsonl

log = logging.getLogger(__name__)


def render(row: dict, tok) -> dict:
    prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=False)
    if not full.startswith(prompt):
        raise ValueError(f"chat template renders the prompt differently in the full chat (row {row['id']})")
    # trailing newline after <|im_end|> would make TRL append a second end token
    return {"id": row["id"], "prompt": prompt, "completion": full[len(prompt):].rstrip("\n")}


def load_split(path: Path, tok) -> list[dict]:
    return [render(row, tok) for row in read_jsonl(path)]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def token_lengths(rows: list[dict], tok) -> list[int]:
    return [len(tok(r["prompt"] + r["completion"], add_special_tokens=False)["input_ids"]) for r in rows]


def drop_too_long(rows: list[dict], lengths: list[int], max_len: int) -> list[dict]:
    # TRL truncates on the right, which would cut the reply we train on, so drop instead
    kept = [r for r, n in zip(rows, lengths, strict=True) if n <= max_len]
    if len(kept) < len(rows):
        log.warning("dropped %d of %d windows longer than max_seq_len=%d",
                    len(rows) - len(kept), len(rows), max_len)
    return kept


def length_report(lengths: list[int], max_len: int) -> dict:
    ordered = sorted(lengths)
    return {
        "tokens_median": statistics.median(ordered),
        "tokens_p95": ordered[int(0.95 * (len(ordered) - 1))],
        "tokens_max": ordered[-1],
        "over_max_seq_len": sum(n > max_len for n in ordered),
    }
