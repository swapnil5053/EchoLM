import logging
import random
from dataclasses import dataclass
from pathlib import Path

from echolm.data.io import read_jsonl
from echolm.eval.generate import as_train_row
from echolm.rl.rewards import CopyIndex
from echolm.rl.rewards import feature_scales
from echolm.train.data import render

log = logging.getLogger(__name__)


@dataclass
class Example:
    id: str
    prompt_ids: list[int]
    reference: str


@dataclass
class RewardContext:
    index: CopyIndex
    scales: dict[str, float]


def val_rows(data_dir: Path) -> list[dict]:
    rows = read_jsonl(data_dir / "val_sft.jsonl")
    return [{"id": r["id"], "prompt": r["prompt"], "reference": r["completion"][0]["content"]} for r in rows]


def encode(rows: list[dict], tok, max_prompt_tokens: int) -> list[Example]:
    out = []
    for row in rows:
        prompt = render(as_train_row(row), tok)["prompt"]
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_prompt_tokens:
            out.append(Example(row["id"], ids, row["reference"]))
    if len(out) < len(rows):
        log.info("skipped %d of %d prompts longer than %d tokens", len(rows) - len(out), len(rows),
                 max_prompt_tokens)
    return out


def reward_context(data_dir: Path) -> RewardContext:
    train = [w for w in read_jsonl(data_dir / "windows.jsonl") if w["split"] == "train"]
    replies = {w["id"]: w["target"] for w in train}
    return RewardContext(CopyIndex(replies), feature_scales(list(replies.values())))


def batches(examples: list[Example], size: int, steps: int, seed: int):
    rng = random.Random(seed)
    order = []
    for _ in range(steps):
        if len(order) < size:
            fresh = list(examples)
            rng.shuffle(fresh)
            order.extend(fresh)
        yield order[:size]
        order = order[size:]
