import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from echolm.data.io import read_jsonl
from echolm.eval.human import FILE

log = logging.getLogger(__name__)


@dataclass
class Round:
    id: str
    model: str
    context: list[dict]
    options: list[str]
    real: int


def load_pairs(data_dir: Path, eval_dir: Path, model: str) -> list[dict]:
    """Test prompts where the model's first sampled reply differs from the real one."""
    path = eval_dir / model / "generations.jsonl"
    if not path.exists() or not (data_dir / "test.jsonl").exists():
        return []
    rows = {r["id"]: r for r in read_jsonl(data_dir / "test.jsonl")}
    first = {}
    for g in read_jsonl(path):
        first.setdefault(g["id"], g["output"])
    return [{"id": i, "context": [m for m in rows[i]["prompt"] if m["role"] != "system"],
             "real": rows[i]["reference"], "fake": out}
            for i, out in first.items()
            if i in rows and out.strip() and out.strip() != rows[i]["reference"].strip()]


class GuessGame:
    def __init__(self, data_dir: Path, eval_dir: Path, models: tuple[str, ...] = ("sft", "grpo"),
                 seed: int | None = None):
        self.pairs = {m: p for m in models if (p := load_pairs(data_dir, eval_dir, m))}
        self.log_path = eval_dir / FILE
        self.rng = random.Random(seed)

    @property
    def models(self) -> list[str]:
        return list(self.pairs)

    def new_round(self, model: str) -> Round:
        if model not in self.pairs:
            raise ValueError(f"no generations for '{model}'; run `echolm eval run --name {model}` first")
        pair = self.rng.choice(self.pairs[model])
        real = self.rng.randrange(2)
        options = [pair["fake"], pair["fake"]]
        options[real] = pair["real"]
        return Round(pair["id"], model, pair["context"], options, real)

    def answer(self, rnd: Round, picked: int) -> bool:
        correct = picked == rnd.real
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": rnd.id, "model": rnd.model, "correct": correct}) + "\n")
        return correct
