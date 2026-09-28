from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from echolm.config import load_dataclass


@dataclass
class EvalConfig:
    base_model: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    max_seq_len: int = 1024
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.9
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    classifier_seed: int = 13


def load_eval_config(path: Path | None) -> EvalConfig:
    cfg = load_dataclass(path, EvalConfig)
    if not cfg.seeds:
        raise ValueError("seeds must list at least one seed")
    return cfg
