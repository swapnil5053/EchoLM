from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    session_gap_min: int = 180
    max_context_turns: int = 8
    max_context_chars: int = 1500
    max_target_chars: int = 600
    min_target_chars: int = 1
    include_openers: bool = True
    val_frac: float = 0.05
    test_frac: float = 0.10
    max_chat_share: float | None = None
    seed: int = 13
    system_prompt: str = "You are {name}. Reply to the chat exactly the way {name} texts."
    blocked_words: list[str] = field(default_factory=list)


def load_config(path: Path | None) -> DataConfig:
    if path is None:
        return DataConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(DataConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    cfg = DataConfig(**raw)
    if not 0 <= cfg.val_frac + cfg.test_frac < 1:
        raise ValueError(f"val_frac + test_frac must be in [0, 1), got {cfg.val_frac + cfg.test_frac}")
    return cfg
