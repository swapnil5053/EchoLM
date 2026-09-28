from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from echolm.config import load_dataclass

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class SftConfig:
    model: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    max_seq_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(default_factory=lambda: list(LORA_TARGETS))
    epochs: float = 3.0
    batch_size: int = 4
    grad_accum: int = 4
    eval_batch_size: int = 2
    lr: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    eval_steps: int = 20
    log_steps: int = 5
    save_total_limit: int | None = None
    n_samples: int = 5
    sample_max_new_tokens: int = 64
    seed: int = 13
    report_to: str = "wandb"
    wandb_samples: bool = False
    wandb_project: str = "echolm"


def load_sft_config(path: Path | None) -> SftConfig:
    cfg = load_dataclass(path, SftConfig)
    if cfg.report_to not in ("wandb", "none"):
        raise ValueError(f"report_to must be 'wandb' or 'none', got '{cfg.report_to}'")
    return cfg
