from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pathlib import Path

from echolm.config import load_dataclass
from echolm.rl.rewards import RewardWeights


@dataclass
class GrpoConfig:
    base_model: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    backend: str = "unsloth"
    init_adapter: str = "auto"
    max_seq_len: int = 1024
    max_prompt_tokens: int = 768
    num_generations: int = 4
    prompts_per_step: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    lr: float = 1e-5
    warmup_steps: int = 10
    weight_decay: float = 0.0
    max_grad_norm: float = 0.2
    max_steps: int = 200
    num_iterations: int = 1
    clip_eps: float = 0.2
    scale_rewards: str = "group"
    eval_steps: int = 25
    val_temperature: float = 0.8
    seed: int = 13
    report_to: str = "wandb"
    wandb_project: str = "echolm"
    weights: dict = field(default_factory=lambda: {f.name: f.default for f in fields(RewardWeights)})


def load_grpo_config(path: Path | None) -> GrpoConfig:
    cfg = load_dataclass(path, GrpoConfig)
    unknown = set(cfg.weights) - {f.name for f in fields(RewardWeights)}
    if unknown:
        raise ValueError(f"unknown reward weights in {path}: {sorted(unknown)}")
    if cfg.backend not in ("unsloth", "hf"):
        raise ValueError(f"backend must be 'unsloth' or 'hf', got '{cfg.backend}'")
    if cfg.scale_rewards not in ("group", "none"):
        raise ValueError(f"scale_rewards must be 'group' or 'none', got '{cfg.scale_rewards}'")
    if cfg.num_generations < 2:
        raise ValueError("num_generations must be at least 2: advantages are relative to the group")
    if cfg.report_to not in ("wandb", "none"):
        raise ValueError(f"report_to must be 'wandb' or 'none', got '{cfg.report_to}'")
    return cfg


def reward_weights(cfg: GrpoConfig) -> RewardWeights:
    return RewardWeights(**{**RewardWeights().__dict__, **cfg.weights})
