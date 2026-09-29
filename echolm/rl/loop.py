import json
import logging
import math
import shutil
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from echolm.eval.overlap import repetition
from echolm.rl.config import GrpoConfig
from echolm.rl.config import reward_weights
from echolm.rl.data import Example
from echolm.rl.data import RewardContext
from echolm.rl.grpo import Rollout
from echolm.rl.grpo import advantages
from echolm.rl.grpo import sample_group
from echolm.rl.grpo import update
from echolm.rl.policy import Policy
from echolm.rl.rewards import score_group

log = logging.getLogger(__name__)

PARTS = ("chrf", "style", "length", "duplicate", "copy", "chatbot", "empty")


def decode(policy: Policy, ids: list[int]) -> str:
    return policy.tok.decode(ids, skip_special_tokens=True).strip()


def rollout(policy: Policy, ex: Example, cfg: GrpoConfig, ctx: RewardContext) -> tuple[list, list]:
    groups = sample_group(policy, ex.prompt_ids, cfg.num_generations, cfg.max_new_tokens,
                          cfg.temperature, cfg.top_p)
    texts = [decode(policy, ids) for ids in groups]
    parts = score_group(texts, ex.reference, ex.id, ctx.index, ctx.scales, reward_weights(cfg))
    adv = advantages([p["total"] for p in parts], cfg.scale_rewards)
    runs = [Rollout(ex.prompt_ids, ids, t, p["total"], a)
            for ids, t, p, a in zip(groups, texts, parts, adv, strict=True)]
    return runs, parts


def summarize(parts: list[dict], runs: list[Rollout], cfg: GrpoConfig) -> dict:
    out = {k: statistics.fmean(p[k] for p in parts) for k in PARTS}
    out["reward"] = statistics.fmean(p["total"] for p in parts)
    out["reward_std"] = statistics.pstdev(p["total"] for p in parts)
    groups = [runs[i:i + cfg.num_generations] for i in range(0, len(runs), cfg.num_generations)]
    out["zero_std_groups"] = sum(all(r.advantage == 0 for r in g) for g in groups) / len(groups)
    out["completion_tokens"] = statistics.fmean(len(r.completion_ids) for r in runs)
    return out


def train_step(policy: Policy, batch: list[Example], cfg: GrpoConfig, ctx: RewardContext,
               optimizer, scheduler) -> dict:
    start = time.perf_counter()
    policy.to_inference(policy.model)
    runs, parts = [], []
    for ex in batch:
        r, p = rollout(policy, ex, cfg, ctx)
        runs += r
        parts += p
    stats = update(policy, runs, cfg, optimizer)
    scheduler.step()
    out = {**summarize(parts, runs, cfg), **stats, "lr": scheduler.get_last_lr()[0]}
    out["sec"] = time.perf_counter() - start
    return out


def validate(policy: Policy, examples: list[Example], cfg: GrpoConfig, ctx: RewardContext) -> dict:
    """Val reward on fixed random draws: every checkpoint samples with the same seed, so the
    comparison between checkpoints is not swamped by sampling noise. Training RNG is left untouched."""
    import torch

    policy.to_inference(policy.model)
    texts, parts = [], []
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.seed)
        for ex in examples:
            outs = sample_group(policy, ex.prompt_ids, cfg.val_samples, cfg.max_new_tokens,
                                cfg.val_temperature, 1.0)
            for text in (decode(policy, ids) for ids in outs):
                texts.append(text)
                parts += score_group([text], ex.reference, ex.id, ctx.index, ctx.scales, reward_weights(cfg))
    out = {k: statistics.fmean(p[k] for p in parts) for k in PARTS if k != "duplicate"}
    out["reward"] = statistics.fmean(p["total"] for p in parts)
    out["distinct"] = repetition(texts)["distinct"]
    return out


def make_optimizer(policy: Policy, cfg: GrpoConfig, total_steps: int):
    import importlib.util

    import torch

    params = [p for p in policy.model.parameters() if p.requires_grad]
    if params[0].is_cuda and importlib.util.find_spec("bitsandbytes"):
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def schedule(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


def save(policy: Policy, path: Path) -> None:
    policy.model.save_pretrained(str(path))
    policy.tok.save_pretrained(str(path))


def finalize(out_dir: Path, best_step: int) -> Path:
    final = out_dir / "adapter"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(out_dir / f"checkpoint-{best_step}", final)
    return final


def write_info(out_dir: Path, cfg: GrpoConfig, extra: dict) -> None:
    info = {"config": asdict(cfg), **extra}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
