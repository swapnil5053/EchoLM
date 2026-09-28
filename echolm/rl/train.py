import logging
import random
import time
from pathlib import Path

from echolm.data.io import read_jsonl
from echolm.rl.config import GrpoConfig
from echolm.rl.data import batches
from echolm.rl.data import encode
from echolm.rl.data import reward_context
from echolm.rl.data import val_rows
from echolm.rl.loop import finalize
from echolm.rl.loop import make_optimizer
from echolm.rl.loop import save
from echolm.rl.loop import train_step
from echolm.rl.loop import validate
from echolm.rl.loop import write_info
from echolm.rl.policy import load_policy
from echolm.rl.tracker import Tracker
from echolm.train.runtime import git_sha
from echolm.train.runtime import keep_awake
from echolm.train.runtime import log_to_file
from echolm.train.select import newest_run
from echolm.train.select import select_checkpoint

log = logging.getLogger(__name__)


def resolve_init(cfg: GrpoConfig, override: Path | None, sft_root: Path) -> Path:
    if override:
        return override
    if cfg.init_adapter != "auto":
        return Path(cfg.init_adapter)
    return select_checkpoint(newest_run(sft_root))


def train_grpo(cfg: GrpoConfig, data_dir: Path, out_root: Path, max_steps: int = -1,
               init: Path | None = None, sft_root: Path = Path("outputs/sft")) -> Path:
    tag = "smoke" if max_steps > 0 else "run"
    out_dir = out_root / f"grpo-{tag}-{time.strftime('%Y%m%d-%H%M')}"
    handler = log_to_file(out_dir / "train.log")
    try:
        with keep_awake():
            fit(cfg, data_dir, out_dir, max_steps if max_steps > 0 else cfg.max_steps,
                resolve_init(cfg, init, sft_root))
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return out_dir


class Evaluator:
    def __init__(self, policy, val, cfg: GrpoConfig, ctx, out_dir: Path, tracker: Tracker):
        self.policy, self.val, self.cfg, self.ctx = policy, val, cfg, ctx
        self.out_dir, self.tracker = out_dir, tracker
        self.history = []

    def __call__(self, step: int) -> None:
        metrics = validate(self.policy, self.val, self.cfg, self.ctx)
        self.tracker.log(metrics, step, "val")
        self.history.append({"step": step, **{k: round(v, 4) for k, v in metrics.items()}})
        save(self.policy, self.out_dir / f"checkpoint-{step}")

    def best(self) -> dict:
        return max(self.history, key=lambda h: h["reward"])


def fit(cfg: GrpoConfig, data_dir: Path, out_dir: Path, steps: int, init: Path) -> None:
    import torch

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    policy = load_policy(cfg, init)
    ctx = reward_context(data_dir)
    train = encode(read_jsonl(data_dir / "grpo.jsonl"), policy.tok, cfg.max_prompt_tokens)
    val = encode(val_rows(data_dir), policy.tok, cfg.max_prompt_tokens)
    log.info("%d train prompts, %d val prompts, %d steps from %s", len(train), len(val), steps, init)
    optimizer, scheduler = make_optimizer(policy, cfg, steps)
    tracker = Tracker(cfg.report_to, cfg.wandb_project, out_dir.name, {"init": str(init)}, out_dir)
    evaluate = Evaluator(policy, val, cfg, ctx, out_dir, tracker)
    start, peak = time.perf_counter(), 0.0
    evaluate(0)
    for step, batch in enumerate(batches(train, cfg.prompts_per_step, steps, cfg.seed), start=1):
        tracker.log(train_step(policy, batch, cfg, ctx, optimizer, scheduler), step, "train")
        if torch.cuda.is_available():
            peak = max(peak, torch.cuda.max_memory_reserved() / 2**30)
        if step % cfg.eval_steps == 0 or step == steps:
            evaluate(step)
    best = evaluate.best()
    final = finalize(out_dir, best["step"])
    write_info(out_dir, cfg, {
        "init_adapter": str(init), "git_sha": git_sha(), "steps": steps, "best_step": best["step"],
        "best_val": best, "val_history": evaluate.history,
        "runtime_s": round(time.perf_counter() - start), "peak_vram_gb": round(peak, 2),
    })
    tracker.finish()
    log.info("best val reward %.4f at step %d (step 0 is the SFT starting point); adapter saved to %s",
             best["reward"], best["step"], final)
