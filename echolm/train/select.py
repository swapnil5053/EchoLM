import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# a later checkpoint has to beat the best val loss by more than this share to be worth its extra
# memorization: on the first real run, step 80 was within 2.6% of the best loss (step 100) but
# copied training text 10x less often
DEFAULT_TOL = 0.03


def pick_step(history: list[list], saved: set[int], tol: float) -> int:
    trained = [(step, loss) for step, loss in history if step > 0 and step in saved]
    if not trained:
        raise ValueError("no saved checkpoint has an eval loss; was the run trained with evals?")
    best = min(loss for _, loss in trained)
    return min(step for step, loss in trained if loss <= best * (1 + tol))


def saved_steps(run_dir: Path) -> set[int]:
    return {int(p.name.split("-")[1]) for p in run_dir.glob("checkpoint-*") if p.is_dir()}


def newest_run(root: Path, pattern: str = "sft-r*") -> Path:
    runs = [p for p in root.glob(pattern) if (p / "run_info.json").exists()]
    if not runs:
        raise ValueError(f"no finished run matching {pattern} in {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def select_checkpoint(run_dir: Path, tol: float = DEFAULT_TOL) -> Path:
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    history = info.get("eval_history")
    if not history:
        log.warning("%s has no eval history, using its final adapter", run_dir.name)
        return run_dir / "adapter"
    step = pick_step(history, saved_steps(run_dir), tol)
    log.info("selected checkpoint-%d of %s (tolerance %.0f%%)", step, run_dir.name, tol * 100)
    return run_dir / f"checkpoint-{step}"
