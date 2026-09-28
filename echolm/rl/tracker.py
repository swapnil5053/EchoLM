import logging
from pathlib import Path

from echolm.train.runtime import setup_wandb

log = logging.getLogger(__name__)

# the handful of numbers worth reading in the console; everything goes to W&B
CONSOLE = ("reward", "chrf", "duplicate", "copy", "loss", "grad_norm", "zero_std_groups", "sec")


class Tracker:
    def __init__(self, report_to: str, project: str, name: str, config: dict, out_dir: Path):
        self.run = None
        if report_to == "wandb":
            setup_wandb(project)
            import wandb

            self.run = wandb.init(project=project, name=name, config=config, dir=str(out_dir))

    def log(self, metrics: dict, step: int, prefix: str) -> None:
        shown = " ".join(f"{k}={metrics[k]:.3f}" for k in CONSOLE if k in metrics)
        log.info("%s step %d %s", prefix, step, shown or metrics)
        if self.run is not None:
            self.run.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
