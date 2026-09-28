import importlib.util
import logging
import shutil
import sys
from pathlib import Path

from echolm.data.io import read_jsonl
from echolm.train.runtime import wandb_logged_in

log = logging.getLogger(__name__)

MIN_FREE_DISK_GB = 5
MIN_FREE_VRAM_GB = 5


def check_python() -> tuple[bool, str]:
    ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
    return ok, f"python {sys.version.split()[0]} (needs 3.10-3.12)"


def check_packages() -> tuple[bool, str]:
    missing = [p for p in ("torch", "unsloth", "transformers", "datasets", "bitsandbytes")
               if importlib.util.find_spec(p) is None]
    if missing:
        return False, f"missing packages: {', '.join(missing)}; run pip install -e \".[train]\""
    return True, "torch, unsloth, transformers, datasets, bitsandbytes installed"


def check_gpu() -> tuple[bool, str]:
    import torch

    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__} has no CUDA; reinstall torch from the cu128 index (README)"
    free, total = (n / 2**30 for n in torch.cuda.mem_get_info())
    name = torch.cuda.get_device_name(0)
    bf16 = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    msg = f"{name}, {free:.1f} of {total:.1f} GB free, {bf16}, torch {torch.__version__}"
    if free < MIN_FREE_VRAM_GB:
        return False, f"{msg}; close browsers / Teams / games to free GPU memory"
    return True, msg


def check_data(data_dir: Path) -> tuple[bool, str]:
    counts = {}
    for name in ("train_sft.jsonl", "val_sft.jsonl"):
        path = data_dir / name
        if not path.exists():
            return False, f"{path} missing; run `echolm format` first"
        counts[name] = len(read_jsonl(path))
    return True, f"{counts['train_sft.jsonl']} train / {counts['val_sft.jsonl']} val windows in {data_dir}"


def check_disk(path: Path) -> tuple[bool, str]:
    free = shutil.disk_usage(path.resolve().anchor).free / 2**30
    drive = path.resolve().anchor
    return free >= MIN_FREE_DISK_GB, f"{free:.0f} GB free on {drive} (needs {MIN_FREE_DISK_GB})"


def check_wandb(report_to: str) -> tuple[bool, str]:
    if report_to != "wandb":
        return True, "W&B off (report_to: none)"
    if wandb_logged_in():
        return True, "W&B logged in"
    return True, "W&B not logged in, will log offline"


def run_checks(data_dir: Path, report_to: str) -> bool:
    results = [check_python(), check_packages(), check_data(data_dir), check_disk(Path.cwd()),
               check_wandb(report_to)]
    if results[1][0]:
        results.append(check_gpu())
    for ok, msg in results:
        log.info("%s %s", "PASS" if ok else "FAIL", msg)
    return all(ok for ok, _ in results)
