import logging
import netrc
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

# SetThreadExecutionState flags: keep the system awake while this thread runs
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake():
    if sys.platform != "win32":
        yield
        return
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    log.info("sleep is blocked while training runs (the lid and power button still work)")
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def wandb_logged_in() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    for name in ("_netrc", ".netrc"):
        path = Path.home() / name
        if not path.exists():
            continue
        try:
            if netrc.netrc(str(path)).authenticators("api.wandb.ai"):
                return True
        except netrc.NetrcParseError as e:
            log.warning("could not read %s: %s", path, e)
    return False


def setup_wandb(project: str) -> None:
    os.environ.setdefault("WANDB_PROJECT", project)
    if os.environ.get("WANDB_MODE"):
        return
    if not wandb_logged_in():
        # an interactive login prompt would hang an unattended run
        os.environ["WANDB_MODE"] = "offline"
        log.warning("not logged in to W&B, logging offline; upload later with "
                    "`wandb sync wandb/offline-run-*`")


def log_to_file(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def git_sha() -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    except FileNotFoundError:
        log.warning("git not found, run_info.json will have no commit sha")
        return None
    if res.returncode != 0:
        log.warning("not inside a git checkout, run_info.json will have no commit sha")
        return None
    return res.stdout.strip()


def repo_url() -> str | None:
    """The origin remote as an https URL, e.g. for linking the code from a model card."""
    try:
        res = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    url = res.stdout.strip().removesuffix(".git")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")
    return url if url.startswith("https://") else None
