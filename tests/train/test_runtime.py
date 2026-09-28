import logging
import os

from echolm.train import runtime


def test_keep_awake_is_a_noop_off_windows():
    with runtime.keep_awake():
        pass


def test_wandb_offline_when_not_logged_in(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(runtime, "wandb_logged_in", lambda: False)
    runtime.setup_wandb("echolm")
    assert os.environ["WANDB_MODE"] == "offline"
    assert os.environ["WANDB_PROJECT"] == "echolm"


def test_wandb_mode_left_alone_when_logged_in(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(runtime, "wandb_logged_in", lambda: True)
    runtime.setup_wandb("echolm")
    assert "WANDB_MODE" not in os.environ


def test_wandb_logged_in_reads_netrc(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(runtime.Path, "home", lambda: tmp_path)
    assert not runtime.wandb_logged_in()
    (tmp_path / ".netrc").write_text("machine api.wandb.ai\n  login user\n  password abc\n")
    (tmp_path / ".netrc").chmod(0o600)
    assert runtime.wandb_logged_in()


def test_log_to_file_writes_utf8(tmp_path):
    handler = runtime.log_to_file(tmp_path / "run" / "train.log")
    logging.getLogger("echolm.test").warning("bhai 😂")
    logging.getLogger().removeHandler(handler)
    handler.close()
    assert "bhai 😂" in (tmp_path / "run" / "train.log").read_text(encoding="utf-8")


def test_git_sha_is_short_or_none():
    sha = runtime.git_sha()
    assert sha is None or 4 <= len(sha) <= 12
