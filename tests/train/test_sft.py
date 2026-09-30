from pathlib import Path

from echolm.train.config import SftConfig
from echolm.train.sft import run_dir
from echolm.train.sft import train_args
from echolm.train.sft import warmup_steps


def test_train_args_precision_and_best_checkpoint():
    args = train_args(SftConfig(), Path("out/run"), 877, "bf16")
    assert args["bf16"] and not args["fp16"]
    assert args["save_steps"] == args["eval_steps"] == 20
    assert args["load_best_model_at_end"] and args["metric_for_best_model"] == "eval_loss"
    assert args["eval_on_start"] and args["prediction_loss_only"]
    assert args["run_name"] == "run"


def test_train_args_fp32_disables_mixed_precision():
    args = train_args(SftConfig(), Path("out"), 877, "fp32")
    assert not args["bf16"] and not args["fp16"]


def test_smoke_run_evaluates_within_its_steps():
    args = train_args(SftConfig(), Path("out"), 877, "bf16", max_steps=5)
    assert args["max_steps"] == 5 and args["eval_steps"] == 5 and args["save_steps"] == 5


def test_warmup_steps():
    # 877 windows / 16 per step = 55 steps per epoch, 3 epochs = 165 steps, 5% = 8
    assert warmup_steps(877, SftConfig(), -1) == 8
    assert warmup_steps(877, SftConfig(), 5) == 1


def test_run_dir_names():
    assert run_dir(SftConfig(), Path("o"), None, 5).name.startswith("sft-smoke-")
    assert run_dir(SftConfig(), Path("o"), None, -1).name.startswith("sft-r16-")
    assert run_dir(SftConfig(), Path("o"), Path("o/old"), -1) == Path("o/old")


def cpu_args(real):
    def wrapped(*a, **k):
        return {**real(*a, **k), "optim": "adamw_torch", "use_cpu": True, "fp16": False}

    return wrapped


def test_sft_runs_end_to_end_on_cpu(tmp_path, monkeypatch, caplog):
    import json
    import sys

    import pytest

    pytest.importorskip("peft")
    pytest.importorskip("datasets")
    import torch

    from echolm.train import sft
    from tests.tiny import DATA
    from tests.tiny import fake_unsloth
    from tests.tiny import write_base_and_adapter

    caplog.set_level("INFO")
    base, _ = write_base_and_adapter(tmp_path)
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth(base))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    monkeypatch.setattr(sft, "train_args", cpu_args(sft.train_args))
    cfg = SftConfig(report_to="none", n_samples=2, sample_max_new_tokens=4, eval_steps=2, log_steps=1)
    out = sft.train_sft(cfg, DATA, tmp_path / "sft", max_steps=4)
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert [step for step, _ in info["eval_history"]] == [0, 2, 4]
    assert (out / "adapter" / "adapter_model.safetensors").exists()
    assert "step 2 | them:" in (out / "train.log").read_text(encoding="utf-8")
