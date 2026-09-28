import json
import sys

import pytest

pytest.importorskip("peft")

from echolm.rl.config import GrpoConfig  # noqa: E402
from echolm.rl.policy import lora_settings  # noqa: E402
from echolm.rl.train import resolve_init  # noqa: E402
from echolm.rl.train import train_grpo  # noqa: E402
from tests.tiny import DATA  # noqa: E402
from tests.tiny import fake_unsloth  # noqa: E402
from tests.tiny import write_base_and_adapter  # noqa: E402


def small_cfg(base, backend):
    return GrpoConfig(base_model=str(base), backend=backend, num_generations=2, prompts_per_step=2,
                      max_new_tokens=6, eval_steps=2, report_to="none", warmup_steps=1)


def check_run(out_dir, steps):
    info = json.loads((out_dir / "run_info.json").read_text(encoding="utf-8"))
    assert info["steps"] == steps
    assert [h["step"] for h in info["val_history"]] == [0, 2, 3][: len(info["val_history"])]
    assert (out_dir / "adapter" / "adapter_model.safetensors").exists()
    assert (out_dir / f"checkpoint-{info['best_step']}").exists()
    assert "train step 1" in (out_dir / "train.log").read_text(encoding="utf-8")


def test_grpo_runs_end_to_end_with_the_hf_backend(tmp_path, caplog):
    caplog.set_level("INFO")
    base, adapter = write_base_and_adapter(tmp_path)
    out = train_grpo(small_cfg(base, "hf"), DATA, tmp_path / "grpo", max_steps=3, init=adapter)
    check_run(out, 3)


def test_grpo_runs_with_the_unsloth_path_and_loads_sft_weights(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    base, adapter = write_base_and_adapter(tmp_path)
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth(base))
    out = train_grpo(small_cfg(base, "unsloth"), DATA, tmp_path / "grpo", max_steps=2, init=adapter)
    check_run(out, 2)


def test_lora_settings_reads_the_adapter(tmp_path):
    _, adapter = write_base_and_adapter(tmp_path)
    s = lora_settings(adapter)
    assert s["r"] == 4 and "q_proj" in s["target_modules"]


def test_resolve_init_prefers_override_then_config(tmp_path):
    cfg = GrpoConfig(init_adapter=str(tmp_path / "cfg"))
    assert resolve_init(cfg, tmp_path / "cli", tmp_path) == tmp_path / "cli"
    assert resolve_init(cfg, None, tmp_path) == tmp_path / "cfg"
