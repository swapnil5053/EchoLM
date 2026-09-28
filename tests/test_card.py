import json

from echolm.card import build_card
from echolm.card import write_card
from tests.eval.test_report import runs


def grpo_run(tmp_path):
    run = tmp_path / "grpo-run-x"
    (run / "adapter").mkdir(parents=True)
    info = {"config": {"prompts_per_step": 4, "num_generations": 4, "lr": 1e-5, "weights": {"chrf": 1.0}},
            "init_adapter": "outputs/sft/sft-r16-a/checkpoint-80", "steps": 200, "best_step": 150,
            "best_val": {"reward": 1.23}, "peak_vram_gb": 6.1, "runtime_s": 3600,
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU"}
    (run / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    return run


def test_card_has_front_matter_training_eval_and_limits(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "stats.json").write_text(json.dumps({"windows": 963, "splits": {"train": 813}}), encoding="utf-8")
    (tmp_path / "eval").mkdir()
    runs(tmp_path / "eval")
    text = build_card(grpo_run(tmp_path), tmp_path / "eval", data)
    assert text.startswith("---\nbase_model: Qwen/Qwen2.5-1.5B-Instruct")
    assert "`checkpoint-80`" in text and "Selected GRPO step: 150" in text
    assert "RTX 4060 Laptop GPU, peak 6.1 GB, 60 min" in text
    assert "## Evaluation" in text and "| grpo |" in text
    assert "do not upload them" in text


def test_card_without_eval_or_stats(tmp_path):
    path = write_card(grpo_run(tmp_path), tmp_path / "missing", tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path.name == "README.md" and "## Evaluation" not in text
