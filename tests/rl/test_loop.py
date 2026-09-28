from echolm.rl.config import GrpoConfig
from echolm.rl.grpo import Rollout
from echolm.rl.loop import finalize
from echolm.rl.loop import summarize


def test_summarize_counts_groups_without_signal():
    parts = [{"chrf": 0.5, "style": 1, "length": 1, "duplicate": 0, "copy": 0, "chatbot": 0, "empty": 0,
              "total": 1.0}] * 4
    runs = [Rollout([1], [2, 3], "a", 1.0, 0.0), Rollout([1], [2], "b", 1.0, 0.0),
            Rollout([1], [2], "c", 1.0, 0.5), Rollout([1], [2], "d", 1.0, -0.5)]
    out = summarize(parts, runs, GrpoConfig(num_generations=2))
    assert out["zero_std_groups"] == 0.5
    assert out["completion_tokens"] == 1.25 and out["reward"] == 1.0


def test_finalize_copies_the_best_checkpoint(tmp_path):
    (tmp_path / "checkpoint-25").mkdir()
    (tmp_path / "checkpoint-25" / "adapter_model.safetensors").write_text("w")
    final = finalize(tmp_path, 25)
    assert (final / "adapter_model.safetensors").read_text() == "w"
