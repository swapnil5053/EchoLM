import pytest

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


def test_validation_uses_fixed_draws_and_leaves_training_rng_alone(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from echolm.rl.data import encode
    from echolm.rl.data import reward_context
    from echolm.rl.data import val_rows
    from echolm.rl.loop import validate
    from echolm.rl.policy import load_policy
    from tests.tiny import DATA
    from tests.tiny import write_base_and_adapter

    base, adapter = write_base_and_adapter(tmp_path)
    cfg = GrpoConfig(base_model=str(base), backend="hf", max_new_tokens=6, val_samples=2, report_to="none")
    policy = load_policy(cfg, adapter)
    val = encode(val_rows(DATA), policy.tok, cfg.max_prompt_tokens)
    ctx = reward_context(DATA)
    torch.manual_seed(0)
    before = torch.get_rng_state()
    first = validate(policy, val, cfg, ctx)
    assert torch.equal(torch.get_rng_state(), before)
    assert validate(policy, val, cfg, ctx) == first
