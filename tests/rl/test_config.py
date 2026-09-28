from pathlib import Path

import pytest

from echolm.rl.config import GrpoConfig
from echolm.rl.config import load_grpo_config
from echolm.rl.config import reward_weights


def test_grpo_yaml_matches_defaults():
    assert load_grpo_config(Path(__file__).parents[2] / "configs" / "grpo.yaml") == GrpoConfig()


@pytest.mark.parametrize("text,match", [
    ("weights: {chrff: 1}\n", "unknown reward weights"),
    ("backend: vllm\n", "backend"),
    ("scale_rewards: batch\n", "scale_rewards"),
    ("num_generations: 1\n", "at least 2"),
])
def test_validation(tmp_path, text, match):
    p = tmp_path / "g.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_grpo_config(p)


def test_partial_weights_keep_defaults():
    w = reward_weights(GrpoConfig(weights={"copy": 3.0}))
    assert w.copy == 3.0 and w.chrf == 1.0
