from pathlib import Path

import pytest

from echolm.eval.config import EvalConfig
from echolm.eval.config import load_eval_config


def test_eval_yaml_matches_defaults():
    assert load_eval_config(Path(__file__).parents[2] / "configs" / "eval.yaml") == EvalConfig()


def test_seeds_required(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("seeds: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seeds"):
        load_eval_config(p)
