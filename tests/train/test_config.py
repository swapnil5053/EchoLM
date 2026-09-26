from pathlib import Path

import pytest

from echolm.train.config import SftConfig
from echolm.train.config import load_sft_config


def test_sft_yaml_matches_defaults():
    assert load_sft_config(Path(__file__).parents[2] / "configs" / "sft.yaml") == SftConfig()


def test_report_to_is_validated(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("report_to: tensorboard\n", encoding="utf-8")
    with pytest.raises(ValueError, match="report_to"):
        load_sft_config(p)
