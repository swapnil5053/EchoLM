from pathlib import Path

import pytest

from echolm.config import DataConfig
from echolm.config import load_config


def test_default_yaml_matches_dataclass_defaults():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "default.yaml")
    assert cfg == DataConfig()


def test_none_gives_defaults():
    assert load_config(None) == DataConfig()


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("sesion_gap_min: 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sesion_gap_min"):
        load_config(p)


def test_split_fractions_validated(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("val_frac: 0.5\ntest_frac: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="val_frac"):
        load_config(p)


def test_irc_yaml_loads_and_split_by_is_checked(tmp_path):
    assert load_config(Path(__file__).parent.parent / "configs" / "irc.yaml").split_by == "time"
    p = tmp_path / "c.yaml"
    p.write_text("split_by: random\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split_by"):
        load_config(p)
