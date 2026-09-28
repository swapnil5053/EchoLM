import json

import pytest

from echolm.train.select import newest_run
from echolm.train.select import pick_step
from echolm.train.select import select_checkpoint

# eval history of the first real run
HISTORY = [[0, 6.3299], [20, 4.7203], [40, 4.4732], [60, 4.2856], [80, 4.2335], [100, 4.1243],
           [120, 4.144], [140, 4.16], [153, 4.1561]]
SAVED = {20, 40, 60, 80, 100, 120, 140}


def test_real_run_picks_step_80():
    assert pick_step(HISTORY, SAVED, 0.03) == 80


def test_zero_tolerance_is_plain_best():
    assert pick_step(HISTORY, SAVED, 0.0) == 100


def test_unsaved_steps_are_skipped():
    assert pick_step(HISTORY, {100, 120}, 0.03) == 100


def test_baseline_alone_is_an_error():
    with pytest.raises(ValueError, match="no saved checkpoint"):
        pick_step([[0, 6.0]], set(), 0.03)


def make_run(path, history):
    path.mkdir(parents=True)
    (path / "run_info.json").write_text(json.dumps({"eval_history": history}), encoding="utf-8")
    for step in SAVED:
        (path / f"checkpoint-{step}").mkdir()
    return path


def test_select_checkpoint_and_fallback(tmp_path):
    run = make_run(tmp_path / "sft-r16-a", HISTORY)
    assert select_checkpoint(run).name == "checkpoint-80"
    old = make_run(tmp_path / "sft-r16-b", None)
    assert select_checkpoint(old).name == "adapter"


def test_newest_run_ignores_unfinished(tmp_path):
    make_run(tmp_path / "sft-r16-a", HISTORY)
    (tmp_path / "sft-r16-z").mkdir()
    assert newest_run(tmp_path).name == "sft-r16-a"
    with pytest.raises(ValueError, match="no finished run"):
        newest_run(tmp_path / "missing")
