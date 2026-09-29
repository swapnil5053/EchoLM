import json
import xml.etree.ElementTree as ET

import pytest
from click.testing import CliRunner

from echolm.cli import cli
from echolm.eval.plots import grpo_chart
from echolm.eval.plots import line_svg
from echolm.eval.plots import sft_chart
from echolm.eval.plots import ticks

SFT_HISTORY = [[0, 6.33], [20, 4.72], [40, 4.47], [60, 4.29], [80, 4.23], [100, 4.12], [120, 4.14]]
GRPO_HISTORY = [{"step": s, "reward": r} for s, r in [(0, 0.749), (25, 0.745), (50, 0.72), (75, 0.777)]]


def sft_run(root):
    run = root / "sft" / "sft-r16-x"
    for step, _ in SFT_HISTORY[1:]:
        (run / f"checkpoint-{step}").mkdir(parents=True)
    (run / "run_info.json").write_text(json.dumps({"eval_history": SFT_HISTORY}), encoding="utf-8")
    return run


def grpo_run(root):
    run = root / "grpo" / "grpo-run-x"
    run.mkdir(parents=True)
    (run / "run_info.json").write_text(json.dumps({"val_history": GRPO_HISTORY}), encoding="utf-8")
    return run


def test_ticks_are_round_numbers():
    assert ticks(0, 200) == [0, 50, 100, 150, 200]
    assert ticks(0.653, 0.799) == [0.675, 0.7, 0.725, 0.75, 0.775]
    assert ticks(5, 5) == [5.0]


def test_line_svg_is_valid_and_themed():
    svg = line_svg([(0, 1.0), (10, 2.0)], "t", "x", "y", mark=(10, "best"), ref=(1.5, "ref"))
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "prefers-color-scheme: dark" in svg
    assert svg.count("<circle") == 3
    assert "best" in svg and "stroke-dasharray" in svg


def test_sft_chart_marks_the_selected_checkpoint(tmp_path):
    assert "selected: step 80" in sft_chart(sft_run(tmp_path))


def test_grpo_chart_marks_best_and_step_zero(tmp_path):
    svg = grpo_chart(grpo_run(tmp_path))
    assert "best: step 75" in svg and "SFT (step 0)" in svg


def test_short_history_is_a_clear_error(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "run_info.json").write_text(json.dumps({"val_history": GRPO_HISTORY[:1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than 2"):
        grpo_chart(run)


def test_plot_command(tmp_path):
    sft_run(tmp_path)
    grpo_run(tmp_path)
    out = tmp_path / "figs"
    res = CliRunner().invoke(cli, ["eval", "plot", "--outputs", str(tmp_path), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert sorted(p.name for p in out.iterdir()) == ["grpo_val_reward.svg", "sft_val_loss.svg"]
