import json

from click.testing import CliRunner

from echolm.cli import cli
from tests.eval.test_plots import grpo_run
from tests.eval.test_plots import sft_run
from tests.eval.test_report import METRICS


def test_plot_command(tmp_path):
    sft_run(tmp_path)
    grpo_run(tmp_path)
    out = tmp_path / "figs"
    res = CliRunner().invoke(cli, ["eval", "plot", "--outputs", str(tmp_path), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert sorted(p.name for p in out.iterdir()) == ["grpo_val_reward.svg", "sft_val_loss.svg"]


def test_report_command_prints_the_report(tmp_path):
    (tmp_path / "sft").mkdir()
    (tmp_path / "sft" / "metrics.json").write_text(json.dumps(METRICS), encoding="utf-8")
    res = CliRunner().invoke(cli, ["eval", "report", "--out", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "| sft |" in res.output


def test_report_command_without_runs(tmp_path):
    res = CliRunner().invoke(cli, ["eval", "report", "--out", str(tmp_path)])
    assert res.exit_code != 0
    assert "echolm eval run" in res.output
