import json

from click.testing import CliRunner

from echolm.cli import cli


def test_full_pipeline_on_synthetic(tmp_path):
    runner = CliRunner()
    raw, parsed, out = tmp_path / "raw", tmp_path / "parsed", tmp_path / "out"
    assert runner.invoke(cli, ["synth", "--out", str(raw)]).exit_code == 0
    for name in ("whatsapp_rohan.txt", "telegram_meera.json"):
        res = runner.invoke(cli, ["parse", str(raw / name), "--me", "Kabir", "--out", str(parsed)])
        assert res.exit_code == 0, res.output
    res = runner.invoke(cli, ["format", "--parsed", str(parsed), "--out", str(out)])
    assert res.exit_code == 0, res.output
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["splits"]["train"] > stats["splits"]["test"] > 0
    row = json.loads((out / "train_sft.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["prompt"][0]["content"].startswith("You are Kabir.")


def test_format_without_parsed_files(tmp_path):
    res = CliRunner().invoke(cli, ["format", "--parsed", str(tmp_path)])
    assert res.exit_code != 0
    assert "run `echolm parse` first" in res.output


def test_wrong_me_is_a_clean_error(tmp_path):
    runner = CliRunner()
    runner.invoke(cli, ["synth", "--out", str(tmp_path)])
    res = runner.invoke(cli, ["parse", str(tmp_path / "whatsapp_rohan.txt"), "--me", "kabir"])
    assert res.exit_code == 1
    assert "senders are ['Kabir', 'Rohan']" in res.output
    assert "Traceback" not in res.output
