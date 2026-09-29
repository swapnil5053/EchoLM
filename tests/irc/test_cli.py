import json
from pathlib import Path

from click.testing import CliRunner

from echolm.cli import cli

IRC_CONFIG = Path(__file__).parent.parent.parent / "configs" / "irc.yaml"


def test_users_export_parse_format(logs, tmp_path):
    runner = CliRunner()
    res = runner.invoke(cli, ["irc", "users", "--logs", str(logs)])
    assert res.exit_code == 0, res.output
    assert res.output.splitlines()[0].startswith("alice")
    export = tmp_path / "exports" / "ubuntu_irc.json"
    res = runner.invoke(cli, ["irc", "export", "--logs", str(logs), "--out", str(export)])
    assert res.exit_code == 0, res.output
    res = runner.invoke(cli, ["parse", str(export), "--me", "Alex", "--out", str(tmp_path / "parsed")])
    assert res.exit_code == 0, res.output
    res = runner.invoke(cli, ["format", "--parsed", str(tmp_path / "parsed"), "--out", str(tmp_path / "out"),
                              "--config", str(IRC_CONFIG)])
    assert res.exit_code == 0, res.output
    stats = json.loads((tmp_path / "out" / "stats.json").read_text(encoding="utf-8"))
    assert stats["splits"]["train"] > 0


def test_export_unknown_user(logs, tmp_path):
    res = CliRunner().invoke(cli, ["irc", "export", "--logs", str(logs), "--user", "ghost",
                                   "--out", str(tmp_path / "x.json")])
    assert res.exit_code != 0
    assert "irc users" in res.output


def test_validate_rejects_unknown_rules(tmp_path):
    res = CliRunner().invoke(cli, ["irc", "validate", "--data", str(tmp_path), "--rules", "addressed,vibes"])
    assert res.exit_code != 0
    assert "vibes" in res.output
