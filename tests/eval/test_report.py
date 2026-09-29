import json

import pytest

from echolm.eval.report import END
from echolm.eval.report import START
from echolm.eval.report import build_report
from echolm.eval.report import order
from echolm.eval.report import write_report

METRICS = {"detect_auc": 0.7, "detect_auc_sd": 0.02, "chrf": 0.2, "chrf_ci": [0.18, 0.22], "style_gap": 1.2,
           "reply_ppl": 40.0, "chatbot_rate": 0.1, "ngram_copy": 0.0, "distinct": 0.9, "median_words": 4,
           "per_seed": {"1": {"n": 50}}, "reference": {"detect_auc": 0.55, "style_gap": 0.0}}


def runs(tmp_path, names=("base", "sft-step80", "grpo")):
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(json.dumps(METRICS), encoding="utf-8")
    return tmp_path


def test_order_follows_the_pipeline():
    assert order(["grpo", "sft-step80", "base", "other"]) == ["base", "sft-step80", "grpo", "other"]


def test_build_report(tmp_path):
    text = build_report(runs(tmp_path))
    assert "| base | 0.700 ±0.020 | 0.200 (0.180–0.220) |" in text
    assert "| your real replies | 0.550 |" in text
    assert text.index("| sft-step80") < text.index("| grpo")
    assert "50 held-out test replies" in text


def test_write_report_updates_readme_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n{START}\nold\n{END}\nafter\n", encoding="utf-8")
    (tmp_path / "eval").mkdir()
    write_report(runs(tmp_path / "eval"), readme)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and "| grpo |" in text and text.endswith("after\n")
    assert (tmp_path / "eval" / "report.json").exists()


def test_readme_without_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n", encoding="utf-8")
    (tmp_path / "eval").mkdir()
    with pytest.raises(ValueError, match="markers"):
        write_report(runs(tmp_path / "eval"), readme)


def test_build_report_without_runs(tmp_path):
    with pytest.raises(ValueError, match="echolm eval run"):
        build_report(tmp_path)


def test_report_adds_the_human_judge_line(tmp_path):
    runs(tmp_path, names=("base", "sft", "grpo"))
    rows = '{"id": "a", "model": "grpo", "correct": false}\n{"id": "b", "model": "sft", "correct": true}\n'
    (tmp_path / "guesses.jsonl").write_text(rows, encoding="utf-8")
    text = build_report(tmp_path)
    assert "sft: real reply spotted in 1 of 1 rounds (100%); grpo: real reply spotted in 0 of 1" in text
