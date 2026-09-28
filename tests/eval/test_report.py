import json

import pytest

from echolm.eval.report import build_report
from echolm.eval.report import order

METRICS = {"p_me": 0.5, "style_gap": 1.2, "reply_ppl": 40.0, "chatbot_rate": 0.1, "exact_copy": 0.0,
           "ngram_copy": 0.0, "distinct": 0.9, "median_words": 4, "classifier_accuracy": 0.8,
           "per_seed": {"1": {"n": 58}}, "reference": {"p_me": 0.7, "style_gap": 0.0}}


def test_order_puts_base_first():
    assert order(["sft", "base", "grpo"]) == ["base", "grpo", "sft"]


def test_build_report(tmp_path):
    for name in ("base", "sft"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(json.dumps(METRICS), encoding="utf-8")
    text = build_report(tmp_path)
    assert "| base | 0.500 |" in text and "| sft |" in text and "| real replies | 0.700" in text
    assert "58 held-out test replies, 1 sampled" in text


def test_build_report_without_runs(tmp_path):
    with pytest.raises(ValueError, match="echolm eval run"):
        build_report(tmp_path)
