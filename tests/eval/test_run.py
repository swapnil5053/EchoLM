import json

import pytest

pytest.importorskip("sklearn")

from echolm.eval.config import EvalConfig  # noqa: E402
from echolm.eval.run import score  # noqa: E402
from tests.eval.fixtures import MINE  # noqa: E402
from tests.eval.fixtures import windows  # noqa: E402


def test_score_writes_metrics(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    run.mkdir()
    ws = windows("train") + windows("test")
    (data / "windows.jsonl").write_text("\n".join(json.dumps(w) for w in ws), encoding="utf-8")
    gens = [{"id": str(i), "seed": 1, "reference": m, "output": "haan bhai"} for i, m in enumerate(MINE)]
    (run / "generations.jsonl").write_text("\n".join(json.dumps(g) for g in gens), encoding="utf-8")
    (run / "nll.json").write_text(json.dumps({"reply_nll": 3.0, "reply_ppl": 20.09}), encoding="utf-8")
    m = score(data, run, EvalConfig())
    saved = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert saved["reply_ppl"] == 20.09 and "reference" in saved
    assert m["distinct"] == pytest.approx(1 / len(MINE), abs=0.001)


def test_generate_then_score_for_base_and_adapter(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    import sys

    from echolm.eval.report import build_report
    from echolm.eval.run import generate
    from tests.tiny import DATA
    from tests.tiny import fake_unsloth
    from tests.tiny import write_base_and_adapter

    base, adapter = write_base_and_adapter(tmp_path)
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth(base))
    cfg = EvalConfig(base_model=str(base), max_new_tokens=6, seeds=[1, 2])
    for name, ref in (("base", "base"), ("sft", str(adapter))):
        generate(ref, DATA, tmp_path / "eval" / name, cfg)
        score(DATA, tmp_path / "eval" / name, cfg)
    gens_file = tmp_path / "eval" / "sft" / "generations.jsonl"
    rows = [json.loads(line) for line in gens_file.open(encoding="utf-8")]
    assert len(rows) == 2 * sum(1 for _ in (DATA / "test.jsonl").open(encoding="utf-8"))
    nll = json.loads((tmp_path / "eval" / "base" / "nll.json").read_text(encoding="utf-8"))
    assert nll["reply_ppl"] > 1
    assert "| sft |" in build_report(tmp_path / "eval")
