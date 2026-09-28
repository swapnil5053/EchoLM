import json

import pytest

pytest.importorskip("sklearn")

from echolm.eval.config import EvalConfig  # noqa: E402
from echolm.eval.run import score  # noqa: E402
from tests.eval.test_authorship import windows  # noqa: E402


def test_score_writes_metrics(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    run.mkdir()
    ws = windows("train") + windows("test")
    (data / "windows.jsonl").write_text("\n".join(json.dumps(w) for w in ws), encoding="utf-8")
    gens = [{"id": str(i), "seed": 1, "reference": "haan", "output": "haan bhai"} for i in range(3)]
    (run / "generations.jsonl").write_text("\n".join(json.dumps(g) for g in gens), encoding="utf-8")
    (run / "nll.json").write_text(json.dumps({"reply_nll": 3.0, "reply_ppl": 20.09}), encoding="utf-8")
    m = score(data, run, EvalConfig())
    assert json.loads((run / "metrics.json").read_text(encoding="utf-8"))["reply_ppl"] == 20.09
    assert m["distinct"] == pytest.approx(1 / 3, abs=0.001)
