import pytest

pytest.importorskip("sklearn")

from echolm.eval.score import chrf_interval  # noqa: E402
from echolm.eval.score import score_generations  # noqa: E402
from echolm.eval.score import score_reference  # noqa: E402
from tests.eval.fixtures import MINE  # noqa: E402
from tests.eval.fixtures import windows  # noqa: E402


def gens():
    rows = []
    for i, ref in enumerate(MINE):
        rows.append({"id": str(i), "seed": 1, "reference": ref, "output": ref})
        rows.append({"id": str(i), "seed": 2, "reference": ref, "output": "I'm sorry, but I can't help."})
    return rows


def test_score_generations():
    m = score_generations(gens(), windows("train") + windows("test"), 0)
    assert set(m["per_seed"]) == {1, 2}
    assert m["per_seed"][1]["chrf"] == 1.0 and m["per_seed"][2]["chatbot_rate"] == 1.0
    assert m["chatbot_rate"] == 0.5
    assert m["detect_auc"] is not None
    lo, hi = m["chrf_ci"]
    assert lo <= m["chrf"] <= hi


def test_chrf_interval_is_tight_for_identical_scores():
    rows = [{"id": str(i), "output": "haan", "reference": "haan"} for i in range(20)]
    assert chrf_interval(rows, 0) == [1.0, 1.0]


def test_reference_row():
    ref = score_reference(windows("train") + windows("test"), 0)
    assert ref["style_gap"] == 0 and ref["chatbot_rate"] == 0
    assert "detect_auc" in ref
