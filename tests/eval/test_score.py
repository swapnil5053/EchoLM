import pytest

pytest.importorskip("sklearn")

from echolm.eval.authorship import train_classifier  # noqa: E402
from echolm.eval.score import score_generations  # noqa: E402
from echolm.eval.score import score_reference  # noqa: E402
from tests.eval.test_authorship import windows  # noqa: E402


def test_score_generations_averages_seeds():
    ws = windows("train") + windows("test")
    clf = train_classifier(ws, 0)
    gens = [{"id": "a", "seed": s, "reference": "haan bhai", "output": o}
            for s, o in ((1, "haan bhai"), (2, "I'm sorry, but I can't help with that."))]
    m = score_generations(gens, ws, clf)
    assert set(m["per_seed"]) == {1, 2}
    assert m["chatbot_rate"] == 0.5
    assert 0 <= m["p_me"] <= 1


def test_reference_has_zero_style_gap():
    ws = windows("train") + windows("test")
    ref = score_reference(ws, train_classifier(ws, 0))
    assert ref["style_gap"] == 0 and ref["chatbot_rate"] == 0
