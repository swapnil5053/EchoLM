from echolm.eval.generate import as_train_row
from echolm.eval.generate import perplexity


def test_as_train_row():
    row = {"id": "x", "chat_id": "c", "prompt": [{"role": "user", "content": "hi"}], "reference": "yo"}
    assert as_train_row(row)["completion"] == [{"role": "assistant", "content": "yo"}]


def test_perplexity():
    assert perplexity(None) is None
    assert perplexity(0.0) == 1.0
