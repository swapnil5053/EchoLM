from echolm.eval.human import FILE
from echolm.eval.human import summary


def test_summary_per_model(tmp_path):
    (tmp_path / FILE).write_text('{"id": "a", "model": "sft", "correct": true}\n'
                                 '{"id": "b", "model": "sft", "correct": false}\n'
                                 '{"id": "a", "model": "grpo", "correct": true}\n', encoding="utf-8")
    assert summary(tmp_path) == {"sft": {"rounds": 2, "correct": 1, "accuracy": 0.5},
                                 "grpo": {"rounds": 1, "correct": 1, "accuracy": 1.0}}


def test_no_guesses(tmp_path):
    assert summary(tmp_path) == {}
