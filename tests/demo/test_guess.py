import json

import pytest

from echolm.demo.guess import GuessGame
from echolm.demo.guess import load_pairs
from echolm.eval.human import summary

SYSTEM = {"role": "system", "content": "You are Alex."}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def dirs(tmp_path):
    data, ev = tmp_path / "data", tmp_path / "eval"
    write(data / "test.jsonl", [
        {"id": i, "prompt": [SYSTEM, {"role": "user", "content": q}], "reference": r}
        for i, q, r in (("a", "wifi broken", "which card?"), ("b", "thanks", "np"), ("c", "hi", "hey"))
    ])
    write(ev / "sft" / "generations.jsonl", [
        {"id": "a", "seed": 1, "output": "try lspci"}, {"id": "a", "seed": 2, "output": "reboot"},
        {"id": "b", "seed": 1, "output": "np"}, {"id": "c", "seed": 1, "output": "  "},
    ])
    return data, ev


def test_pairs_use_the_first_sample_and_skip_identical_or_empty(dirs):
    pairs = load_pairs(*dirs, "sft")
    assert pairs == [{"id": "a", "context": [{"role": "user", "content": "wifi broken"}],
                      "real": "which card?", "fake": "try lspci"}]
    assert load_pairs(*dirs, "grpo") == []


def test_rounds_place_the_real_reply_at_random_and_log_answers(dirs):
    game = GuessGame(*dirs, seed=0)
    assert game.models == ["sft"]
    seen = set()
    for _ in range(20):
        rnd = game.new_round("sft")
        assert sorted(rnd.options) == ["try lspci", "which card?"]
        assert rnd.options[rnd.real] == "which card?"
        seen.add(rnd.real)
        game.answer(rnd, rnd.real)
    assert seen == {0, 1}
    game.answer(game.new_round("sft"), 1 - rnd.real)
    assert summary(dirs[1]) == {"sft": {"rounds": 21, "correct": 20, "accuracy": 0.952}}


def test_unknown_model_is_a_clear_error(dirs):
    with pytest.raises(ValueError, match="echolm eval run"):
        GuessGame(*dirs).new_round("grpo")


def test_totals_add_up_saved_rounds(dirs):
    game = GuessGame(*dirs, seed=1)
    assert game.totals("sft") == [0, 0]
    rnd = game.new_round("sft")
    game.answer(rnd, rnd.real)
    assert GuessGame(*dirs).totals("sft") == [1, 1]
