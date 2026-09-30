import pytest

pytest.importorskip("gradio")

from echolm.demo.app import build_app  # noqa: E402
from echolm.demo.app import compare  # noqa: E402
from echolm.demo.app import respond  # noqa: E402


class FakeBank:
    names = ["base", "sft", "grpo"]

    def __init__(self):
        self.calls = []

    def reply(self, name, messages, temperature, max_new_tokens=64):
        self.calls.append((name, messages))
        return f"{name} reply\nsecond line"


def test_respond_appends_bubbles_and_clears_the_box():
    bank = FakeBank()
    history, box = respond(bank, "sys")("kya scene", [], "sft", 0.8)
    assert box == ""
    assert history == [{"role": "user", "content": "kya scene"},
                       {"role": "assistant", "content": "sft reply"},
                       {"role": "assistant", "content": "second line"}]
    assert bank.calls[0][1][0] == {"role": "system", "content": "sys"}


def test_respond_ignores_empty_messages():
    assert respond(FakeBank(), "sys")("  ", [], "sft", 0.8) == ([], "")


def test_compare_answers_with_every_model():
    outs = compare(FakeBank(), "sys")("hi", 0.8)
    assert outs == ["base reply\nsecond line", "sft reply\nsecond line", "grpo reply\nsecond line"]


def test_build_app():
    app = build_app(FakeBank(), "sys")
    assert app is not None


class FakeGame:
    models = ["sft"]

    def __init__(self):
        from echolm.demo.guess import Round

        self.rnd = Round("a", "sft", [{"role": "user", "content": "q"}], ["fake", "real"], 1)
        self.log = []

    def new_round(self, model):
        return self.rnd

    def answer(self, rnd, picked):
        self.log.append(picked == rnd.real)
        return picked == rnd.real

    def totals(self, model):
        return [sum(self.log), len(self.log)]


def test_guess_round_and_answer():
    from echolm.demo.app import guess_answer
    from echolm.demo.app import guess_round

    game = FakeGame()
    context, a, b, rnd, score = guess_round(game, "Alex")("sft")
    assert "Which reply did Alex send?" in context and "q</p>" in context
    assert (a["value"], b["value"]) == ("fake", "real")
    assert a["interactive"] and "no answers yet" in score
    a, b, score, cleared = guess_answer(game, 1)(rnd)
    assert cleared is None and "1<small> / 1" in score
    assert a["elem_classes"] == ["slip", "model"] and b["elem_classes"] == ["slip", "real"]
    assert not b["interactive"] and "Right: B was the real reply" in score
    assert guess_answer(game, 0)(None)[3] is None
    assert game.log == [True]


def test_build_app_guess_only():
    assert build_app(None, "sys", FakeGame()) is not None
    assert build_app(FakeBank(), "sys", FakeGame()) is not None
