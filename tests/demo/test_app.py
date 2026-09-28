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
