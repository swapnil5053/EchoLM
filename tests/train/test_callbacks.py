import pytest

pytest.importorskip("transformers")

from echolm.train.callbacks import last_turn  # noqa: E402
from echolm.train.callbacks import pick_samples  # noqa: E402


def test_pick_samples_is_seeded_and_bounded():
    rows = [{"id": i} for i in range(20)]
    assert pick_samples(rows, 5, 1) == pick_samples(rows, 5, 1)
    assert len(pick_samples(rows, 5, 1)) == 5
    assert pick_samples(rows[:3], 5, 1) == rows[:3]


def test_last_turn():
    prompt = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\npehla<|im_end|>\n"
              "<|im_start|>assistant\nok<|im_end|>\n<|im_start|>user\nchal na<|im_end|>\n"
              "<|im_start|>assistant\n")
    assert last_turn(prompt) == "chal na"
