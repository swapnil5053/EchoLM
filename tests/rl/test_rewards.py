import pytest

from echolm.rl.rewards import CopyIndex
from echolm.rl.rewards import RewardWeights
from echolm.rl.rewards import duplicates
from echolm.rl.rewards import feature_scales
from echolm.rl.rewards import length_match
from echolm.rl.rewards import score_group
from echolm.rl.rewards import style_match

TRAIN = {"a": "baby I miss you so much im sorry", "b": "haan chal kal milte", "c": "ok"}


@pytest.fixture
def scales():
    return feature_scales(list(TRAIN.values()) + ["kuch nahi yaar", "😂😂", "Okay. See you."])


def test_copy_index_ignores_the_windows_own_reply():
    index = CopyIndex(TRAIN)
    text = "baby i miss you so much im sorry"
    assert index.copies_other(text, "b")
    assert not index.copies_other(text, "a")
    assert not index.copies_other("haan", "b")


def test_style_match_prefers_the_same_register(scales):
    ref = "haan yaar\nkal pakka 😂"
    same = style_match("acha bhai\nchal theek 😂", ref, scales)
    assert same > style_match("Sure. See you tomorrow.", ref, scales)
    assert style_match(ref, ref, scales) == 1.0


def test_length_match():
    assert length_match("a b c", "x y z") == 1.0
    assert length_match("a", "a b c d e f g h") < 0.5


def test_duplicates_normalizes_case_and_spaces():
    assert duplicates(["Haan", "haan ", "ok", "bhai"]) == [True, True, False, False]


def test_score_group_penalties(scales):
    index = CopyIndex(TRAIN)
    group = ["haan kal milte", "haan kal milte", "I'm sorry, but I can't help with that.", "",
             "baby i miss you so much im sorry"]
    parts = score_group(group, "haan chal kal milte", "b", index, scales, RewardWeights())
    assert parts[0]["duplicate"] == -1 and parts[1]["duplicate"] == -1
    assert parts[2]["chatbot"] == -1
    assert parts[3]["empty"] == -1 and parts[3]["chrf"] == 0
    assert parts[4]["copy"] == -1
    assert parts[0]["chrf"] > parts[2]["chrf"]
    assert max(p["total"] for p in parts) == parts[0]["total"]


def test_weights_scale_components(scales):
    index = CopyIndex(TRAIN)
    w = RewardWeights(chrf=0, style=0, length=0, duplicate=0, copy=0, chatbot=2, empty=0)
    parts = score_group(["How can I help you?"], "haan", "x", index, scales, w)
    assert parts[0]["total"] == -2
