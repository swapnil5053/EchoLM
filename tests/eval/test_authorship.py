import pytest

pytest.importorskip("sklearn")

from echolm.eval.authorship import held_out_accuracy  # noqa: E402
from echolm.eval.authorship import p_me  # noqa: E402
from echolm.eval.authorship import split_texts  # noqa: E402
from echolm.eval.authorship import train_classifier  # noqa: E402

MINE = ["haan bhai", "acha yaar", "bas bhai kuch nahi", "chal bhai", "haan yaar theek", "bhai 😂",
        "kya scene hai", "nahi yaar", "acha acha", "haan chal", "bhai sun", "theek hai bhai"]
THEIRS = ["Okay, see you tomorrow.", "Did you finish the report?", "Please call me back.",
          "That sounds good.", "What time works for you?", "Sure, no problem.", "Thanks a lot.",
          "I will check and tell you.", "Are you coming today?", "Let me know.", "Good night.", "See you."]


def windows(split):
    return [{"split": split, "target": m,
             "context": [{"role": "them", "text": t}, {"role": "me", "text": "x"}]}
            for m, t in zip(MINE, THEIRS, strict=True)]


def test_split_texts_strips_placeholders():
    ws = [{"split": "train", "target": "<URL>", "context": [{"role": "them", "text": "[media]"}]},
          {"split": "train", "target": "haan", "context": [{"role": "them", "text": "ok"}]}]
    assert split_texts(ws, "train") == (["haan"], ["ok"])


def test_classifier_separates_the_two_styles():
    ws = windows("train") + windows("test")
    clf = train_classifier(ws, 0)
    assert p_me(clf, ["haan bhai yaar"]) > p_me(clf, ["Okay, sounds good. See you tomorrow."])
    assert held_out_accuracy(clf, ws) > 0.5


def test_classifier_needs_enough_data():
    with pytest.raises(ValueError, match="too few"):
        train_classifier(windows("train")[:3], 0)
