import pytest

pytest.importorskip("sklearn")

from echolm.eval.detect import detect_auc  # noqa: E402
from echolm.eval.detect import mean_auc  # noqa: E402
from tests.eval.fixtures import MINE  # noqa: E402
from tests.eval.fixtures import THEIRS  # noqa: E402

BOT = ["Hello! How can I assist you today?", "I'm sorry, but I can't help with that.",
       "Certainly! Here is what you asked for.", "Feel free to reach out anytime.",
       "I would be happy to help you with that.", "Is there anything else I can do?",
       "Great question! Let me explain.", "Thank you for your message.", "I understand your concern.",
       "Let me know if you need more help.", "Here are some suggestions for you.", "Sure, I can do that."]


def test_obvious_bot_is_detected():
    assert detect_auc(MINE, BOT, 0) > 0.9


def test_same_distribution_is_near_chance():
    auc = detect_auc(MINE[:6] * 2, MINE[6:] * 2, 0)
    assert auc < 0.8


def test_too_few_samples():
    assert detect_auc(MINE[:5], BOT[:5], 0) is None


def test_mean_auc_over_seeds():
    auc, sd = mean_auc(MINE, {1: BOT, 2: THEIRS}, 0)
    assert 0.5 < auc <= 1 and sd >= 0
    assert mean_auc(MINE, {1: BOT[:3]}, 0) == (None, None)
