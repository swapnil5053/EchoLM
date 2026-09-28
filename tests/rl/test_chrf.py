import pytest

from echolm.rl.chrf import char_ngrams
from echolm.rl.chrf import chrf

# reference values from sacrebleu 2.6 CHRF().sentence_score(hyp, [ref]).score / 100
CASES = [
    ("ok", "okay", 0.4717),
    ("haan bhai chal", "haan chal bhai", 0.45859),
    ("baby I miss you so much", "i miss you baby", 0.54315),
    ("kuch nahi yaar\nbas assignment", "kuch nahi", 0.61231),
    ("thank youuu", "thank you love", 0.58975),
    ("x", "yz", 0.0),
    ("a", "a", 1.0),
]


@pytest.mark.parametrize("hyp,ref,expected", CASES)
def test_matches_sacrebleu(hyp, ref, expected):
    assert chrf(hyp, ref) == pytest.approx(expected, abs=1e-4)


def test_whitespace_is_ignored():
    assert char_ngrams("a b", 2) == char_ngrams("ab", 2)


def test_empty_strings():
    assert chrf("", "haan") == 0.0
    assert chrf("haan", "") == 0.0
