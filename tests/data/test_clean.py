import pytest

from echolm.data.clean import clean
from echolm.data.clean import dedupe
from echolm.data.clean import mask_pii


@pytest.mark.parametrize("text,expected", [
    ("mera number +91 98765 43210 hai", "mera number <PHONE> hai"),
    ("call 9876543210", "call <PHONE>"),
    ("mail kar abc.def@gmail.com pe", "mail kar <EMAIL> pe"),
    ("dekh https://example.com/x?y=1 ye", "dekh <URL> ye"),
    ("otp 482913 hai code", "otp <CODE> hai code"),
    ("5 baje milte, 2024 batch", "5 baje milte, 2024 batch"),
    ("scores 2024\n2025\n2026 wale", "scores 2024\n2025\n2026 wale"),
    ("Acha BHAI kya scene 😂", "Acha BHAI kya scene 😂"),
])
def test_mask_pii(text, expected):
    assert mask_pii(text, []) == expected


def test_blocked_words_are_case_insensitive():
    assert mask_pii("Priya ko bol", ["priya"]) == "<REDACTED> ko bol"


def test_dedupe_drops_repeated_platform_ids(mk):
    a = mk(0, True, "hi", msg_id="7")
    assert dedupe([a, mk(0, True, "hi", msg_id="7"), mk(1, True, "hi", msg_id="8")]) == [
        a, mk(1, True, "hi", msg_id="8")]


def test_dedupe_keeps_same_minute_repeats_without_ids(mk):
    msgs = [mk(0, True, "haha"), mk(0, True, "haha")]
    assert dedupe(msgs) == msgs


def test_dedupe_keeps_caption_and_media_with_same_id(mk):
    msgs = [mk(0, False, "", kind="media", msg_id="3"), mk(0, False, "dekh", msg_id="3")]
    assert dedupe(msgs) == msgs


def test_clean_sorts_and_keeps_placeholders(mk):
    msgs = [mk(2, True, "later"), mk(1, False, "x", kind="deleted"), mk(0, False, "first")]
    out = clean(msgs, [])
    assert [m.text for m in out] == ["first", "x", "later"]
