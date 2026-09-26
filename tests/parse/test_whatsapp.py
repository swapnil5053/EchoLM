from datetime import datetime

import pytest

from echolm.data.models import DELETED
from echolm.data.models import MEDIA
from echolm.data.models import TEXT
from echolm.parse.whatsapp import classify
from echolm.parse.whatsapp import detect_date_order
from echolm.parse.whatsapp import parse_whatsapp

ANDROID = """\
13/03/2024, 21:14 - Messages and calls are end-to-end encrypted.
13/03/2024, 21:14 - Rohan: kya kar raha hai
13/03/2024, 21:15 - Kabir: kuch nahi
bas assignment
13/03/2024, 21:16 - Rohan: <Media omitted>
13/03/2024, 21:17 - Kabir: acha theek hai <This message was edited>
"""

IOS = """\
\u200e[3/14/24, 9:14:03\u202fPM] Rohan: bhai chal
[3/14/24, 12:05:00\u202fAM] Kabir: haan
[3/14/24, 12:06:00\u202fPM] Kabir: \u200eimage omitted
"""


def write(tmp_path, text):
    p = tmp_path / "WhatsApp Chat with Rohan.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_android_export_parses_multiline_and_system(tmp_path):
    msgs = parse_whatsapp(write(tmp_path, ANDROID), "Kabir")
    assert [m.sender for m in msgs] == ["Rohan", "Kabir", "Rohan", "Kabir"]
    assert msgs[1].text == "kuch nahi\nbas assignment"
    assert msgs[1].is_me and not msgs[0].is_me
    assert msgs[2].kind == MEDIA
    assert msgs[3].text == "acha theek hai"
    assert msgs[0].ts == datetime(2024, 3, 13, 21, 14)


def test_ios_export_with_12h_clock(tmp_path):
    msgs = parse_whatsapp(write(tmp_path, IOS), "Kabir")
    assert msgs[0].ts == datetime(2024, 3, 14, 21, 14, 3)
    assert msgs[1].ts.hour == 0
    assert msgs[2].ts.hour == 12
    assert msgs[2].kind == MEDIA


def test_date_order_detection():
    assert detect_date_order(["13/03/2024, 21:14 - A: x"]) == "dmy"
    assert detect_date_order(["3/14/24, 9:14 PM - A: x"]) == "mdy"
    assert detect_date_order(["3/4/24, 9:14 PM - A: x"]) == "dmy"


def test_explicit_date_order_overrides_detection(tmp_path):
    msgs = parse_whatsapp(write(tmp_path, "3/4/24, 09:14 - Kabir: hi\n"), "Kabir", "mdy")
    assert msgs[0].ts.month == 3 and msgs[0].ts.day == 4


@pytest.mark.parametrize("text,kind", [
    ("This message was deleted", DELETED),
    ("You deleted this message", DELETED),
    ("<Media omitted>", MEDIA),
    ("null", MEDIA),
    ("nullify karna padega", TEXT),
    ("acha", TEXT),
])
def test_classify(text, kind):
    assert classify(text) == kind


def test_hinglish_text_is_untouched(tmp_path):
    msgs = parse_whatsapp(write(tmp_path, "13/03/2024, 21:14 - Kabir: Acha BHAI 😂 kya scene\n"), "Kabir")
    assert msgs[0].text == "Acha BHAI 😂 kya scene"


def test_unknown_me_lists_senders(tmp_path):
    with pytest.raises(ValueError, match="senders are"):
        parse_whatsapp(write(tmp_path, ANDROID), "Nobody")


def test_empty_export_raises(tmp_path):
    with pytest.raises(ValueError, match="no messages found"):
        parse_whatsapp(write(tmp_path, "just some text\n"), "Kabir")
