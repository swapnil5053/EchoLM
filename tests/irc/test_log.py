from datetime import date
from datetime import datetime

from echolm.irc.log import addressee
from echolm.irc.log import is_bot
from echolm.irc.log import parse_day
from echolm.irc.log import speakers


def test_parse_day_keeps_raw_line_numbers(day_text):
    lines = parse_day(day_text, date(2024, 3, 1))
    assert lines[0].idx == 1
    assert lines[0].nick == "dave"
    assert lines[0].ts == datetime(2024, 3, 1, 9, 0)
    assert all(ln.nick != "carol" for ln in lines)


def test_parse_day_handles_seconds_and_op_prefix():
    lines = parse_day("[23:59:07] <@op> hi\nnot a log line", date(2024, 1, 1))
    assert [(ln.nick, ln.text, ln.ts.minute) for ln in lines] == [("op", "hi", 59)]


def test_addressee_needs_a_known_nick():
    known = {"alice": "alice", "bob2": "Bob2"}
    assert addressee("alice: try this", known) == ("alice", "try this")
    assert addressee("BOB2, yes", known) == ("Bob2", "yes")
    assert addressee("note: it breaks", known) == (None, "note: it breaks")
    assert addressee("plain text", known) == (None, "plain text")


def test_bots():
    assert is_bot("ubottu") and is_bot("lubotu3`") and is_bot("helpbot2")
    assert not is_bot("bob2") and not is_bot("robotnik_fan")


def test_speakers_maps_lowercase(day_text):
    assert speakers(parse_day(day_text, date(2024, 3, 1)))["alice"] == "alice"
