from datetime import date

from echolm.irc.log import parse_day
from echolm.irc.threads import assign
from echolm.irc.threads import pair_counts
from echolm.irc.threads import threads_for


def rules_by_text(assigned):
    return {a.text: (a.line.nick, a.partner, a.rule) for a in assigned}


def test_assign_rules(day_text):
    got = rules_by_text(assign(parse_day(day_text, date(2024, 3, 1))))
    assert got["which card? run lspci | grep -i net"] == ("alice", "dave", "addressed")
    assert got["known bug, try the backport driver"] == ("alice", "dave", "answer")
    assert got["sudo apt install backport-iwlwifi-dkms"] == ("alice", "dave", "continuation")
    assert got["yes until 2027"] == ("bob", "erin", "addressed")


def test_bots_commands_and_stale_lines_are_dropped(day_text):
    got = rules_by_text(assign(parse_day(day_text, date(2024, 3, 1))))
    assert all(nick != "ubottu" for nick, _, _ in got.values())
    assert "!paste" not in got
    assert "random thought" not in got
    assert "anyone know why wifi drops after suspend?" not in got


def test_rules_can_be_switched_off(day_text):
    got = assign(parse_day(day_text, date(2024, 3, 1)), rules=("addressed",))
    assert {a.rule for a in got} == {"addressed"}


def test_threads_for_one_person(day_text):
    assigned = assign(parse_day(day_text, date(2024, 3, 1)))
    threads = threads_for(assigned, "alice")
    assert set(threads) == {"dave", "erin"}
    assert [a.line.nick for a in threads["erin"]] == ["erin", "alice", "erin", "alice"]
    assert pair_counts(assigned)[frozenset(("alice", "dave"))] == 4
