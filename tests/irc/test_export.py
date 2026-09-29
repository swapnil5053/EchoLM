import json

import pytest

from echolm.irc.export import assigned_days
from echolm.irc.export import build_export
from echolm.irc.export import rank_users
from echolm.irc.export import scrub
from echolm.irc.export import write_export
from echolm.parse.telegram import parse_telegram


def test_rank_users(logs):
    ranked = rank_users(assigned_days(logs, 1.0), min_msgs=4)
    assert ranked[0] == {"nick": "alice", "replies": 10, "partners": 2}
    assert {r["nick"] for r in ranked} == {"alice", "dave", "erin", "bob"}
    assert "bob" not in {r["nick"] for r in rank_users(assigned_days(logs, 1.0), min_msgs=5)}


def test_scrub_replaces_whole_nicks_only():
    names = {"alice": "Alex", "dave": "user0001"}
    assert scrub("ask Alice or dave-bot, not davey", names) == "ask Alex or dave-bot, not davey"


def test_build_export_is_a_parseable_telegram_export(logs, tmp_path):
    data = build_export(assigned_days(logs, 1.0), "alice", "Alex", min_msgs=4)
    chats = data["chats"]["list"]
    assert [c["name"] for c in chats] == ["user0001", "user0002"]
    assert {m["from"] for c in chats for m in c["messages"]} == {"Alex", "user0001", "user0002"}
    assert len({m["id"] for c in chats for m in c["messages"]}) == sum(len(c["messages"]) for c in chats)
    p = tmp_path / "irc.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    msgs = parse_telegram(p, "Alex")
    assert sum(m.is_me for m in msgs) == 10
    assert not any("alice" in m.text.lower() for m in msgs)


def test_max_replies_keeps_latest_threads(logs):
    data = build_export(assigned_days(logs, 1.0), "alice", "Alex", min_msgs=4, max_replies=1)
    assert [c["name"] for c in data["chats"]["list"]] == ["user0001"]
    assert data["chats"]["list"][0]["messages"][0]["text"] == "can i ask you too"


def test_unknown_nick_is_a_clear_error(logs):
    with pytest.raises(ValueError, match="irc users"):
        build_export(assigned_days(logs, 1.0), "nobody", "Alex")


def test_write_export_picks_the_top_nick(logs, tmp_path):
    stats = write_export(logs, tmp_path / "out.json", None, "Alex", 1.0, 4)
    assert stats == {"partners": 2, "messages": 16, "my_messages": 10}


def test_empty_logs_is_a_clear_error(tmp_path):
    p = tmp_path / "logs.jsonl"
    p.write_text('{"day": "2024-01-01", "channel": "#ubuntu", "text": "=== join"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="irc fetch"):
        assigned_days(p, 1.0)
