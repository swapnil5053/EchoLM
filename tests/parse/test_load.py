import pytest

from echolm.data.models import Msg
from echolm.parse.load import drop_group_chats
from echolm.parse.load import load_export


def test_drops_chats_with_more_than_two_senders(mk):
    one = [mk(0, False), mk(1, True)]
    group = [mk(0, False, chat="g"), mk(1, True, chat="g"),
             Msg(one[0].ts, "g", "third", "yo", False)]
    assert drop_group_chats(one + group) == one


def test_dispatches_on_extension(tmp_path):
    p = tmp_path / "chat.txt"
    p.write_text("13/03/2024, 21:14 - Kabir: hi\n13/03/2024, 21:15 - Rohan: yo\n", encoding="utf-8")
    assert len(load_export(p, "Kabir")) == 2


def test_rejects_unknown_extension(tmp_path):
    p = tmp_path / "chat.csv"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported export type"):
        load_export(p, "Kabir")
