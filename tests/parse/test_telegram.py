import json

import pytest

from echolm.data.models import FORWARDED
from echolm.data.models import MEDIA
from echolm.data.models import TEXT
from echolm.parse.telegram import chats_in
from echolm.parse.telegram import flatten_text
from echolm.parse.telegram import parse_telegram


def msg(i, frm, text, **extra):
    return {"id": i, "type": "message", "date": f"2025-01-0{i}T10:00:00", "from": frm,
            "from_id": "user1" if frm == "Kabir" else "user2", "text": text, **extra}


def chat(messages, kind="personal_chat"):
    return {"name": "Meera", "type": kind, "id": 42, "messages": messages}


def write(tmp_path, data):
    p = tmp_path / "result.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_flatten_text_entities():
    assert flatten_text(["dekh ", {"type": "link", "text": "https://x.y"}]) == "dekh https://x.y"
    assert flatten_text("acha") == "acha"


def test_single_chat_export(tmp_path):
    data = chat([
        {"id": 1, "type": "service", "date": "2025-01-01T09:00:00", "actor": "Meera", "text": ""},
        msg(2, "Meera", "kya kar rahi hai"),
        msg(3, "Kabir", "kuch nahi", reply_to_message_id=2),
    ])
    msgs = parse_telegram(write(tmp_path, data), "Kabir")
    assert [m.text for m in msgs] == ["kya kar rahi hai", "kuch nahi"]
    assert msgs[1].is_me and msgs[1].reply_to == "2"
    assert msgs[0].chat_id == "tg:42"


def test_me_matches_user_id(tmp_path):
    msgs = parse_telegram(write(tmp_path, chat([msg(1, "Kabir", "hi")])), "user1")
    assert msgs[0].is_me


def test_caption_splits_into_media_and_text(tmp_path):
    data = chat([msg(1, "Meera", "dekh", photo="p.jpg"), msg(2, "Kabir", "", media_type="sticker")])
    msgs = parse_telegram(write(tmp_path, data), "Kabir")
    assert [m.kind for m in msgs] == [MEDIA, TEXT, MEDIA]


def test_forwarded_is_marked(tmp_path):
    data = chat([msg(1, "Kabir", "fwd text", forwarded_from="Channel"), msg(2, "Kabir", "own")])
    msgs = parse_telegram(write(tmp_path, data), "Kabir")
    assert msgs[0].kind == FORWARDED


def test_full_export_keeps_only_personal_chats(tmp_path):
    data = {"chats": {"list": [chat([msg(1, "Kabir", "hi")]),
                               chat([msg(1, "Kabir", "group hi")], kind="private_group")]}}
    msgs = parse_telegram(write(tmp_path, data), "Kabir")
    assert [m.text for m in msgs] == ["hi"]


def test_no_my_messages_raises(tmp_path):
    with pytest.raises(ValueError, match="no 1:1 messages"):
        parse_telegram(write(tmp_path, chat([msg(1, "Meera", "hi")])), "Kabir")


def test_not_an_export():
    with pytest.raises(ValueError, match="not a Telegram export"):
        chats_in({"foo": 1})
