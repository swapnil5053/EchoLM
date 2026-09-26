import json
import logging
from datetime import datetime
from pathlib import Path

from echolm.data.models import FORWARDED
from echolm.data.models import MEDIA
from echolm.data.models import TEXT
from echolm.data.models import Msg

log = logging.getLogger(__name__)

MEDIA_KEYS = ("photo", "file", "media_type", "sticker_emoji", "location_information", "poll")


def flatten_text(field) -> str:
    if isinstance(field, str):
        return field
    parts = [p if isinstance(p, str) else p.get("text", "") for p in field or []]
    return "".join(parts)


def chats_in(data: dict) -> list[dict]:
    if "chats" in data:
        return data["chats"].get("list", [])
    if "messages" in data:
        return [data]
    raise ValueError("not a Telegram export: expected a 'messages' or 'chats' key")


def convert(raw: dict, chat_id: str, me: str) -> list[Msg]:
    if raw.get("type") != "message":
        return []
    sender = raw.get("from") or ""
    is_me = me in (raw.get("from_id"), sender)
    ts = datetime.fromisoformat(raw["date"])
    base = {"ts": ts, "chat_id": chat_id, "sender": sender, "is_me": is_me,
            "msg_id": str(raw.get("id")), "reply_to": None}
    if raw.get("reply_to_message_id") is not None:
        base["reply_to"] = str(raw["reply_to_message_id"])
    text = flatten_text(raw.get("text", ""))
    if "forwarded_from" in raw:
        return [Msg(text=text, kind=FORWARDED, **base)]
    out = []
    if any(k in raw for k in MEDIA_KEYS):
        out.append(Msg(text="", kind=MEDIA, **base))
    if text.strip():
        out.append(Msg(text=text, kind=TEXT, **base))
    return out


def parse_chat(chat: dict, me: str) -> list[Msg]:
    chat_id = f"tg:{chat.get('id', chat.get('name', 'unknown'))}"
    msgs = []
    for raw in chat.get("messages", []):
        msgs.extend(convert(raw, chat_id, me))
    return msgs


def parse_telegram(path: Path, me: str) -> list[Msg]:
    data = json.loads(path.read_text(encoding="utf-8"))
    msgs = []
    for chat in chats_in(data):
        if chat.get("type") != "personal_chat":
            log.info("skipping %s chat '%s' (1:1 chats only)", chat.get("type"), chat.get("name"))
            continue
        chat_msgs = parse_chat(chat, me)
        if not any(m.is_me for m in chat_msgs):
            log.warning("no messages from '%s' in chat '%s', skipping", me, chat.get("name"))
            continue
        msgs.extend(chat_msgs)
    if not msgs:
        raise ValueError(f"no 1:1 messages from '{me}' found in Telegram export {path}")
    log.info("%s: %d messages", path.name, len(msgs))
    return msgs
