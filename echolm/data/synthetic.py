import json
import logging
import random
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

from echolm.data.synthetic_dialogues import DIALOGUES

log = logging.getLogger(__name__)

ME = "Kabir"
ME_TG_ID = "user1001"
START = datetime(2025, 1, 6, 10, 0)

WA_EXTRA = [
    ("them", "<Media omitted>"), ("me", "haha kya photo hai"), ("them", "This message was deleted"),
    ("me", "kya bheja tha?\nabhi delete kyu kiya"), ("them", "Mera naya number +91 90000 00000"),
    ("me", "save kar liya <This message was edited>"),
]


def timeline(dialogues: list, rng: random.Random) -> list[list[tuple[datetime, str, str]]]:
    day = START
    sessions = []
    for dialogue in dialogues:
        day += timedelta(days=rng.randint(1, 4))
        ts = day.replace(hour=rng.randint(10, 22), minute=rng.randint(0, 59))
        rows = []
        for who, text in dialogue:
            ts += timedelta(seconds=rng.randint(20, 240))
            rows.append((ts, who, text))
        sessions.append(rows)
    return sessions


def whatsapp_text(sessions: list, friend: str) -> str:
    first = sessions[0][0][0]
    lines = [f"{first:%d/%m/%Y, %H:%M} - Messages and calls are end-to-end encrypted. "
             "No one outside of this chat, not even WhatsApp, can read or listen to them."]
    for rows in sessions:
        for ts, who, text in rows:
            name = ME if who == "me" else friend
            lines.append(f"{ts:%d/%m/%Y, %H:%M} - {name}: {text}")
    return "\n".join(lines) + "\n"


def tg_message(i: int, ts: datetime, who: str, text, friend: str) -> dict:
    return {
        "id": i, "type": "message", "date": ts.isoformat(),
        "date_unixtime": str(int(ts.replace(tzinfo=timezone.utc).timestamp())),
        "from": ME if who == "me" else friend,
        "from_id": ME_TG_ID if who == "me" else "user2002",
        "text": text, "text_entities": [],
    }


def tg_extras(start: int, ts: datetime, friend: str, quote_id: int) -> list[dict]:
    t = [ts + timedelta(minutes=k) for k in range(5)]
    photo = tg_message(start, t[0], "them", "Dekh kya mila", friend)
    photo["photo"] = "photos/photo_1.jpg"
    reply = tg_message(start + 1, t[1], "me", "bhai ye wahi hai na jo tu bol rahi thi", friend)
    reply["reply_to_message_id"] = quote_id
    fwd = tg_message(start + 2, t[2], "them", "Placement drive on 12th, register by Friday", friend)
    fwd["forwarded_from"] = "PES Placements"
    link = tg_message(start + 3, t[3], "me", ["register kar diya ", {"type": "link", "text":
                      "https://example.com/form"}], friend)
    sticker = tg_message(start + 4, t[4], "me", "", friend)
    sticker.update({"media_type": "sticker", "sticker_emoji": "😂", "file": "stickers/s.webp"})
    return [photo, reply, fwd, link, sticker]


def telegram_json(sessions: list, friend: str) -> dict:
    msgs = [{"id": 1, "type": "service", "date": sessions[0][0][0].isoformat(),
             "actor": friend, "action": "phone_call", "text": ""}]
    for rows in sessions:
        for ts, who, text in rows:
            msgs.append(tg_message(len(msgs) + 1, ts, who, text, friend))
    last = sessions[-1][-1][0] + timedelta(days=2)
    msgs += tg_extras(len(msgs) + 1, last, friend, quote_id=2)
    return {"name": friend, "type": "personal_chat", "id": 4242, "messages": msgs}


def write_synthetic(out: Path, seed: int) -> None:
    rng = random.Random(seed)
    order = list(range(len(DIALOGUES)))
    rng.shuffle(order)
    wa = timeline([DIALOGUES[i] for i in order[:12]] + [WA_EXTRA], rng)
    tg = timeline([DIALOGUES[i] for i in order[12:]], rng)
    out.mkdir(parents=True, exist_ok=True)
    (out / "whatsapp_rohan.txt").write_text(whatsapp_text(wa, "Rohan"), encoding="utf-8")
    tg_data = telegram_json(tg, "Meera")
    (out / "telegram_meera.json").write_text(json.dumps(tg_data, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    log.info("wrote synthetic exports to %s (%d + %d sessions)", out, len(wa), len(tg))
