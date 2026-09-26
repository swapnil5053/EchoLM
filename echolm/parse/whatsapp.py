import logging
import re
from datetime import datetime
from pathlib import Path

from echolm.data.models import DELETED
from echolm.data.models import MEDIA
from echolm.data.models import TEXT
from echolm.data.models import Msg

log = logging.getLogger(__name__)

# Android: "12/03/2024, 21:14 - Rohan: text"   iOS: "[12/03/24, 9:14:03 PM] Rohan: text"
HEADER = re.compile(
    r"^\[?(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4}),?\s+"
    r"(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?\s*([ap]\.?\s?m\.?)?\]?\s*(?:-\s)?(.*)$",
    re.IGNORECASE,
)
MEDIA_MARKERS = (
    "<media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "contact card omitted",
    "<attached:",
)
DELETED_MARKERS = ("this message was deleted", "you deleted this message")
EDITED_SUFFIX = re.compile(r"\s*<this message was edited>\s*$", re.IGNORECASE)
INVISIBLE = str.maketrans({"\u200e": "", "\u200f": "", "\u202f": " ", "\u00a0": " "})


def detect_date_order(lines: list[str]) -> str:
    firsts, seconds = [], []
    for line in lines[:500]:
        m = HEADER.match(line.translate(INVISIBLE))
        if m:
            firsts.append(int(m.group(1)))
            seconds.append(int(m.group(2)))
    if any(n > 12 for n in firsts):
        return "dmy"
    if any(n > 12 for n in seconds):
        return "mdy"
    log.warning("date order is ambiguous in the first 500 lines, assuming day/month")
    return "dmy"


def parse_ts(m: re.Match, order: str) -> datetime:
    a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    day, month = (a, b) if order == "dmy" else (b, a)
    year = year + 2000 if year < 100 else year
    hour, minute, sec = int(m.group(4)), int(m.group(5)), int(m.group(6) or 0)
    ampm = (m.group(7) or "").replace(".", "").replace(" ", "").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return datetime(year, month, day, hour, minute, sec)


def classify(text: str) -> str:
    low = text.strip().lower()
    if any(low.startswith(d) for d in DELETED_MARKERS):
        return DELETED
    if low == "null" or any(low.startswith(mk) for mk in MEDIA_MARKERS):
        return MEDIA
    return TEXT


def split_records(lines: list[str], order: str) -> list[tuple[datetime, str]]:
    records = []
    for raw in lines:
        line = raw.translate(INVISIBLE).rstrip("\r\n")
        m = HEADER.match(line)
        if m:
            records.append((parse_ts(m, order), m.group(8)))
        elif records:
            ts, body = records[-1]
            records[-1] = (ts, f"{body}\n{line}")
    return records


def parse_whatsapp(path: Path, me: str, date_order: str | None = None) -> list[Msg]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    order = date_order or detect_date_order(lines)
    chat_id = f"wa:{path.stem}"
    msgs, skipped = [], 0
    for ts, body in split_records(lines, order):
        sender, sep, text = body.partition(": ")
        if not sep:
            skipped += 1
            continue
        text = EDITED_SUFFIX.sub("", text)
        msgs.append(Msg(ts, chat_id, sender.strip(), text, sender.strip() == me, classify(text)))
    if not msgs:
        raise ValueError(f"no messages found in WhatsApp export {path}")
    if not any(m.is_me for m in msgs):
        names = sorted({m.sender for m in msgs})
        raise ValueError(f"sender '{me}' not found in {path.name}; senders are {names}")
    log.info("%s: %d messages, %d system lines skipped", path.name, len(msgs), skipped)
    return msgs
