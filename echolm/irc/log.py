import re
from dataclasses import dataclass
from datetime import date
from datetime import datetime

LINE = re.compile(r"^\[(\d{1,2}):(\d{2})(?::\d{2})?\] <[@+%]?([^>\s]+)> ?(.*)$")
ADDRESS = re.compile(r"^@?([^\s:,]+)\s*[:,]\s*(.*)$", re.DOTALL)
BOT = re.compile(r"(bot\d*|^l?ubot+u\d*)\W*$", re.IGNORECASE)


@dataclass
class Line:
    idx: int
    ts: datetime
    nick: str
    text: str


def is_bot(nick: str) -> bool:
    return bool(BOT.search(nick))


def parse_day(text: str, day: date) -> list[Line]:
    out = []
    for idx, raw in enumerate(text.splitlines()):
        m = LINE.match(raw.strip())
        if not m:
            continue
        hh, mm, nick, body = m.groups()
        ts = datetime(day.year, day.month, day.day, int(hh) % 24, int(mm))
        out.append(Line(idx, ts, nick, body.strip()))
    return out


def addressee(text: str, known: dict[str, str]) -> tuple[str | None, str]:
    """Return (nick, rest) when the message starts with "nick:" or "nick," for a nick seen in the channel.

    `known` maps lower-cased nicks to their spelling so "Bob2:" and "bob2:" resolve to the same person.
    """
    m = ADDRESS.match(text)
    if not m:
        return None, text
    nick = known.get(m.group(1).lower())
    if nick is None:
        return None, text
    return nick, m.group(2).strip()


def speakers(lines: list[Line]) -> dict[str, str]:
    return {ln.nick.lower(): ln.nick for ln in lines}
