import json
import logging
import re
from collections import Counter
from collections import defaultdict
from datetime import date
from pathlib import Path

from echolm.irc.log import parse_day
from echolm.irc.threads import Assigned
from echolm.irc.threads import assign
from echolm.irc.threads import threads_for

log = logging.getLogger(__name__)


def assigned_days(logs: Path, gap_min: float) -> list[list[Assigned]]:
    """One day at a time, so only the attached lines of years of logs are held in memory."""
    days = []
    with logs.open(encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                row = json.loads(raw)
                days.append(assign(parse_day(row["text"], date.fromisoformat(row["day"])), gap_min))
    days = sorted(filter(None, days), key=lambda d: d[0].line.ts)
    if not days:
        raise ValueError(f"no chat lines in {logs}; delete it and run `echolm irc fetch` again")
    log.info("%d days, %d attached lines", len(days), sum(map(len, days)))
    return days


def rank_users(days: list[list[Assigned]], min_msgs: int = 4) -> list[dict]:
    """Rank nicks by how much 1:1 conversation they have, counting partners with real back and forth."""
    pair_msgs = defaultdict(Counter)
    for day in days:
        for a in day:
            if a.line.nick != a.partner:
                pair_msgs[frozenset((a.line.nick, a.partner))][a.line.nick] += 1
    users = defaultdict(lambda: {"replies": 0, "partners": 0})
    for pair, counts in pair_msgs.items():
        if len(counts) < 2 or sum(counts.values()) < min_msgs:
            continue
        for nick in pair:
            users[nick]["replies"] += counts[nick]
            users[nick]["partners"] += 1
    ranked = sorted(users.items(), key=lambda kv: -kv[1]["replies"])
    return [{"nick": nick, **stats} for nick, stats in ranked]


def scrub(text: str, names: dict[str, str]) -> str:
    for nick, alias in names.items():
        if len(nick) >= 3:
            text = re.sub(rf"(?<![\w-]){re.escape(nick)}(?![\w-])", alias, text, flags=re.IGNORECASE)
    return text


def to_message(a: Assigned, i: int, names: dict[str, str]) -> dict:
    sender = names[a.line.nick]
    return {"id": i, "type": "message", "date": a.line.ts.isoformat(), "from": sender,
            "from_id": sender, "text": scrub(a.text, names)}


def collect(days: list[list[Assigned]], me: str, min_msgs: int) -> dict[str, list[Assigned]]:
    threads = defaultdict(list)
    for day in days:
        for partner, msgs in threads_for(day, me).items():
            threads[partner].extend(msgs)
    return {p: msgs for p, msgs in threads.items()
            if len(msgs) >= min_msgs and len({a.line.nick for a in msgs}) == 2}


def latest(threads: dict[str, list[Assigned]], me: str, max_replies: int | None) -> list[str]:
    """Most recent partners until `max_replies` of my lines are covered, returned oldest first."""
    order = sorted(threads, key=lambda p: threads[p][-1].line.ts, reverse=True)
    keep, mine = [], 0
    for p in order:
        if max_replies and mine >= max_replies:
            break
        keep.append(p)
        mine += sum(a.line.nick == me for a in threads[p])
    return sorted(keep, key=lambda p: threads[p][0].line.ts)


def build_export(days: list[list[Assigned]], me: str, alias: str, min_msgs: int = 4,
                 max_replies: int | None = None) -> dict:
    threads = collect(days, me, min_msgs)
    keep = latest(threads, me, max_replies)
    chats, i = [], 0
    for n, partner in enumerate(keep, start=1):
        names = {me: alias, partner: f"user{n:04d}"}
        msgs = []
        for a in threads[partner]:
            i += 1
            msgs.append(to_message(a, i, names))
        chats.append({"id": n, "name": names[partner], "type": "personal_chat", "messages": msgs})
    if not chats:
        raise ValueError(f"'{me}' has no 1:1 threads with at least {min_msgs} messages; "
                         "run `echolm irc users` to pick a busier nick")
    return {"about": "Reconstructed from public-domain Ubuntu IRC logs (common-pile/ubuntu_irc)",
            "chats": {"list": chats}}


def write_export(logs: Path, out: Path, me: str | None, alias: str, gap_min: float, min_msgs: int,
                 max_replies: int | None = None) -> dict:
    days = assigned_days(logs, gap_min)
    if me is None:
        ranked = rank_users(days, min_msgs)
        if not ranked:
            raise ValueError(f"no 1:1 threads with at least {min_msgs} messages in {logs}")
        me = ranked[0]["nick"]
        log.info("picked the most active nick: %s", me)
    data = build_export(days, me, alias, min_msgs, max_replies)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    chats = data["chats"]["list"]
    msgs = [m for c in chats for m in c["messages"]]
    stats = {"partners": len(chats), "messages": len(msgs),
             "my_messages": sum(m["from"] == alias for m in msgs)}
    log.info("%s: %d partners, %d messages (%d as %s)", out, stats["partners"], stats["messages"],
             stats["my_messages"], alias)
    return stats
