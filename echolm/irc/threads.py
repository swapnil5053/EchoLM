from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from echolm.irc.log import Line
from echolm.irc.log import addressee
from echolm.irc.log import is_bot
from echolm.irc.log import speakers

RULES = ("addressed", "answer", "continuation")


@dataclass
class Assigned:
    line: Line
    partner: str
    rule: str
    text: str


def usable(ln: Line) -> bool:
    return bool(ln.text) and not ln.text.startswith("!") and not is_bot(ln.nick)


def assign(lines: list[Line], gap_min: float = 1.0, rules: tuple[str, ...] = RULES) -> list[Assigned]:
    """Attach each line to the one person it is talking to, or drop it.

    addressed: "nick: ..." prefix. answer: first unprefixed line after being addressed by nick.
    continuation: unprefixed line right after the speaker's own attached line.
    """
    known = speakers(lines)
    gap = timedelta(minutes=gap_min)
    last_out, last_in, out = {}, {}, []
    for ln in filter(usable, lines):
        to, rest = addressee(ln.text, known)
        hit = None
        if to and to != ln.nick and "addressed" in rules:
            hit = Assigned(ln, to, "addressed", rest)
        elif to is None:
            hit = follow(ln, last_out.get(ln.nick), last_in.get(ln.nick), gap, rules)
        last_in.pop(ln.nick, None)
        last_out[ln.nick] = (hit.partner, ln.ts) if hit else None
        if hit and hit.rule == "addressed":
            last_in[hit.partner] = (ln.nick, ln.ts)
        if hit and rest:
            out.append(hit)
    return out


def follow(ln: Line, prev, incoming, gap: timedelta, rules: tuple[str, ...]) -> Assigned | None:
    if incoming and "answer" in rules and ln.ts - incoming[1] <= gap:
        return Assigned(ln, incoming[0], "answer", ln.text)
    if prev and prev[0] and "continuation" in rules and ln.ts - prev[1] <= gap:
        return Assigned(ln, prev[0], "continuation", ln.text)
    return None


def pair_counts(assigned: list[Assigned]) -> Counter:
    return Counter(frozenset((a.line.nick, a.partner)) for a in assigned if a.line.nick != a.partner)


def threads_for(assigned: list[Assigned], me: str) -> dict[str, list[Assigned]]:
    out = defaultdict(list)
    for a in assigned:
        if a.line.nick == me and a.partner != me:
            out[a.partner].append(a)
        elif a.partner == me and a.line.nick != me:
            out[a.line.nick].append(a)
    return dict(out)
