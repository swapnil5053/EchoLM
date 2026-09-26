import hashlib
import logging
from collections import Counter
from datetime import timedelta
from itertools import groupby

from echolm.config import DataConfig
from echolm.data.models import Msg
from echolm.data.models import Turn
from echolm.data.models import Window
from echolm.data.models import render_msg

log = logging.getLogger(__name__)


def split_sessions(msgs: list[Msg], gap_min: int) -> list[list[Msg]]:
    gap = timedelta(minutes=gap_min)
    out = []
    for m in msgs:
        if out and m.ts - out[-1][-1].ts <= gap:
            out[-1].append(m)
        else:
            out.append([m])
    return out


def to_turns(session: list[Msg]) -> list[Turn]:
    turns = []
    for m in session:
        if turns and turns[-1].is_me == m.is_me:
            turns[-1].msgs.append(m)
        else:
            turns.append(Turn(m.is_me, [m]))
    return turns


def build_context(prev: list[Turn], max_turns: int, max_chars: int) -> list[tuple[Turn, str]]:
    picked, used = [], 0
    for turn in reversed(prev[-max_turns:]):
        text = turn.render()
        if used + len(text) > max_chars:
            if not picked:
                picked.append((turn, text[-max_chars:]))
            break
        picked.append((turn, text))
        used += len(text)
    return picked[::-1]


def find_quote(target: Turn, context: list[tuple[Turn, str]], by_id: dict[str, Msg]) -> str | None:
    shown = {m.msg_id for t, _ in context for m in t.msgs}
    for m in target.msgs:
        if m.reply_to and m.reply_to not in shown and m.reply_to in by_id:
            return render_msg(by_id[m.reply_to])
    return None


def window_id(chat_id: str, ts: str, target: str) -> str:
    return hashlib.sha1(f"{chat_id}|{ts}|{target}".encode()).hexdigest()[:12]


def make_window(
    turns: list[Turn], i: int, sess: int, by_id: dict[str, Msg], cfg: DataConfig
) -> tuple[Window | None, str | None]:
    target = turns[i].own_text()
    if not target:
        return None, "empty_target"
    if len(target) > cfg.max_target_chars:
        return None, "target_too_long"
    if len(target) < cfg.min_target_chars:
        return None, "target_too_short"
    context = build_context(turns[:i], cfg.max_context_turns, cfg.max_context_chars)
    if not context and not cfg.include_openers:
        return None, "opener"
    ts = turns[i].msgs[0].ts.isoformat()
    chat_id = turns[i].msgs[0].chat_id
    rows = [{"role": "me" if t.is_me else "them", "text": text} for t, text in context]
    quoted = find_quote(turns[i], context, by_id)
    return Window(window_id(chat_id, ts, target), chat_id, sess, ts, rows, target, quoted), None


def chat_windows(msgs: list[Msg], cfg: DataConfig, skips: Counter) -> list[Window]:
    by_id = {m.msg_id: m for m in msgs if m.msg_id}
    out = []
    for sess, session in enumerate(split_sessions(msgs, cfg.session_gap_min)):
        turns = to_turns(session)
        for i, turn in enumerate(turns):
            if not turn.is_me:
                continue
            win, reason = make_window(turns, i, sess, by_id, cfg)
            if win:
                out.append(win)
            else:
                skips[reason] += 1
    return out


def build_windows(msgs: list[Msg], cfg: DataConfig) -> list[Window]:
    skips = Counter()
    out = []
    ordered = sorted(msgs, key=lambda m: (m.chat_id, m.ts))
    for _, chat in groupby(ordered, key=lambda m: m.chat_id):
        out.extend(chat_windows(list(chat), cfg, skips))
    if not out:
        raise ValueError(f"no training windows produced; skipped: {dict(skips)}")
    log.info("built %d windows, skipped %s", len(out), dict(skips))
    return out
