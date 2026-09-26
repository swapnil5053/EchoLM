import logging
from collections import defaultdict
from pathlib import Path

from echolm.data.models import Msg
from echolm.parse.telegram import parse_telegram
from echolm.parse.whatsapp import parse_whatsapp

log = logging.getLogger(__name__)


def drop_group_chats(msgs: list[Msg]) -> list[Msg]:
    senders = defaultdict(set)
    for m in msgs:
        senders[m.chat_id].add(m.sender)
    groups = {cid for cid, names in senders.items() if len(names) > 2}
    for cid in sorted(groups):
        log.warning("skipping %s: %d senders, group chats are not supported yet",
                    cid, len(senders[cid]))
    return [m for m in msgs if m.chat_id not in groups]


def load_export(path: Path, me: str, date_order: str | None = None) -> list[Msg]:
    if path.suffix.lower() == ".txt":
        msgs = parse_whatsapp(path, me, date_order)
    elif path.suffix.lower() == ".json":
        msgs = parse_telegram(path, me)
    else:
        raise ValueError(f"unsupported export type '{path.suffix}' for {path.name}; "
                         "expected WhatsApp .txt or Telegram .json")
    return drop_group_chats(msgs)
