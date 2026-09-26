import logging
import random
from collections import Counter
from collections import defaultdict
from dataclasses import replace

from echolm.data.models import Window

log = logging.getLogger(__name__)


def n_held(n_sessions: int, frac: float) -> int:
    if frac <= 0 or n_sessions < 3:
        return 0
    return max(1, round(n_sessions * frac))


def session_splits(sessions: list[int], val_frac: float, test_frac: float) -> dict[int, str]:
    ordered = sorted(sessions)
    n_test = n_held(len(ordered), test_frac)
    n_val = n_held(len(ordered) - n_test, val_frac)
    labels = {}
    for rank, sess in enumerate(reversed(ordered)):
        labels[sess] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    return labels


def assign_splits(wins: list[Window], val_frac: float, test_frac: float) -> list[Window]:
    per_chat = defaultdict(set)
    for w in wins:
        per_chat[w.chat_id].add(w.session)
    labels = {cid: session_splits(list(s), val_frac, test_frac) for cid, s in per_chat.items()}
    return [replace(w, split=labels[w.chat_id][w.session]) for w in wins]


def cap_chat_share(wins: list[Window], share: float | None, seed: int) -> list[Window]:
    train = [w for w in wins if w.split == "train"]
    counts = Counter(w.chat_id for w in train)
    if share is None or len(counts) < 2:
        return wins
    rng = random.Random(seed)
    drop = set()
    for cid, n in counts.items():
        others = len(train) - n
        limit = int(share / (1 - share) * others) if share < 1 else n
        if n > limit:
            ids = sorted(w.id for w in train if w.chat_id == cid)
            drop.update(rng.sample(ids, n - limit))
            log.info("capped %s from %d to %d train windows", cid, n, limit)
    return [w for w in wins if w.id not in drop]


def split_windows(wins: list[Window], val_frac: float, test_frac: float,
                  share: float | None, seed: int) -> list[Window]:
    out = cap_chat_share(assign_splits(wins, val_frac, test_frac), share, seed)
    counts = Counter(w.split for w in out)
    log.info("split: %s", dict(counts))
    return out
