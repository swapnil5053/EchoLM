import math
import statistics
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass

from echolm.eval.overlap import NGRAM
from echolm.eval.overlap import ngrams
from echolm.eval.overlap import norm
from echolm.eval.style import CHATBOT
from echolm.eval.style import features
from echolm.rl.chrf import chrf


@dataclass
class RewardWeights:
    chrf: float = 1.0
    style: float = 0.5
    length: float = 0.5
    duplicate: float = 0.5
    copy: float = 1.0
    chatbot: float = 1.0
    empty: float = 1.0


# 6-gram index over training replies, so a completion can be checked against every reply but its own
class CopyIndex:
    def __init__(self, replies: dict[str, str]):
        self.owners = defaultdict(set)
        for rid, text in replies.items():
            for gram in ngrams(text, NGRAM):
                self.owners[gram].add(rid)

    def copies_other(self, text: str, own_id: str) -> bool:
        return any(self.owners.get(g, set()) - {own_id} for g in ngrams(text, NGRAM))


def feature_scales(replies: list[str]) -> dict[str, float]:
    rows = [scaled_features(t) for t in replies]
    return {k: statistics.pstdev([r[k] for r in rows]) or 1.0 for k in rows[0]}


def scaled_features(text: str) -> dict[str, float]:
    f = features(text)
    for k in ("chars", "words", "lines"):
        f[k] = math.log1p(f[k])
    return f


def style_match(out: str, ref: str, scales: dict[str, float]) -> float:
    # against the reply to this message, not an average reply, which would reward the most typical answer
    fo, fr = scaled_features(out), scaled_features(ref)
    gaps = [min(1.0, abs(fo[k] - fr[k]) / scales[k]) for k in fr]
    return 1.0 - sum(gaps) / len(gaps)


def length_match(out: str, ref: str) -> float:
    return math.exp(-abs(math.log1p(len(out.split())) - math.log1p(len(ref.split()))))


def duplicates(group: list[str]) -> list[bool]:
    counts = Counter(norm(o) for o in group)
    return [counts[norm(o)] > 1 for o in group]


def score_group(group: list[str], ref: str, own_id: str, index: CopyIndex,
                scales: dict[str, float], w: RewardWeights) -> list[dict]:
    dup = duplicates(group)
    out = []
    for text, is_dup in zip(group, dup, strict=True):
        empty = not text.strip()
        parts = {
            "chrf": 0.0 if empty else chrf(text, ref),
            "style": 0.0 if empty else style_match(text, ref, scales),
            "length": 0.0 if empty else length_match(text, ref),
            "duplicate": -float(is_dup),
            "copy": -float(index.copies_other(text, own_id)),
            "chatbot": -float(bool(CHATBOT.search(text))),
            "empty": -float(empty),
        }
        parts["total"] = sum(getattr(w, k) * v for k, v in parts.items())
        out.append(parts)
    return out
