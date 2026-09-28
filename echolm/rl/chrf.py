from collections import Counter

MAX_N = 6
BETA = 2.0


def char_ngrams(text: str, n: int) -> Counter:
    s = "".join(text.split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyp: str, ref: str) -> float:
    # chrF (Popović 2015) as sacrebleu computes it: character 1..6-grams with whitespace removed,
    # precision and recall averaged over the orders both strings have, then F-beta with beta=2
    precs, recs = [], []
    for n in range(1, MAX_N + 1):
        h, r = char_ngrams(hyp, n), char_ngrams(ref, n)
        if not h or not r:
            continue
        match = sum((h & r).values())
        precs.append(match / sum(h.values()))
        recs.append(match / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = BETA * BETA
    return (1 + b2) * p * r / (b2 * p + r)
