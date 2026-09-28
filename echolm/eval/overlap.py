from collections import Counter

NGRAM = 6
# replies shorter than this ("haan", "ok bhai") match training text by chance, not by copying
MIN_COPY_WORDS = 4


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = norm(text).split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def copy_rates(outputs: list[str], train_targets: list[str]) -> dict:
    exact = {norm(t) for t in train_targets}
    grams = set().union(*(ngrams(t, NGRAM) for t in train_targets)) if train_targets else set()
    long_outs = [o for o in outputs if len(o.split()) >= MIN_COPY_WORDS]
    gram_outs = [o for o in outputs if len(o.split()) >= NGRAM]
    exact_hits = sum(norm(o) in exact for o in long_outs)
    gram_hits = sum(bool(ngrams(o, NGRAM) & grams) for o in gram_outs)
    return {
        "exact_copy": round(exact_hits / max(1, len(long_outs)), 3),
        "ngram_copy": round(gram_hits / max(1, len(gram_outs)), 3),
        "n_long": len(long_outs),
    }


def repetition(outputs: list[str]) -> dict:
    counts = Counter(norm(o) for o in outputs)
    return {
        "distinct": round(len(counts) / len(outputs), 3),
        "top_share": round(counts.most_common(1)[0][1] / len(outputs), 3),
    }
