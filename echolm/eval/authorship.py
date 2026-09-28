import logging

from echolm.eval.style import strip_placeholders

log = logging.getLogger(__name__)


def split_texts(windows: list[dict], split: str) -> tuple[list[str], list[str]]:
    rows = [w for w in windows if w["split"] == split]
    mine = {strip_placeholders(w["target"]) for w in rows}
    theirs = {strip_placeholders(c["text"]) for w in rows for c in w["context"] if c["role"] == "them"}
    return sorted(t for t in mine if t), sorted(t for t in theirs if t)


def train_classifier(windows: list[dict], seed: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    mine, theirs = split_texts(windows, "train")
    if len(mine) < 10 or len(theirs) < 10:
        raise ValueError(f"too few messages to train the authorship classifier ({len(mine)} / {len(theirs)})")
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=False, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    clf.fit(mine + theirs, [1] * len(mine) + [0] * len(theirs))
    return clf


def held_out_accuracy(clf, windows: list[dict]) -> float | None:
    # balanced: the mean of the hit rate on my messages and on theirs, so class sizes don't matter
    mine, theirs = split_texts(windows, "test")
    if not mine or not theirs:
        return None
    hit_mine = sum(clf.predict(mine)) / len(mine)
    hit_theirs = 1 - sum(clf.predict(theirs)) / len(theirs)
    return round((hit_mine + hit_theirs) / 2, 3)


def p_me(clf, texts: list[str]) -> float:
    cleaned = [strip_placeholders(t) or t for t in texts]
    probs = clf.predict_proba(cleaned)[:, 1]
    return round(float(probs.mean()), 3)
