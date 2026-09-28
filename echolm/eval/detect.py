import logging
import random
import statistics

from echolm.eval.style import strip_placeholders

log = logging.getLogger(__name__)

MIN_PER_CLASS = 10
FOLDS = 5


def detect_auc(real: list[str], generated: list[str], seed: int) -> float | None:
    # a classifier tries to tell your real replies from the model's, with cross-validation on
    # equal-sized samples; 0.5 means it cannot tell them apart, 1.0 means every reply is spotted.
    # nothing in training optimizes against it, unlike the reward-backed columns
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline

    n = min(len(real), len(generated))
    if n < MIN_PER_CLASS:
        return None
    rng = random.Random(seed)
    texts = rng.sample(real, n) + rng.sample(generated, n)
    texts = [strip_placeholders(t) or "<empty>" for t in texts]
    labels = [0] * n + [1] * n
    clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True),
                        LogisticRegression(max_iter=2000, C=1.0))
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    probs = cross_val_predict(clf, texts, labels, cv=folds, method="predict_proba")[:, 1]
    return round(float(roc_auc_score(labels, probs)), 3)


def mean_auc(real: list[str], per_seed: dict[int, list[str]], seed: int) -> tuple[float | None, float | None]:
    aucs = [a for s, outs in sorted(per_seed.items()) if (a := detect_auc(real, outs, seed + s)) is not None]
    if not aucs:
        return None, None
    spread = statistics.pstdev(aucs) if len(aucs) > 1 else 0.0
    return round(statistics.fmean(aucs), 3), round(spread, 3)
