from echolm.eval.authorship import p_me
from echolm.eval.overlap import copy_rates
from echolm.eval.overlap import repetition
from echolm.eval.style import chatbot_rate
from echolm.eval.style import style_gap


def score_outputs(outputs: list[str], references: list[str], train_targets: list[str], clf) -> dict:
    return {
        "n": len(outputs),
        "p_me": p_me(clf, outputs),
        "style_gap": style_gap(outputs, references),
        "chatbot_rate": chatbot_rate(outputs),
        **copy_rates(outputs, train_targets),
        **repetition(outputs),
        "median_words": sorted(len(o.split()) for o in outputs)[len(outputs) // 2],
    }


def score_generations(gens: list[dict], windows: list[dict], clf) -> dict:
    train_targets = [w["target"] for w in windows if w["split"] == "train"]
    references = sorted({g["id"]: g["reference"] for g in gens}.items())
    per_seed = {}
    for seed in sorted({g["seed"] for g in gens}):
        outs = [g["output"] for g in gens if g["seed"] == seed]
        per_seed[seed] = score_outputs(outs, [r for _, r in references], train_targets, clf)
    return {"per_seed": per_seed, **average(list(per_seed.values()))}


def average(runs: list[dict]) -> dict:
    keys = ("p_me", "chatbot_rate", "exact_copy", "ngram_copy", "distinct", "top_share", "median_words")
    out = {k: round(sum(r[k] for r in runs) / len(runs), 3) for k in keys}
    out["style_gap"] = round(sum(r["style_gap"]["mean"] for r in runs) / len(runs), 3)
    return out


def score_reference(windows: list[dict], clf) -> dict:
    # the real test replies scored the same way: the ceiling a model is compared against
    refs = [w["target"] for w in windows if w["split"] == "test"]
    train_targets = [w["target"] for w in windows if w["split"] == "train"]
    scored = score_outputs(refs, refs, train_targets, clf)
    return {**{k: scored[k] for k in ("p_me", "chatbot_rate", "exact_copy", "ngram_copy",
                                       "distinct", "top_share", "median_words")},
            "style_gap": scored["style_gap"]["mean"]}
