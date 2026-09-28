import random
import statistics

from echolm.eval.detect import detect_auc
from echolm.eval.detect import mean_auc
from echolm.eval.overlap import copy_rates
from echolm.eval.overlap import repetition
from echolm.eval.style import chatbot_rate
from echolm.eval.style import style_gap
from echolm.rl.chrf import chrf

BOOTSTRAP = 1000
AVERAGED = ("chrf", "style_gap", "chatbot_rate", "exact_copy", "ngram_copy", "distinct", "median_words")


def score_outputs(outputs: list[str], references: list[str], train_targets: list[str]) -> dict:
    return {
        "n": len(outputs),
        "chrf": round(statistics.fmean(chrf(o, r) for o, r in zip(outputs, references, strict=True)), 4),
        "style_gap": style_gap(outputs, references)["mean"],
        "chatbot_rate": chatbot_rate(outputs),
        **copy_rates(outputs, train_targets),
        **repetition(outputs),
        "median_words": sorted(len(o.split()) for o in outputs)[len(outputs) // 2],
    }


def chrf_interval(gens: list[dict], seed: int) -> list[float]:
    # 95% bootstrap interval over test prompts, each prompt's chrF averaged over its samples
    per_prompt = {}
    for g in gens:
        per_prompt.setdefault(g["id"], []).append(chrf(g["output"], g["reference"]))
    means = [statistics.fmean(v) for v in per_prompt.values()]
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(means, k=len(means))) for _ in range(BOOTSTRAP))
    return [round(boots[int(0.025 * BOOTSTRAP)], 4), round(boots[int(0.975 * BOOTSTRAP) - 1], 4)]


def score_generations(gens: list[dict], windows: list[dict], seed: int) -> dict:
    train_targets = [w["target"] for w in windows if w["split"] == "train"]
    real = [w["target"] for w in windows if w["split"] == "test"]
    seeds = sorted({g["seed"] for g in gens})
    per_seed, outputs = {}, {}
    for s in seeds:
        rows = [g for g in gens if g["seed"] == s]
        outputs[s] = [g["output"] for g in rows]
        per_seed[s] = score_outputs(outputs[s], [g["reference"] for g in rows], train_targets)
    out = {k: round(statistics.fmean(p[k] for p in per_seed.values()), 4) for k in AVERAGED}
    out["detect_auc"], out["detect_auc_sd"] = mean_auc(real, outputs, seed)
    out["chrf_ci"] = chrf_interval(gens, seed)
    return {"per_seed": per_seed, **out}


def score_reference(windows: list[dict], seed: int) -> dict:
    # the real test replies scored like a model, for the columns where that means something.
    # detect_auc here pits your later (test) replies against your earlier (train) ones: the lowest
    # a model can realistically reach, since your own style drifts over time
    refs = [w["target"] for w in windows if w["split"] == "test"]
    train_targets = [w["target"] for w in windows if w["split"] == "train"]
    scored = score_outputs(refs, refs, train_targets)
    keep = ("chatbot_rate", "exact_copy", "ngram_copy", "distinct", "median_words", "style_gap")
    return {**{k: scored[k] for k in keep}, "detect_auc": detect_auc(refs, train_targets, seed)}
