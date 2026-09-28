import json
import logging
from pathlib import Path

from echolm.data.io import read_jsonl
from echolm.data.io import write_jsonl
from echolm.eval.authorship import held_out_accuracy
from echolm.eval.authorship import train_classifier
from echolm.eval.config import EvalConfig
from echolm.eval.generate import generate_all
from echolm.eval.generate import load
from echolm.eval.generate import perplexity
from echolm.eval.generate import reply_nll
from echolm.eval.score import score_generations
from echolm.eval.score import score_reference

log = logging.getLogger(__name__)


def generate(model_ref: str, data_dir: Path, run_dir: Path, cfg: EvalConfig) -> None:
    rows = read_jsonl(data_dir / "test.jsonl")
    model, tok = load(model_ref, cfg)
    log.info("sampling %d test prompts x %d seeds with %s", len(rows), len(cfg.seeds), model_ref)
    write_jsonl(run_dir / "generations.jsonl", generate_all(model, tok, rows, cfg))
    try:
        nll = reply_nll(model, tok, rows)
    except Exception:
        # perplexity is one extra column; a failure here must not throw away the generations
        log.exception("scoring the real replies failed, perplexity will be missing")
        nll = None
    info = {"model": model_ref, "reply_nll": nll, "reply_ppl": perplexity(nll)}
    (run_dir / "nll.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def score(data_dir: Path, run_dir: Path, cfg: EvalConfig) -> dict:
    windows = read_jsonl(data_dir / "windows.jsonl")
    clf = train_classifier(windows, cfg.classifier_seed)
    metrics = score_generations(read_jsonl(run_dir / "generations.jsonl"), windows, clf)
    nll_path = run_dir / "nll.json"
    if nll_path.exists():
        metrics.update(json.loads(nll_path.read_text(encoding="utf-8")))
    metrics["classifier_accuracy"] = held_out_accuracy(clf, windows)
    metrics["reference"] = score_reference(windows, clf)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("%s: p_me %s, style_gap %s, chatbot %s, ngram_copy %s, distinct %s", run_dir.name,
             metrics["p_me"], metrics["style_gap"], metrics["chatbot_rate"], metrics["ngram_copy"],
             metrics["distinct"])
    return metrics
