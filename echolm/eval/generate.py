import logging
import math
import os

from echolm.eval.config import EvalConfig
from echolm.train.data import render
from echolm.train.data import tokenize

log = logging.getLogger(__name__)

BASE = "base"


def load(model_ref: str, cfg: EvalConfig):
    # full logits are needed to score the real replies; unsloth drops them by default to save memory
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    from unsloth import FastLanguageModel

    name = cfg.base_model if model_ref == BASE else model_ref
    model, tok = FastLanguageModel.from_pretrained(
        model_name=name, max_seq_length=cfg.max_seq_len, load_in_4bit=True, dtype=None
    )
    FastLanguageModel.for_inference(model)
    return model, tok


def as_train_row(row: dict) -> dict:
    return {"id": row["id"], "prompt": row["prompt"],
            "completion": [{"role": "assistant", "content": row["reference"]}]}


def sample_reply(model, tok, prompt: str, cfg: EvalConfig, seed: int) -> str:
    import torch

    torch.manual_seed(seed)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, do_sample=True,
                             temperature=cfg.temperature, top_p=cfg.top_p, pad_token_id=pad)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_all(model, tok, rows: list[dict], cfg: EvalConfig) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        prompt = render(as_train_row(row), tok)["prompt"]
        for seed in cfg.seeds:
            out.append({"id": row["id"], "seed": seed, "reference": row["reference"],
                        "output": sample_reply(model, tok, prompt, cfg, seed)})
        if (i + 1) % 10 == 0:
            log.info("generated replies for %d / %d test prompts", i + 1, len(rows))
    return out


def reply_nll(model, tok, rows: list[dict]) -> float:
    import torch
    import torch.nn.functional as F

    total, count = 0.0, 0
    for row in rows:
        t = tokenize(render(as_train_row(row), tok), tok)
        ids = torch.tensor([t["input_ids"]], device=model.device)
        labels = torch.tensor([t["labels"]], device=model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
        loss = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100, reduction="sum")
        total += loss.item()
        count += int((labels[0, 1:] != -100).sum())
    return round(total / count, 4)


def perplexity(nll: float | None) -> float | None:
    return None if nll is None else round(math.exp(nll), 2)
