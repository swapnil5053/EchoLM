import logging
import random
from collections.abc import Callable

import torch
from transformers import TrainerCallback

log = logging.getLogger(__name__)


def pick_samples(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, n)


def last_turn(prompt: str) -> str:
    # the rendered prompt ends with "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"
    body = prompt.rsplit("<|im_start|>user\n", 1)[-1]
    return body.split("<|im_end|>", 1)[0]


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int) -> str:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class SampleCallback(TrainerCallback):
    def __init__(self, tok, rows: list[dict], max_new_tokens: int, use_wandb: bool,
                 to_inference: Callable, to_training: Callable):
        self.tok = tok
        self.rows = rows
        self.max_new_tokens = max_new_tokens
        self.use_wandb = use_wandb
        self.to_inference = to_inference
        self.to_training = to_training

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not state.is_world_process_zero:
            return
        self.to_inference(model)
        outs = [generate(model, self.tok, r["prompt"], self.max_new_tokens) for r in self.rows]
        self.to_training(model)
        table = [[state.global_step, last_turn(r["prompt"]), r["completion"].split("<|im_end|>")[0], out]
                 for r, out in zip(self.rows, outs, strict=True)]
        for _, them, real, fake in table:
            log.info("step %d | them: %r | real: %r | model: %r", state.global_step, them[-80:], real, fake)
        if self.use_wandb:
            import wandb

            wandb.log({"samples": wandb.Table(columns=["step", "them", "real", "model"], data=table)})


class VramCallback(TrainerCallback):
    def __init__(self):
        self.peak_gb = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            self.peak_gb = max(self.peak_gb, torch.cuda.max_memory_reserved() / 2**30)
            log.info("step %d peak vram %.2f GB", state.global_step, self.peak_gb)
