import json
import logging
from pathlib import Path

from echolm.train.select import newest_run
from echolm.train.select import select_checkpoint

log = logging.getLogger(__name__)

BASE = "base"
FALLBACK_SYSTEM = "Reply to the chat exactly the way I text."


def system_prompt(data_dir: Path) -> str:
    path = data_dir / "train_sft.jsonl"
    if not path.exists():
        return FALLBACK_SYSTEM
    with path.open(encoding="utf-8") as f:
        return json.loads(f.readline())["prompt"][0]["content"]


def find_adapters(outputs: Path) -> dict[str, Path]:
    found = {}
    try:
        found["sft"] = select_checkpoint(newest_run(outputs / "sft"))
    except ValueError as e:
        log.warning("no SFT adapter: %s", e)
    grpo = sorted((p for p in (outputs / "grpo").glob("grpo-run-*") if (p / "adapter").exists()),
                  key=lambda p: p.stat().st_mtime)
    if grpo:
        found["grpo"] = grpo[-1] / "adapter"
    return found


def to_messages(system: str, history: list[dict], incoming: str) -> list[dict]:
    # in the demo you play the other person: your turns are "user", the model answers as you
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history]
    return msgs + [{"role": "user", "content": incoming}]


def bubbles(reply: str) -> list[str]:
    return [line.strip() for line in reply.split("\n") if line.strip()] or ["…"]


class ModelBank:
    def __init__(self, base_model: str, adapters: dict[str, Path]):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        from echolm.rl.policy import hf_load_kwargs

        self.tok = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, **hf_load_kwargs())
        self.names = [BASE]
        for name, path in adapters.items():
            if isinstance(model, PeftModel):
                model.load_adapter(str(path), adapter_name=name)
            else:
                model = PeftModel.from_pretrained(model, str(path), adapter_name=name)
            self.names.append(name)
            log.info("loaded %s adapter from %s", name, path)
        self.model = model.eval()

    def reply(self, name: str, messages: list[dict], temperature: float, max_new_tokens: int = 64) -> str:
        import contextlib

        import torch

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        ctx = contextlib.nullcontext()
        if name == BASE and hasattr(self.model, "disable_adapter"):
            ctx = self.model.disable_adapter()
        elif name != BASE:
            self.model.set_adapter(name)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        with ctx, torch.no_grad():
            out = self.model.generate(**enc, do_sample=temperature > 0, temperature=max(temperature, 1e-3),
                                      top_p=0.9, max_new_tokens=max_new_tokens, pad_token_id=pad)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
