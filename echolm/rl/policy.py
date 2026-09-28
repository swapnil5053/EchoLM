import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from echolm.rl.config import GrpoConfig

log = logging.getLogger(__name__)

# torch / transformers / peft are imported inside functions: unsloth has to be imported before them
# so its patches apply, and the reward code and tests should not need a GPU stack


@dataclass
class Policy:
    model: object
    tok: object
    to_inference: object
    to_training: object


def lora_settings(adapter: Path) -> dict:
    raw = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    targets = raw["target_modules"]
    return {"r": raw["r"], "lora_alpha": raw["lora_alpha"], "lora_dropout": raw.get("lora_dropout", 0.0),
            "target_modules": sorted(targets) if isinstance(targets, list) else targets}


def load_adapter_weights(model, adapter: Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    result = set_peft_model_state_dict(model, load_file(str(adapter / "adapter_model.safetensors")))
    missing = [k for k in result.missing_keys if "lora_" in k]
    if result.unexpected_keys or missing:
        raise ValueError(f"adapter {adapter} does not fit the model: {len(result.unexpected_keys)} "
                         f"unexpected and {len(missing)} missing LoRA weights")


def load_unsloth(cfg: GrpoConfig, adapter: Path) -> Policy:
    # the loss needs per-token logits; unsloth skips computing them by default to save memory
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model, max_seq_length=cfg.max_seq_len, load_in_4bit=True, dtype=None
    )
    model = FastLanguageModel.get_peft_model(
        model, bias="none", use_gradient_checkpointing="unsloth", random_state=cfg.seed,
        **lora_settings(adapter),
    )
    load_adapter_weights(model, adapter)
    return Policy(model, tok, FastLanguageModel.for_inference, FastLanguageModel.for_training)


def to_eval(model) -> None:
    model.eval()


def to_train(model) -> None:
    model.train()


def load_hf(cfg: GrpoConfig, adapter: Path) -> Policy:
    import torch
    import transformers
    from packaging.version import Version
    from peft import PeftModel
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    cuda = torch.cuda.is_available()
    # renamed from torch_dtype in transformers 4.56
    dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kwargs = {dtype_key: torch.bfloat16 if cuda else torch.float32}
    if cuda:
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    return Policy(model, tok, to_eval, to_train)


def load_policy(cfg: GrpoConfig, adapter: Path) -> Policy:
    log.info("loading %s with the %s backend, LoRA from %s", cfg.base_model, cfg.backend, adapter)
    policy = load_unsloth(cfg, adapter) if cfg.backend == "unsloth" else load_hf(cfg, adapter)
    trainable = sum(p.numel() for p in policy.model.parameters() if p.requires_grad)
    if trainable == 0:
        raise ValueError(f"no trainable parameters after loading {adapter}")
    log.info("%.1fM trainable LoRA parameters", trainable / 1e6)
    return policy


def eos_ids(policy: Policy) -> set[int]:
    ids = {policy.tok.eos_token_id}
    gen = getattr(policy.model, "generation_config", None)
    extra = getattr(gen, "eos_token_id", None) if gen else None
    ids.update(extra if isinstance(extra, list) else [extra])
    return {i for i in ids if i is not None}
