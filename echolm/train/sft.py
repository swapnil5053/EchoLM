import json
import logging
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

from echolm.train.config import SftConfig
from echolm.train.data import drop_too_long
from echolm.train.data import file_hash
from echolm.train.data import length_report
from echolm.train.data import load_split
from echolm.train.data import model_columns
from echolm.train.data import pad_batch
from echolm.train.runtime import git_sha
from echolm.train.runtime import keep_awake
from echolm.train.runtime import log_to_file
from echolm.train.runtime import setup_wandb

log = logging.getLogger(__name__)

# torch / transformers are imported inside functions: unsloth has to be imported before them
# so its patches apply, and `echolm parse` / `format` shouldn't pay for a torch import


def load_model(cfg: SftConfig):
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg.model, max_seq_length=cfg.max_seq_len, load_in_4bit=True, dtype=None
    )
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets, bias="none",
        use_gradient_checkpointing="unsloth", random_state=cfg.seed,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok, FastLanguageModel


def warmup_steps(n_train: int, cfg: SftConfig, max_steps: int) -> int:
    per_epoch = math.ceil(n_train / (cfg.batch_size * cfg.grad_accum))
    total = max_steps if max_steps > 0 else math.ceil(per_epoch * cfg.epochs)
    return max(1, round(cfg.warmup_ratio * total))


def train_args(cfg: SftConfig, out_dir: Path, n_train: int, precision: str, max_steps: int = -1) -> dict:
    eval_steps = min(cfg.eval_steps, max_steps) if max_steps > 0 else cfg.eval_steps
    return {
        "output_dir": str(out_dir), "run_name": out_dir.name, "seed": cfg.seed,
        "num_train_epochs": cfg.epochs, "max_steps": max_steps, "learning_rate": cfg.lr,
        "per_device_train_batch_size": cfg.batch_size, "per_device_eval_batch_size": cfg.eval_batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "lr_scheduler_type": "cosine", "warmup_steps": warmup_steps(n_train, cfg, max_steps),
        "weight_decay": cfg.weight_decay, "optim": "adamw_8bit",
        "logging_steps": cfg.log_steps, "logging_first_step": True,
        "eval_strategy": "steps", "eval_steps": eval_steps, "eval_on_start": True,
        "save_strategy": "steps", "save_steps": eval_steps, "save_total_limit": cfg.save_total_limit,
        "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False,
        "prediction_loss_only": True, "remove_unused_columns": False,
        "dataloader_num_workers": 0, "torch_empty_cache_steps": eval_steps,
        "bf16": precision == "bf16", "fp16": precision == "fp16",
        "report_to": cfg.report_to,
    }


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    return {k: torch.tensor(v) for k, v in pad_batch(batch, pad_id).items()}


def build_trainer(model, tok, args: dict, train: list[dict], val: list[dict], callbacks: list):
    from datasets import Dataset
    from transformers import Trainer
    from transformers import TrainingArguments

    return Trainer(
        model=model, args=TrainingArguments(**args), processing_class=tok,
        train_dataset=Dataset.from_list(model_columns(train)),
        eval_dataset=Dataset.from_list(model_columns(val)),
        data_collator=partial(collate, pad_id=tok.pad_token_id), callbacks=callbacks,
    )


def prepare(cfg: SftConfig, data_dir: Path, tok) -> tuple[list[dict], list[dict], dict]:
    train = load_split(data_dir / "train_sft.jsonl", tok)
    val = load_split(data_dir / "val_sft.jsonl", tok)
    report = length_report(train, cfg.max_seq_len)
    log.info("train token lengths: %s", report)
    train = drop_too_long(train, cfg.max_seq_len)
    val = drop_too_long(val, cfg.max_seq_len)
    if not val:
        raise ValueError("val split is empty; raise val_frac in configs/default.yaml "
                         "and rerun `echolm format`")
    return train, val, report


def run_info(cfg, data_dir: Path, report: dict, trainer, result, peak_gb: float, evals: list) -> dict:
    return {
        "config": asdict(cfg), "git_sha": git_sha(),
        "data": {"train_hash": file_hash(data_dir / "train_sft.jsonl"),
                 "val_hash": file_hash(data_dir / "val_sft.jsonl"), **report},
        "baseline_eval_loss": evals[0][1] if evals else None,
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "steps": trainer.state.global_step,
        "train_runtime_s": round(result.metrics["train_runtime"]),
        "peak_vram_gb": round(peak_gb, 2),
        "eval_history": evals,
    }


def run_dir(cfg: SftConfig, out_root: Path, resume: Path | None, max_steps: int) -> Path:
    if resume:
        return resume
    tag = "smoke" if max_steps > 0 else f"r{cfg.lora_r}"
    return out_root / f"sft-{tag}-{time.strftime('%Y%m%d-%H%M')}"


def train_sft(cfg: SftConfig, data_dir: Path, out_root: Path, max_steps: int = -1,
              resume: Path | None = None) -> Path:
    out_dir = run_dir(cfg, out_root, resume, max_steps)
    handler = log_to_file(out_dir / "train.log")
    try:
        with keep_awake():
            fit(cfg, data_dir, out_dir, max_steps, resume is not None)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return out_dir


def fit(cfg: SftConfig, data_dir: Path, out_dir: Path, max_steps: int, resume: bool) -> None:
    if cfg.report_to == "wandb":
        setup_wandb(cfg.wandb_project)
    model, tok, fast = load_model(cfg)
    import torch

    from echolm.train.callbacks import EvalLogCallback
    from echolm.train.callbacks import SampleCallback
    from echolm.train.callbacks import VramCallback
    from echolm.train.callbacks import pick_samples

    train, val, report = prepare(cfg, data_dir, tok)
    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    samples = SampleCallback(tok, pick_samples(val, cfg.n_samples, cfg.seed), cfg.sample_max_new_tokens,
                             cfg.report_to == "wandb" and cfg.wandb_samples,
                             fast.for_inference, fast.for_training)
    vram = VramCallback()
    args = train_args(cfg, out_dir, len(train), precision, max_steps)
    evals = EvalLogCallback()
    trainer = build_trainer(model, tok, args, train, val, [evals, samples, vram])
    result = trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    info = run_info(cfg, data_dir, report, trainer, result, vram.peak_gb, evals.history)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("best eval loss %s at %s", info["best_eval_loss"], info["best_checkpoint"])
    log.info("adapter saved to %s", out_dir / "adapter")
