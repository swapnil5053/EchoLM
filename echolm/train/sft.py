import json
import logging
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from echolm.train.config import SftConfig
from echolm.train.data import drop_too_long
from echolm.train.data import file_hash
from echolm.train.data import length_report
from echolm.train.data import load_split
from echolm.train.data import token_lengths

log = logging.getLogger(__name__)

# torch / transformers / trl are imported inside functions: unsloth has to be imported before
# them so its patches apply, and `echolm parse` / `format` shouldn't pay for a torch import


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
    return model, tok, FastLanguageModel


def sft_args(cfg: SftConfig, out_dir: Path, run_name: str, precision: str) -> dict:
    return {
        "output_dir": str(out_dir), "run_name": run_name, "seed": cfg.seed,
        "num_train_epochs": cfg.epochs, "learning_rate": cfg.lr,
        "per_device_train_batch_size": cfg.batch_size, "per_device_eval_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "lr_scheduler_type": "cosine", "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay, "optim": "adamw_8bit",
        "logging_steps": cfg.log_steps,
        "eval_strategy": "steps", "eval_steps": cfg.eval_steps,
        "save_strategy": "steps", "save_steps": cfg.eval_steps, "save_total_limit": cfg.save_total_limit,
        "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False,
        "max_length": cfg.max_seq_len, "completion_only_loss": True, "packing": False,
        "bf16": precision == "bf16", "fp16": precision == "fp16",
        "report_to": cfg.report_to, "dataset_num_proc": 1,
    }


def build_trainer(model, tok, args: dict, train: list[dict], val: list[dict], callbacks: list):
    from datasets import Dataset
    from trl import SFTConfig
    from trl import SFTTrainer

    def to_ds(rows):
        return Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])

    return SFTTrainer(model=model, processing_class=tok, args=SFTConfig(**args),
                      train_dataset=to_ds(train), eval_dataset=to_ds(val), callbacks=callbacks)


def prepare(cfg: SftConfig, data_dir: Path, tok) -> tuple[list[dict], list[dict], dict]:
    train = load_split(data_dir / "train_sft.jsonl", tok)
    val = load_split(data_dir / "val_sft.jsonl", tok)
    lengths = token_lengths(train, tok)
    report = length_report(lengths, cfg.max_seq_len)
    log.info("train token lengths: %s", report)
    train = drop_too_long(train, lengths, cfg.max_seq_len)
    val = drop_too_long(val, token_lengths(val, tok), cfg.max_seq_len)
    return train, val, report


def git_sha() -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    except FileNotFoundError:
        log.warning("git not found, run_info.json will have no commit sha")
        return None
    if res.returncode != 0:
        log.warning("not inside a git checkout, run_info.json will have no commit sha")
        return None
    return res.stdout.strip()


def run_info(cfg, data_dir: Path, report: dict, trainer, result, peak_gb: float) -> dict:
    return {
        "config": asdict(cfg), "git_sha": git_sha(),
        "data": {"train_hash": file_hash(data_dir / "train_sft.jsonl"),
                 "val_hash": file_hash(data_dir / "val_sft.jsonl"), **report},
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_runtime_s": round(result.metrics["train_runtime"]),
        "peak_vram_gb": round(peak_gb, 2),
    }


def train_sft(cfg: SftConfig, data_dir: Path, out_root: Path) -> Path:
    run_name = f"sft-r{cfg.lora_r}-{time.strftime('%Y%m%d-%H%M')}"
    out_dir = out_root / run_name
    if cfg.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    model, tok, fast = load_model(cfg)
    import torch

    from echolm.train.callbacks import SampleCallback
    from echolm.train.callbacks import VramCallback
    from echolm.train.callbacks import pick_samples

    train, val, report = prepare(cfg, data_dir, tok)
    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    samples = SampleCallback(tok, pick_samples(val, cfg.n_samples, cfg.seed), cfg.sample_max_new_tokens,
                             cfg.report_to == "wandb", fast.for_inference, fast.for_training)
    vram = VramCallback()
    trainer = build_trainer(model, tok, sft_args(cfg, out_dir, run_name, precision), train, val,
                            [samples, vram])
    log.info("baseline eval before training: %s", trainer.evaluate())
    result = trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    info = run_info(cfg, data_dir, report, trainer, result, vram.peak_gb)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("best eval loss %s, adapter saved to %s", info["best_eval_loss"], out_dir / "adapter")
    return out_dir
