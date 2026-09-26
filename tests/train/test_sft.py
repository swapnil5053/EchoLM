from pathlib import Path

from echolm.train.config import SftConfig
from echolm.train.sft import git_sha
from echolm.train.sft import sft_args


def test_sft_args_precision_and_best_checkpoint():
    args = sft_args(SftConfig(), Path("out"), "run", "bf16")
    assert args["bf16"] and not args["fp16"]
    assert args["save_steps"] == args["eval_steps"]
    assert args["load_best_model_at_end"] and args["metric_for_best_model"] == "eval_loss"
    assert args["completion_only_loss"]


def test_sft_args_fp32_disables_mixed_precision():
    args = sft_args(SftConfig(), Path("out"), "run", "fp32")
    assert not args["bf16"] and not args["fp16"]


def test_git_sha_is_short_or_none():
    sha = git_sha()
    assert sha is None or 4 <= len(sha) <= 12
