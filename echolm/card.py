import json
import logging
from pathlib import Path

from echolm.eval.report import build_report

log = logging.getLogger(__name__)

FRONT = """---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags: [lora, sft, grpo, style-transfer, hinglish, unsloth]
---
"""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def training_section(grpo_info: dict, stats: dict) -> list[str]:
    cfg = grpo_info.get("config", {})
    best = grpo_info.get("best_val", {})
    return [
        "## Training",
        "",
        f"- Data: {stats.get('windows', '?')} reply windows from 1:1 chats "
        f"({stats.get('splits', {})}), split by time so test replies are later than all training data.",
        f"- SFT: LoRA on all attention and MLP projections, loss on the reply tokens only; the starting "
        f"checkpoint was `{Path(grpo_info.get('init_adapter', '?')).name}`, the earliest within 3% of the "
        "best val loss.",
        f"- GRPO: {grpo_info.get('steps', '?')} steps, {cfg.get('prompts_per_step', '?')} prompts x "
        f"{cfg.get('num_generations', '?')} samples per step, lr {cfg.get('lr', '?')}, no KL term; "
        f"reward weights {cfg.get('weights', {})}.",
        f"- Selected GRPO step: {grpo_info.get('best_step', '?')} (val reward {best.get('reward', '?')}; "
        "step 0 is the SFT model, so GRPO is only kept if it beats it).",
        f"- Hardware: one RTX 4060 Laptop GPU (8 GB), peak {grpo_info.get('peak_vram_gb', '?')} GB, "
        f"{round(grpo_info.get('runtime_s', 0) / 60)} min.",
        "",
    ]


LIMITS = """## Intended use and limits

- A personal style model: it imitates how one person texts one other person. It knows nothing about
  either person beyond the style and topics in the chat, and it will state things that are not true.
- Trained on a private conversation that includes another person's messages. Keep these weights and
  the generations private; do not upload them. The public repository only ships a synthetic dataset.
- 1.5B parameters: short, stylistically faithful replies, weak multi-turn reasoning.
"""


def build_card(grpo_run: Path, eval_root: Path, data_dir: Path) -> str:
    info = read_json(grpo_run / "run_info.json")
    stats = read_json(data_dir / "stats.json")
    parts = [FRONT, "# EchoLM personal style adapter", "",
             "LoRA adapter for Qwen2.5-1.5B-Instruct trained with SFT and then GRPO to reply the way one "
             "person texts (romanized Hindi-English code-switching included).", ""]
    parts += training_section(info, stats)
    if (eval_root).exists() and any(eval_root.glob("*/metrics.json")):
        parts += ["## Evaluation", "", build_report(eval_root)]
    parts.append(LIMITS)
    return "\n".join(parts)


def write_card(grpo_run: Path, eval_root: Path, data_dir: Path) -> Path:
    path = grpo_run / "adapter" / "README.md"
    path.write_text(build_card(grpo_run, eval_root, data_dir), encoding="utf-8")
    log.info("model card written to %s", path)
    return path
