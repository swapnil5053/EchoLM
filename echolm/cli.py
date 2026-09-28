import logging
from collections import Counter
from pathlib import Path

import click

from echolm.config import load_config
from echolm.data.clean import clean
from echolm.data.export import export_all
from echolm.data.io import read_jsonl
from echolm.data.io import write_jsonl
from echolm.data.models import Msg
from echolm.data.split import split_windows
from echolm.data.synthetic import write_synthetic
from echolm.data.window import build_windows
from echolm.parse.load import load_export

log = logging.getLogger("echolm")

CONFIG = click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
                      default=None, help="YAML data config (defaults to built-in values)")


@click.group()
@click.option("-v", "--verbose", is_flag=True)
def cli(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")


@cli.command(help="Parse one WhatsApp .txt or Telegram .json export into cleaned messages.")
@click.argument("export", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--me", required=True, help="your name as it appears in the export, or Telegram user id")
@click.option("--date-order", type=click.Choice(["dmy", "mdy"]), default=None,
              help="WhatsApp only; auto-detected when omitted")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), default=Path("data/parsed"))
@CONFIG
def parse(export: Path, me: str, date_order: str | None, out: Path, config_path: Path | None) -> None:
    cfg = load_config(config_path)
    try:
        raw = load_export(export, me, date_order)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    msgs = clean(raw, cfg.blocked_words)
    write_jsonl(out / f"{export.stem}.jsonl", [m.to_dict() for m in msgs])


def load_parsed(parsed: Path) -> list[Msg]:
    files = sorted(parsed.glob("*.jsonl"))
    if not files:
        raise click.UsageError(f"no parsed .jsonl files in {parsed}; run `echolm parse` first")
    return [Msg.from_dict(row) for f in files for row in read_jsonl(f)]


@cli.command("format", help="Build context windows, split by time, write SFT / GRPO / test sets.")
@click.option("--parsed", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=Path("data/parsed"))
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              default=Path("data/processed"))
@click.option("--name", default=None, help="name used in the system prompt (default: your sender name)")
@CONFIG
def format_cmd(parsed: Path, out: Path, name: str | None, config_path: Path | None) -> None:
    cfg = load_config(config_path)
    msgs = load_parsed(parsed)
    name = name or Counter(m.sender for m in msgs if m.is_me).most_common(1)[0][0]
    wins = build_windows(msgs, cfg)
    wins = split_windows(wins, cfg.val_frac, cfg.test_frac, cfg.max_chat_share, cfg.seed)
    info = export_all(wins, out, cfg.system_prompt.replace("{name}", name))
    log.info("done: %s", info)


@cli.command(help="Regenerate the synthetic demo exports (fake WhatsApp + Telegram chats).")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              default=Path("examples/synthetic"))
@click.option("--seed", default=7, show_default=True)
def synth(out: Path, seed: int) -> None:
    write_synthetic(out, seed)


@cli.group(help="Training stages.")
def train() -> None:
    pass


SFT_CONFIG = click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
                          default=Path("configs/sft.yaml"), show_default=True)
DATA = click.option("--data", type=click.Path(exists=True, file_okay=False, path_type=Path),
                    default=Path("data/processed"))


@train.command("check", help="Check GPU, packages, data and disk before a long run.")
@DATA
@SFT_CONFIG
def train_check_cmd(data: Path, config_path: Path) -> None:
    from echolm.train.check import run_checks
    from echolm.train.config import load_sft_config

    if not run_checks(data, load_sft_config(config_path).report_to):
        raise click.ClickException("preflight failed, fix the FAIL lines above")


@train.command("sft", help="Supervised fine-tuning with LoRA on the formatted windows.")
@DATA
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), default=Path("outputs/sft"))
@SFT_CONFIG
@click.option("--max-steps", type=int, default=-1,
              help="stop after this many steps (a quick smoke test of the whole loop)")
@click.option("--resume", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="run folder to continue from its last checkpoint")
def train_sft_cmd(data: Path, out: Path, config_path: Path, max_steps: int, resume: Path | None) -> None:
    from echolm.train.config import load_sft_config
    from echolm.train.sft import train_sft

    out_dir = train_sft(load_sft_config(config_path), data, out, max_steps, resume)
    log.info("run saved to %s", out_dir)


if __name__ == "__main__":
    cli()
