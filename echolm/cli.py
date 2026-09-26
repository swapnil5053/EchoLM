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
    msgs = clean(load_export(export, me, date_order), cfg.blocked_words)
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
