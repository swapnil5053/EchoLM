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
from echolm.irc.cli import irc
from echolm.parse.load import load_export

log = logging.getLogger("echolm")

CONFIG = click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
                      default=None, help="YAML data config (defaults to built-in values)")


@click.group()
@click.option("-v", "--verbose", is_flag=True)
def cli(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")


cli.add_command(irc)


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
    wins = split_windows(wins, cfg.val_frac, cfg.test_frac, cfg.max_chat_share, cfg.seed,
                         cfg.split_by)
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


@train.command("select", help="Print the SFT checkpoint to build on (earliest within tolerance of the best).")
@click.option("--run", "run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="SFT run folder (default: newest in --root)")
@click.option("--root", type=click.Path(file_okay=False, path_type=Path), default=Path("outputs/sft"),
              show_default=True)
@click.option("--tol", type=float, default=0.03, show_default=True)
def train_select_cmd(run_dir: Path | None, root: Path, tol: float) -> None:
    from echolm.train.select import newest_run
    from echolm.train.select import select_checkpoint

    try:
        path = select_checkpoint(run_dir or newest_run(root), tol)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(str(path))


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


@train.command("grpo", help="GRPO on top of the selected SFT checkpoint with the style rewards.")
@DATA
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), default=Path("outputs/grpo"))
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=Path("configs/grpo.yaml"), show_default=True)
@click.option("--init", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="adapter to start from (default: init_adapter in the config)")
@click.option("--backend", type=click.Choice(["unsloth", "hf"]), default=None,
              help="override the model backend from the config")
@click.option("--max-steps", type=int, default=-1, help="stop after this many steps (smoke test)")
def train_grpo_cmd(data: Path, out: Path, config_path: Path, init: Path | None, backend: str | None,
                   max_steps: int) -> None:
    from echolm.rl.config import load_grpo_config
    from echolm.rl.train import train_grpo

    cfg = load_grpo_config(config_path)
    if backend:
        cfg.backend = backend
    out_dir = train_grpo(cfg, data, out, max_steps, init)
    log.info("run saved to %s", out_dir)


@cli.group("eval", help="Compare models on the held-out test split.")
def eval_group() -> None:
    pass


@eval_group.command("run", help="Sample replies to the test prompts with one model and score them.")
@click.option("--model", "model_ref", required=True,
              help="'base' for the untrained model, or an adapter folder such as outputs/sft/<run>/adapter")
@click.option("--name", required=True, help="name for this model in the report, e.g. base or sft")
@DATA
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), default=Path("outputs/eval"))
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=Path("configs/eval.yaml"), show_default=True)
@click.option("--score-only", is_flag=True, help="re-score existing generations without the GPU")
def eval_run_cmd(model_ref: str, name: str, data: Path, out: Path, config_path: Path,
                 score_only: bool) -> None:
    from echolm.eval.config import load_eval_config
    from echolm.eval.run import generate
    from echolm.eval.run import score

    cfg = load_eval_config(config_path)
    run_dir = out / name
    if not score_only:
        generate(model_ref, data, run_dir, cfg)
    score(data, run_dir, cfg)


@eval_group.command("report", help="Write outputs/eval/report.md comparing every scored model.")
@click.option("--out", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=Path("outputs/eval"))
@click.option("--readme", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="also replace the results table between the markers in this README")
def eval_report_cmd(out: Path, readme: Path | None) -> None:
    from echolm.eval.report import write_report

    try:
        path = write_report(out, readme)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(path.read_text(encoding="utf-8"))


@cli.command(help="Write a model card (README.md) into a GRPO run's adapter folder.")
@click.option("--run", "run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="GRPO run folder (default: newest in --root)")
@click.option("--root", type=click.Path(file_okay=False, path_type=Path), default=Path("outputs/grpo"),
              show_default=True)
@click.option("--eval", "eval_root", type=click.Path(file_okay=False, path_type=Path),
              default=Path("outputs/eval"))
@DATA
@click.option("--irc", is_flag=True, help="the adapter was trained on the Ubuntu IRC benchmark")
def card(run_dir: Path | None, root: Path, eval_root: Path, data: Path, irc: bool) -> None:
    from echolm.card import write_card
    from echolm.train.select import newest_run

    try:
        run_dir = run_dir or newest_run(root, "grpo-run-*")
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(str(write_card(run_dir, eval_root, data, irc)))


@cli.command(help="Chat with the base, SFT and GRPO models side by side in the browser (local only).")
@click.option("--outputs", type=click.Path(file_okay=False, path_type=Path), default=Path("outputs"))
@DATA
@click.option("--sft", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="SFT adapter (default: the selected checkpoint of the newest SFT run)")
@click.option("--grpo", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="GRPO adapter (default: the newest GRPO run)")
@click.option("--port", type=int, default=7860, show_default=True)
def demo(outputs: Path, data: Path, sft: Path | None, grpo: Path | None, port: int) -> None:
    from echolm.demo.app import build_app
    from echolm.demo.models import ModelBank
    from echolm.demo.models import find_adapters
    from echolm.demo.models import system_prompt
    from echolm.eval.config import EvalConfig

    adapters = find_adapters(outputs)
    adapters.update({k: v for k, v in (("sft", sft), ("grpo", grpo)) if v})
    bank = ModelBank(EvalConfig().base_model, adapters)
    # bound to this machine only: the replies are generated from private chats
    build_app(bank, system_prompt(data)).launch(server_name="127.0.0.1", server_port=port, share=False,
                                                inbrowser=True)


if __name__ == "__main__":
    cli()
