import json
from pathlib import Path

import click

from echolm.irc.export import assigned_days
from echolm.irc.export import rank_users
from echolm.irc.export import write_export
from echolm.irc.threads import RULES

LOGS = click.option("--logs", type=click.Path(exists=True, dir_okay=False, path_type=Path),
                    default=Path("data/irc/logs.jsonl"), show_default=True)
GAP = click.option("--gap-min", type=float, default=1.0, show_default=True,
                   help="minutes an unprefixed line may follow and still count as the same exchange")
MIN_MSGS = click.option("--min-msgs", type=int, default=4, show_default=True,
                        help="drop 1:1 threads shorter than this")


@click.group(help="Public test data: rebuild one person's 1:1 chats from the Ubuntu IRC logs.")
def irc() -> None:
    pass


@irc.command(help="Stream common-pile/ubuntu_irc once and keep one channel's logs.")
@click.option("--channel", default="#ubuntu", show_default=True)
@click.option("--since", type=int, default=2016, show_default=True, help="first year to keep")
@click.option("--max-docs", type=int, default=None, help="stop after this many daily logs")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), default=Path("data/irc/logs.jsonl"))
def fetch(channel: str, since: int, max_docs: int | None, out: Path) -> None:
    from echolm.irc.fetch import fetch as run

    try:
        run(out, channel, since, max_docs)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@irc.command(help="List the nicks with the most 1:1 conversation.")
@LOGS
@GAP
@MIN_MSGS
@click.option("--top", type=int, default=15, show_default=True)
def users(logs: Path, gap_min: float, min_msgs: int, top: int) -> None:
    for row in rank_users(assigned_days(logs, gap_min), min_msgs)[:top]:
        click.echo(f"{row['nick']:<24} {row['replies']:>7} replies  {row['partners']:>6} partners")


@irc.command("export", help="Write one nick's threads as a Telegram-style export for `echolm parse`.")
@LOGS
@GAP
@MIN_MSGS
@click.option("--user", "nick", default=None, help="IRC nick to imitate (default: the most active one)")
@click.option("--alias", default="Alex", show_default=True, help="name used for that nick in the export")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path),
              default=Path("exports/ubuntu_irc.json"))
@click.option("--max-replies", type=int, default=2000, show_default=True,
              help="keep the most recent threads up to about this many of the user's lines (0 = all)")
def export_cmd(logs: Path, gap_min: float, min_msgs: int, nick: str | None, alias: str, out: Path,
               max_replies: int) -> None:
    try:
        write_export(logs, out, nick, alias, gap_min, min_msgs, max_replies or None)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@irc.command(help="Score thread rebuilding against the hand-labelled irc-disentanglement reply links.")
@click.option("--data", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="a split folder of github.com/jkkummerfeld/irc-disentanglement, e.g. data/test")
@GAP
@click.option("--rules", default=",".join(RULES), show_default=True)
@click.option("--skip", multiple=True, help="log name to leave out, e.g. 2008-07-14_18 (repeatable)")
def validate(data: Path, gap_min: float, rules: str, skip: tuple[str, ...]) -> None:
    from echolm.irc.validate import validate as run

    picked = tuple(r.strip() for r in rules.split(",") if r.strip())
    unknown = set(picked) - set(RULES)
    if unknown:
        raise click.BadParameter(f"unknown rules {sorted(unknown)}; choose from {RULES}")
    click.echo(json.dumps({k: round(v, 4) for k, v in run(data, picked, gap_min, skip).items()}, indent=2))
