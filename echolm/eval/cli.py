from pathlib import Path

import click

DATA = click.option("--data", type=click.Path(exists=True, file_okay=False, path_type=Path),
                    default=Path("data/processed"))


@click.group("eval", help="Compare models on the held-out test split.")
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


@eval_group.command("plot", help="Draw the SFT val-loss and GRPO val-reward curves as SVG files.")
@click.option("--outputs", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=Path("outputs"), show_default=True, help="folder holding sft/ and grpo/ runs")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), default=Path("docs/figures"),
              show_default=True)
def eval_plot_cmd(outputs: Path, out: Path) -> None:
    from echolm.eval.plots import write_plots
    from echolm.train.select import newest_run

    try:
        sft = newest_run(outputs / "sft")
        grpo = newest_run(outputs / "grpo", "grpo-run-*") if (outputs / "grpo").exists() else None
        paths = write_plots(sft, grpo, out)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo("\n".join(str(p) for p in paths))
