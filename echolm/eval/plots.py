import json
import logging
import math
from pathlib import Path

from echolm.train.select import DEFAULT_TOL
from echolm.train.select import pick_step
from echolm.train.select import saved_steps

log = logging.getLogger(__name__)

W, H = 640, 300
LEFT, RIGHT, TOP, BOTTOM = 64, 24, 48, 44

STYLE = """<style>
  svg { --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --grid: #e4e3de; --line: #2a78d6; }
  @media (prefers-color-scheme: dark) {
    svg { --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --grid: #3a3a37; --line: #3987e5; }
  }
  text { font: 12px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--ink2); }
  .title { font-size: 14px; font-weight: 600; fill: var(--ink); }
  .note { fill: var(--ink); }
</style>"""


def ticks(lo: float, hi: float, most: int = 6) -> list[float]:
    """Round-number ticks inside [lo, hi]: the finest step of 1, 2, 2.5 or 5 x 10^k giving <= `most`."""
    span = (hi - lo) or 1.0
    power = 10 ** math.floor(math.log10(span / most))
    step = next(m * power for m in (1, 2, 2.5, 5, 10, 20) if span / (m * power) <= most)
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    return [round(i * step, 10) for i in range(first, last + 1)]


def scaler(lo: float, hi: float, a: float, b: float):
    span = (hi - lo) or 1.0
    return lambda v: a + (v - lo) / span * (b - a)


def padded(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.1 or abs(hi) * 0.1 or 1.0
    return lo - pad, hi + pad


def frame(title: str, xlabel: str, ylabel: str) -> list[str]:
    mid = (TOP + H - BOTTOM) / 2
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-label="{title}">', STYLE,
            f'<rect width="{W}" height="{H}" rx="8" fill="var(--surface)"/>',
            f'<text class="title" x="{LEFT}" y="26">{title}</text>',
            f'<text x="{(LEFT + W - RIGHT) / 2}" y="{H - 8}" text-anchor="middle">{xlabel}</text>',
            f'<text x="14" y="{mid}" text-anchor="middle" transform="rotate(-90 14 {mid})">{ylabel}</text>']


def axes(x, y, xs: list[float], ylo: float, yhi: float) -> list[str]:
    out = []
    for v in ticks(ylo, yhi):
        out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                   'stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{LEFT - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
    out += [f'<text x="{x(v):.1f}" y="{H - BOTTOM + 18}" text-anchor="middle">{v:g}</text>'
            for v in ticks(min(xs), max(xs))]
    return out


def series(x, y, points: list[tuple[float, float]], xlabel: str) -> list[str]:
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(a):.1f},{y(b):.1f}" for i, (a, b) in enumerate(points))
    out = [f'<path d="{path}" fill="none" stroke="var(--line)" stroke-width="2" '
           'stroke-linejoin="round" stroke-linecap="round"/>']
    out += [f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4" fill="var(--line)" stroke="var(--surface)" '
            f'stroke-width="2"><title>{xlabel} {a:g}: {b:.4g}</title></circle>' for a, b in points]
    return out


def reference(y, value: float, label: str) -> list[str]:
    return [f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y(value):.1f}" y2="{y(value):.1f}" '
            'stroke="var(--ink2)" stroke-width="1.5" stroke-dasharray="5 4"/>',
            f'<text x="{W - RIGHT}" y="{y(value) - 6:.1f}" text-anchor="end">{label}</text>']


def highlight(x, y, point: tuple[float, float], label: str) -> list[str]:
    a, b = point
    return [f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="8" fill="none" stroke="var(--ink)" '
            'stroke-width="2"/>',
            f'<text class="note" x="{min(x(a), W - RIGHT):.1f}" y="{y(b) - 15:.1f}" '
            f'text-anchor="{"end" if x(a) > W - 120 else "middle"}">{label}</text>']


def line_svg(points: list[tuple[float, float]], title: str, xlabel: str, ylabel: str,
             mark: tuple[float, str] | None = None, ref: tuple[float, str] | None = None) -> str:
    """Single-series line chart as a standalone SVG that follows the reader's light or dark theme.

    `mark` circles the point at that x with a note; `ref` draws a dashed horizontal line at that y.
    """
    xs = [p[0] for p in points]
    ylo, yhi = padded([p[1] for p in points] + ([ref[0]] if ref else []))
    x, y = scaler(min(xs), max(xs), LEFT, W - RIGHT), scaler(ylo, yhi, H - BOTTOM, TOP)
    out = frame(title, xlabel, ylabel) + axes(x, y, xs, ylo, yhi)
    if ref:
        out += reference(y, *ref)
    out += series(x, y, points, xlabel)
    if mark:
        out += highlight(x, y, next(p for p in points if p[0] == mark[0]), mark[1])
    return "\n".join(out + ["</svg>"]) + "\n"


def read_info(run: Path) -> dict:
    path = run / "run_info.json"
    if not path.exists():
        raise ValueError(f"no run_info.json in {run}; is it a finished run?")
    return json.loads(path.read_text(encoding="utf-8"))


def sft_chart(run: Path, tol: float = DEFAULT_TOL) -> str:
    history = read_info(run).get("eval_history") or []
    if len(history) < 2:
        raise ValueError(f"{run} has fewer than 2 evals, nothing to plot")
    step = pick_step(history, saved_steps(run), tol)
    return line_svg([(s, v) for s, v in history], "SFT: validation loss", "training step", "val loss",
                    mark=(step, f"selected: step {step}"))


def grpo_chart(run: Path) -> str:
    history = read_info(run).get("val_history") or []
    if len(history) < 2:
        raise ValueError(f"{run} has fewer than 2 validations, nothing to plot")
    points = [(h["step"], h["reward"]) for h in history]
    best = max(history, key=lambda h: h["reward"])["step"]
    return line_svg(points, "GRPO: validation reward", "GRPO step", "val reward",
                    mark=(best, f"best: step {best}"), ref=(points[0][1], "SFT (step 0)"))


def write_plots(sft_run: Path, grpo_run: Path | None, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    charts = {"sft_val_loss.svg": sft_chart(sft_run)}
    if grpo_run is not None:
        charts["grpo_val_reward.svg"] = grpo_chart(grpo_run)
    paths = []
    for name, svg in charts.items():
        path = out / name
        path.write_text(svg, encoding="utf-8")
        log.info("wrote %s", path)
        paths.append(path)
    return paths
