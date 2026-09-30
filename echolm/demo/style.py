import html
import re
from pathlib import Path

CSS = Path(__file__).with_name("style.css").read_text(encoding="utf-8")

MASTHEAD = ("<div class='masthead'><h1>EchoLM</h1><p>A 1.5B model fine-tuned to reply the way one "
            "person writes. Chat with it, compare it with the untrained model, or try to spot the real "
            "reply.</p></div>")


def theme(gr):
    return gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_sm,
        text_size=gr.themes.sizes.text_lg,
    )


def persona(system: str) -> str:
    m = re.match(r"You are ([^.]+)\.", system)
    return m.group(1) if m else "reply"


def transcript(context: list[dict], name: str = "reply") -> str:
    if not context:
        return "<div class='empty'>No earlier messages; this reply opened the conversation.</div>"
    who = html.escape(name)
    rows = [bubble("them", "them", m["content"]) if m["role"] == "user" else bubble("me", who, m["content"])
            for m in context[-6:]]
    return f"<div class='transcript'>{''.join(rows)}</div><p class='ask'>Which reply did {who} send?</p>"


def bubble(side: str, who: str, text: str) -> str:
    return f"<div class='msg {side}'><span class='who'>{who}</span><p>{html.escape(text)}</p></div>"


def scoreboard(tally: list[int], verdict: str = "", good: bool = True) -> str:
    right, total = tally
    share = f"{right / total:.0%} spotted" if total else "no answers yet"
    tone = "good" if good else "bad"
    note = f"<span class='verdict {tone}'>{html.escape(verdict)}</span>" if verdict else ""
    return (f"<div class='score'><span class='num'>{right}<small>/{total}</small></span>"
            f"<span class='label'>{share} · 50% means people cannot tell</span>{note}</div>")
