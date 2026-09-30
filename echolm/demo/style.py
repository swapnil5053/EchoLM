import html
import re
from pathlib import Path

CSS = Path(__file__).with_name("style.css").read_text(encoding="utf-8")

MASTHEAD = ("<div class='masthead'><h1>EchoLM</h1><p>a 1.5B model trained to reply the way one "
            "person writes</p></div>")

# A / B pick a card and N deals the next pair, unless the cursor is in a text field
KEYS = """<script>
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || /INPUT|TEXTAREA/.test(document.activeElement.tagName)) return;
  const id = {a: "slip-a", b: "slip-b", n: "next-pair"}[e.key.toLowerCase()];
  const el = id && document.getElementById(id);
  if (el && !el.disabled) { e.preventDefault(); el.click(); }
});
</script>"""


def theme(gr):
    system = ["Segoe UI", "-apple-system", "system-ui", "sans-serif"]
    base = gr.themes.Base(primary_hue=gr.themes.colors.indigo, neutral_hue=gr.themes.colors.gray, font=system,
                          font_mono=["Cascadia Mono", "Consolas", "ui-monospace", "monospace"],
                          radius_size=gr.themes.sizes.radius_md, text_size=gr.themes.sizes.text_md)
    # the same values as style.css, so Gradio's own widgets match the custom parts
    return base.set(
        body_background_fill="#f3f4f6", body_background_fill_dark="#14161a",
        background_fill_primary="#ffffff", background_fill_primary_dark="#1c1f25",
        background_fill_secondary="#f3f4f6", background_fill_secondary_dark="#14161a",
        block_background_fill="#ffffff", block_background_fill_dark="#1c1f25",
        input_background_fill="#ffffff", input_background_fill_dark="#14161a",
        border_color_primary="#d9dce3", border_color_primary_dark="#30343c",
        color_accent="#2346c8", color_accent_soft="#eaeefc", color_accent_soft_dark="#1f2640",
        slider_color="#2346c8", slider_color_dark="#8fa3ff",
        body_text_color="#1c2233", body_text_color_dark="#e8e9ec",
        body_text_color_subdued="#5d6475", body_text_color_subdued_dark="#a3a9b5",
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
    """Hits over rounds with a bar against the 50% line, where a judge is only guessing."""
    right, total = tally
    share = right / total if total else 0.0
    label = f"{share:.0%} spotted, 50% is a coin flip" if total else "no answers yet"
    tone = "good" if good else "bad"
    note = f"<span class='verdict {tone}'>{html.escape(verdict)}</span>" if verdict else ""
    return (f"<div class='score'><span class='num'>{right}<small> / {total}</small></span>"
            f"<span class='bar'><i style='width:{share:.0%}'></i><b></b></span>"
            f"<span class='label'>{label}</span>{note}</div>")
