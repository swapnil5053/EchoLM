import inspect
import logging

from echolm.demo.models import bubbles
from echolm.demo.models import to_messages
from echolm.demo.style import CSS
from echolm.demo.style import KEYS
from echolm.demo.style import MASTHEAD
from echolm.demo.style import persona
from echolm.demo.style import scoreboard
from echolm.demo.style import theme
from echolm.demo.style import transcript

log = logging.getLogger(__name__)

LABELS = {"base": "Qwen2.5-1.5B (untrained)", "sft": "SFT", "grpo": "SFT + GRPO"}


def chatbot(gr, **kwargs):
    # gradio 5 needs type="messages"; gradio 6 only has the messages format and dropped the argument
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
        kwargs["type"] = "messages"
    return gr.Chatbot(**kwargs)


def respond(bank, system: str):
    def fn(message: str, history: list[dict], name: str, temperature: float):
        if not message.strip():
            return history, ""
        reply = bank.reply(name, to_messages(system, history, message), temperature)
        new = [{"role": "user", "content": message}]
        new += [{"role": "assistant", "content": b} for b in bubbles(reply)]
        return history + new, ""

    return fn


def compare(bank, system: str):
    def fn(message: str, temperature: float):
        msgs = to_messages(system, [], message)
        return ["\n".join(bubbles(bank.reply(n, msgs, temperature))) for n in bank.names]

    return fn


def slip(gr, text: str, mark: str | None = None):
    classes = ["slip", mark] if mark else ["slip"]
    return gr.update(value=text, elem_classes=classes, interactive=mark is None)


def guess_round(game, name: str):
    def fn(model: str):
        import gradio as gr

        rnd = game.new_round(model)
        return (transcript(rnd.context, name), slip(gr, rnd.options[0]), slip(gr, rnd.options[1]), rnd,
                scoreboard(game.totals(model)))

    return fn


def guess_answer(game, picked: int):
    def fn(rnd):
        import gradio as gr

        if rnd is None:
            return gr.update(), gr.update(), gr.update(), None
        correct = game.answer(rnd, picked)
        marks = ["real" if i == rnd.real else "model" for i in range(2)]
        verdict = f"{'Right' if correct else 'Not this time'}: {'AB'[rnd.real]} was the real reply"
        # the round is cleared so a second click cannot log the same pair twice
        slips = [slip(gr, text, mark) for text, mark in zip(rnd.options, marks, strict=True)]
        return *slips, scoreboard(game.totals(rnd.model), verdict, correct), None

    return fn


def guess_tab(gr, game, name: str) -> None:
    with gr.Row(equal_height=True):
        model = gr.Dropdown(choices=[(LABELS.get(m, m), m) for m in game.models], value=game.models[-1],
                            show_label=False, container=False, scale=4)
        new = gr.Button("Next pair", elem_id="next-pair", scale=1, min_width=150)
    start = "<div class='empty'>Press <kbd>N</kbd> or <b>Next pair</b> to see a conversation.</div>"
    context = gr.HTML(start)
    with gr.Row(equal_height=True):
        a = gr.Button("", elem_id="slip-a", elem_classes=["slip"], interactive=False)
        b = gr.Button("", elem_id="slip-b", elem_classes=["slip"], interactive=False)
    score = gr.HTML(scoreboard(game.totals(game.models[-1])))
    gr.Markdown("One reply is what the person sent, the other is the model's sample for the same message. "
                "Keys: <kbd>A</kbd> <kbd>B</kbd> to pick, <kbd>N</kbd> for the next pair. Answers are saved "
                "and `echolm eval report` adds the hit rate to the results.", elem_classes=["hint"])
    rnd = gr.State(None)
    new.click(guess_round(game, name), [model], [context, a, b, rnd, score], api_name="guess_round")
    a.click(guess_answer(game, 0), [rnd], [a, b, score, rnd], api_name=False)
    b.click(guess_answer(game, 1), [rnd], [a, b, score, rnd], api_name=False)
    model.change(lambda m: scoreboard(game.totals(m)), [model], [score], api_name=False)


def model_tabs(gr, bank, system: str) -> None:
    choices = [(LABELS.get(n, n), n) for n in bank.names]
    with gr.Tab("Chat"):
        with gr.Row():
            model = gr.Dropdown(choices=choices, value=bank.names[-1], label="model")
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="temperature")
        chat = chatbot(gr, height=460, label="chat")
        box = gr.Textbox(placeholder="type a message and press enter", show_label=False)
        box.submit(respond(bank, system), [box, chat, model, temperature], [chat, box], api_name="chat")
        gr.Button("clear").click(lambda: ([], ""), None, [chat, box], api_name=False)
    with gr.Tab("Compare"):
        prompt = gr.Textbox(label="message", placeholder="one message, every model answers it")
        temp2 = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="temperature")
        outs = [gr.Textbox(label=LABELS.get(n, n), lines=3) for n in bank.names]
        prompt.submit(compare(bank, system), [prompt, temp2], outs, api_name="compare")


def styling(gr) -> dict:
    return {"theme": theme(gr), "css": CSS, "head": KEYS}


def build_app(bank, system: str, game=None):
    import gradio as gr

    styled = styling(gr) if "css" in inspect.signature(gr.Blocks.__init__).parameters else {}
    with gr.Blocks(title="EchoLM", analytics_enabled=False, **styled) as app:
        gr.HTML(MASTHEAD)
        if game is not None and game.models:
            with gr.Tab("Real or model?"):
                guess_tab(gr, game, persona(system))
        if bank is not None:
            model_tabs(gr, bank, system)
    return app


def serve(app, port: int) -> None:
    import gradio as gr

    # gradio 6 moved theme and css from Blocks to launch()
    styled = styling(gr) if "css" in inspect.signature(gr.Blocks.launch).parameters else {}
    # bound to this machine only: the replies may come from private chats
    app.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=True, **styled)
