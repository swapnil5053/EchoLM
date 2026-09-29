import inspect
import logging

from echolm.demo.guess import render_context
from echolm.demo.models import bubbles
from echolm.demo.models import to_messages

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


def guess_round(game):
    def fn(model: str):
        rnd = game.new_round(model)
        return render_context(rnd.context), rnd.options[0], rnd.options[1], rnd, ""

    return fn


def guess_answer(game, picked: int):
    def fn(rnd, tally: list[int]):
        done = f"{tally[0]} / {tally[1]} spotted"
        if rnd is None:
            return "press **new round** for the next pair", tally, done, None
        correct = game.answer(rnd, picked)
        tally = [tally[0] + correct, tally[1] + 1]
        verdict = "Right" if correct else "Wrong"
        text = f"**{verdict}.** The real reply was **{'AB'[rnd.real]}**."
        # the round is cleared so a second click cannot log the same pair twice
        score = f"{tally[0]} / {tally[1]} spotted ({tally[0] / tally[1]:.0%}; 50% = can't tell)"
        return text, tally, score, None

    return fn


def guess_tab(gr, game) -> None:
    gr.Markdown("One reply is what the person really wrote, the other is the model's (sampled for the "
                "held-out test set). Pick the real one. Answers are saved to the eval folder and "
                "`echolm eval report` adds the score to the table.")
    with gr.Row():
        model = gr.Dropdown(choices=[(LABELS.get(m, m), m) for m in game.models], value=game.models[-1],
                            label="model")
        new = gr.Button("new round", variant="primary")
    context = gr.Markdown()
    with gr.Row():
        a, b = gr.Textbox(label="A", lines=3), gr.Textbox(label="B", lines=3)
    with gr.Row():
        pick_a, pick_b = gr.Button("A is real"), gr.Button("B is real")
    result, score = gr.Markdown(), gr.Markdown("0 / 0")
    rnd, tally = gr.State(None), gr.State([0, 0])
    new.click(guess_round(game), [model], [context, a, b, rnd, result], api_name="guess_round")
    pick_a.click(guess_answer(game, 0), [rnd, tally], [result, tally, score, rnd], api_name=False)
    pick_b.click(guess_answer(game, 1), [rnd, tally], [result, tally, score, rnd], api_name=False)


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


def build_app(bank, system: str, game=None):
    import gradio as gr

    with gr.Blocks(title="EchoLM", analytics_enabled=False) as app:
        gr.Markdown("# EchoLM\nYou play the other person in the chat; the model replies the way "
                    "its owner texts.")
        if bank is not None:
            model_tabs(gr, bank, system)
        if game is not None and game.models:
            with gr.Tab("Real or model?"):
                guess_tab(gr, game)
    return app
