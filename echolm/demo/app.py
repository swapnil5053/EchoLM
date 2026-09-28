import inspect
import logging

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


def build_app(bank, system: str):
    import gradio as gr

    choices = [(LABELS.get(n, n), n) for n in bank.names]
    with gr.Blocks(title="EchoLM", analytics_enabled=False) as app:
        gr.Markdown("# EchoLM\nYou play the other person in the chat; the model replies the way "
                    "its owner texts.")
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
    return app
