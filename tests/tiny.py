# a tiny random Qwen2 model, tokenizer and LoRA adapter on disk, so the training and eval code can
# run end to end on CPU in seconds; the numbers are meaningless, the plumbing is what's tested
from pathlib import Path

CHAT_TEMPLATE = (
    "{%- for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content']"
    " + '<|im_end|>' + '\\n'}}{%- endfor %}{%- if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}"
    "{%- endif %}"
)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DATA = Path(__file__).parents[1] / "examples" / "synthetic" / "processed"


def build_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers import decoders
    from tokenizers import models
    from tokenizers import pre_tokenizers
    from tokenizers import trainers
    from transformers import PreTrainedTokenizerFast

    files = sorted(DATA.glob("*.jsonl"))
    texts = [line for f in files for line in f.read_text(encoding="utf-8").splitlines()]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = CHAT_TEMPLATE
    return tok


def build_model(vocab: int, seed: int = 0):
    import torch
    from transformers import Qwen2Config
    from transformers import Qwen2ForCausalLM

    torch.manual_seed(seed)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                        eos_token_id=2, pad_token_id=0))


def write_base_and_adapter(root: Path) -> tuple[Path, Path]:
    from peft import LoraConfig
    from peft import get_peft_model

    tok = build_tokenizer()
    base, adapter = root / "base", root / "adapter"
    model = build_model(len(tok))
    model.save_pretrained(base)
    tok.save_pretrained(base)
    lora = get_peft_model(build_model(len(tok)), LoraConfig(r=4, lora_alpha=4, target_modules=TARGETS,
                                                             task_type="CAUSAL_LM"))
    lora.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    return base, adapter


def fake_unsloth(base: Path):
    # stands in for `unsloth` on machines without a GPU: same FastLanguageModel surface, plain HF inside
    import os
    import types

    from peft import LoraConfig
    from peft import PeftModel
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    class Fast:
        @staticmethod
        def from_pretrained(model_name, max_seq_length, load_in_4bit, dtype):
            tok = AutoTokenizer.from_pretrained(base)
            name = str(model_name)
            if os.path.exists(os.path.join(name, "adapter_config.json")):
                return PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), name), tok
            return AutoModelForCausalLM.from_pretrained(name if os.path.isdir(name) else base), tok

        @staticmethod
        def get_peft_model(model, r, lora_alpha, lora_dropout, target_modules, bias, **kwargs):
            cfg = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             target_modules=target_modules, bias=bias, task_type="CAUSAL_LM")
            return get_peft_model(model, cfg)

        for_inference = staticmethod(lambda m: m.eval())
        for_training = staticmethod(lambda m: m.train())

    return types.SimpleNamespace(FastLanguageModel=Fast)
