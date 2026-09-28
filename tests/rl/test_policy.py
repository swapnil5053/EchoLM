import pytest

pytest.importorskip("peft")

from echolm.rl.policy import load_adapter_weights  # noqa: E402
from tests.tiny import TARGETS  # noqa: E402
from tests.tiny import build_model  # noqa: E402
from tests.tiny import write_base_and_adapter  # noqa: E402


def test_load_adapter_weights_copies_the_sft_lora(tmp_path):
    import torch
    from peft import LoraConfig
    from peft import get_peft_model
    from safetensors.torch import load_file

    _, adapter = write_base_and_adapter(tmp_path)
    model = get_peft_model(build_model(600, seed=3), LoraConfig(r=4, lora_alpha=4, target_modules=TARGETS,
                                                                  task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p)
    load_adapter_weights(model, adapter)
    saved = load_file(str(adapter / "adapter_model.safetensors"))
    key = next(k for k in saved if "q_proj.lora_B" in k)
    live = dict(model.named_parameters())[key.replace(".weight", ".default.weight")]
    assert torch.equal(live.detach(), saved[key])


def test_load_adapter_weights_rejects_a_mismatched_adapter(tmp_path):
    from peft import LoraConfig
    from peft import get_peft_model

    _, adapter = write_base_and_adapter(tmp_path)
    model = get_peft_model(build_model(600), LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj"],
                                                        task_type="CAUSAL_LM"))
    with pytest.raises(ValueError, match="does not fit"):
        load_adapter_weights(model, adapter)
