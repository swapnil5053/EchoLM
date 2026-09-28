import json

import pytest

from echolm.demo.models import FALLBACK_SYSTEM
from echolm.demo.models import bubbles
from echolm.demo.models import find_adapters
from echolm.demo.models import system_prompt
from echolm.demo.models import to_messages
from tests.tiny import DATA


def test_system_prompt_from_data_or_fallback(tmp_path):
    assert system_prompt(DATA).startswith("You are Kabir.")
    assert system_prompt(tmp_path) == FALLBACK_SYSTEM


def test_to_messages_puts_the_other_person_in_the_user_role():
    history = [{"role": "user", "content": "kya kar raha"}, {"role": "assistant", "content": "kuch nahi"}]
    msgs = to_messages("sys", history, "chal")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "chal"


def test_bubbles_split_bursts():
    assert bubbles("haan\n\nkal milte ") == ["haan", "kal milte"]
    assert bubbles("   ") == ["…"]


def test_find_adapters(tmp_path):
    run = tmp_path / "sft" / "sft-r16-a"
    (run / "checkpoint-20").mkdir(parents=True)
    (run / "run_info.json").write_text(json.dumps({"eval_history": [[0, 5.0], [20, 4.0]]}), encoding="utf-8")
    (tmp_path / "grpo" / "grpo-run-x" / "adapter").mkdir(parents=True)
    (tmp_path / "grpo" / "grpo-smoke-y" / "adapter").mkdir(parents=True)
    found = find_adapters(tmp_path)
    assert found["sft"].name == "checkpoint-20"
    assert found["grpo"].parent.name == "grpo-run-x"
    assert find_adapters(tmp_path / "none") == {}


def test_model_bank_switches_adapters(tmp_path):
    pytest.importorskip("peft")
    from echolm.demo.models import ModelBank
    from tests.tiny import write_base_and_adapter

    base, adapter = write_base_and_adapter(tmp_path)
    bank = ModelBank(str(base), {"sft": adapter, "grpo": adapter})
    assert bank.names == ["base", "sft", "grpo"]
    msgs = to_messages("You are Kabir.", [], "kya kar raha hai")
    for name in bank.names:
        assert isinstance(bank.reply(name, msgs, 0.8, max_new_tokens=5), str)
    assert isinstance(bank.reply("base", msgs, 0.0, max_new_tokens=5), str)
