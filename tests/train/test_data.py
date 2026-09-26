import json

import pytest

from echolm.train.data import drop_too_long
from echolm.train.data import file_hash
from echolm.train.data import length_report
from echolm.train.data import load_split
from echolm.train.data import render
from echolm.train.data import token_lengths

ROW = {
    "id": "w1",
    "prompt": [{"role": "system", "content": "You are Kabir."}, {"role": "user", "content": "kya kar raha"}],
    "completion": [{"role": "assistant", "content": "kuch nahi\nbas"}],
}


def test_render_splits_at_generation_prompt(tok):
    out = render(ROW, tok)
    assert out["prompt"].endswith("<|im_start|>assistant\n")
    assert out["completion"] == "kuch nahi\nbas<|im_end|>"
    assert out["id"] == "w1"


def test_render_rejects_mismatched_template(tok):
    class Odd(type(tok)):
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return "X" if add_generation_prompt else "Y"

    with pytest.raises(ValueError, match="row w1"):
        render(ROW, Odd())


def test_load_split(tmp_path, tok):
    p = tmp_path / "train_sft.jsonl"
    p.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    assert load_split(p, tok)[0]["completion"].startswith("kuch nahi")


def test_drop_too_long_keeps_order():
    rows = [{"id": i} for i in range(4)]
    assert drop_too_long(rows, [5, 50, 7, 51], 10) == [{"id": 0}, {"id": 2}]


def test_length_report():
    rep = length_report([3, 1, 2, 40], 10)
    assert rep["tokens_max"] == 40
    assert rep["over_max_seq_len"] == 1
    assert rep["tokens_median"] == 2.5


def test_token_lengths(tok):
    assert token_lengths([{"prompt": "a b", "completion": " c"}], tok) == [3]


def test_file_hash_changes_with_content(tmp_path):
    p = tmp_path / "x"
    p.write_text("a", encoding="utf-8")
    h = file_hash(p)
    p.write_text("b", encoding="utf-8")
    assert file_hash(p) != h and len(h) == 16
