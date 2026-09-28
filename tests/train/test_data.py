import json

import pytest

from echolm.train.data import IGNORE
from echolm.train.data import drop_too_long
from echolm.train.data import file_hash
from echolm.train.data import length_report
from echolm.train.data import load_split
from echolm.train.data import model_columns
from echolm.train.data import pad_batch
from echolm.train.data import render
from echolm.train.data import tokenize

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


def test_tokenize_masks_prompt_and_keeps_reply(tok):
    out = tokenize({"prompt": "a b c", "completion": "d e"}, tok)
    assert out["input_ids"] == ["a", "b", "c", "d", "e"]
    assert out["labels"] == [IGNORE, IGNORE, IGNORE, "d", "e"]
    assert out["attention_mask"] == [1] * 5


def test_load_split(tmp_path, tok):
    p = tmp_path / "train_sft.jsonl"
    p.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    row = load_split(p, tok)[0]
    assert row["completion"].startswith("kuch nahi")
    assert row["labels"][-1] == "bas<|im_end|>"
    assert IGNORE in row["labels"]


def test_drop_too_long_keeps_order():
    rows = [{"input_ids": [0] * n} for n in (5, 50, 7, 51)]
    assert [len(r["input_ids"]) for r in drop_too_long(rows, 10)] == [5, 7]


def test_length_report():
    rep = length_report([{"input_ids": [0] * n} for n in (3, 1, 2, 40)], 10)
    assert rep["tokens_max"] == 40
    assert rep["over_max_seq_len"] == 1
    assert rep["tokens_median"] == 2.5


def test_model_columns_drops_text_fields():
    row = {"id": "x", "prompt": "p", "completion": "c",
           "input_ids": [1], "attention_mask": [1], "labels": [1]}
    assert model_columns([row]) == [{"input_ids": [1], "attention_mask": [1], "labels": [1]}]


def test_pad_batch_pads_to_multiple_of_eight():
    batch = [{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [IGNORE, 6, 7]},
             {"input_ids": [5], "attention_mask": [1], "labels": [5]}]
    out = pad_batch(batch, pad_id=0)
    assert [len(x) for x in out["input_ids"]] == [8, 8]
    assert out["input_ids"][1] == [5] + [0] * 7
    assert out["attention_mask"][1] == [1] + [0] * 7
    assert out["labels"][0] == [IGNORE, 6, 7] + [IGNORE] * 5


def test_file_hash_changes_with_content(tmp_path):
    p = tmp_path / "x"
    p.write_text("a", encoding="utf-8")
    h = file_hash(p)
    p.write_text("b", encoding="utf-8")
    assert file_hash(p) != h and len(h) == 16
