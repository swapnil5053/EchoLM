import json

import pytest

from echolm.data.export import OPENER
from echolm.data.export import export_all
from echolm.data.export import sft_row
from echolm.data.export import to_prompt
from echolm.data.models import Window

CTX = [{"role": "them", "text": "kya kar raha"}, {"role": "me", "text": "kuch nahi"},
       {"role": "them", "text": "chal"}]


def win(context=CTX, quoted=None, split="train", i=0):
    return Window(f"w{i}", "c1", i, "2025-01-01T10:00:00", list(context), "haan chal", quoted, split)


def test_prompt_maps_roles():
    roles = [m["role"] for m in to_prompt(win(), "sys")]
    assert roles == ["system", "user", "assistant", "user"]


def test_opener_gets_placeholder_or_quote():
    assert to_prompt(win([]), "sys")[-1]["content"] == OPENER
    assert to_prompt(win([], quoted="purana msg"), "sys")[-1]["content"] == "> purana msg"


def test_quote_is_prefixed_to_last_message():
    assert to_prompt(win(quoted="movie?"), "sys")[-1]["content"] == "> movie?\nchal"


def test_prompt_does_not_mutate_window():
    w = win(quoted="q")
    to_prompt(w, "sys")
    assert w.context == CTX


def test_sft_row_completion_is_target():
    assert sft_row(win(), "sys")["completion"] == [{"role": "assistant", "content": "haan chal"}]


def test_export_writes_all_files(tmp_path):
    wins = [win(i=0), win(i=1), win(split="val", i=2), win(split="test", i=3)]
    info = export_all(wins, tmp_path, "sys")
    assert info["splits"] == {"train": 2, "val": 1, "test": 1}
    test_rows = [json.loads(line) for line in (tmp_path / "test.jsonl").open(encoding="utf-8")]
    assert test_rows[0]["reference"] == "haan chal"
    names = {p.name for p in tmp_path.iterdir()}
    assert {"train_sft.jsonl", "val_sft.jsonl", "grpo.jsonl", "test.jsonl", "stats.json"} <= names


def test_export_without_train_raises(tmp_path):
    with pytest.raises(ValueError, match="train split is empty"):
        export_all([win(split="test")], tmp_path, "sys")
