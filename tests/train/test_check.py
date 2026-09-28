import json

from echolm.train.check import check_data
from echolm.train.check import check_disk
from echolm.train.check import check_python
from echolm.train.check import check_wandb


def test_check_data_missing_and_present(tmp_path):
    ok, msg = check_data(tmp_path)
    assert not ok and "echolm format" in msg
    for name in ("train_sft.jsonl", "val_sft.jsonl"):
        (tmp_path / name).write_text(json.dumps({"id": "a"}) + "\n", encoding="utf-8")
    ok, msg = check_data(tmp_path)
    assert ok and "1 train / 1 val" in msg


def test_check_python_and_disk(tmp_path):
    assert check_python()[0]
    assert "GB free" in check_disk(tmp_path)[1]


def test_check_wandb_never_fails():
    assert check_wandb("none")[0]
    assert check_wandb("wandb")[0]
