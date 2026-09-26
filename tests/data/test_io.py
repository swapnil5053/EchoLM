import pytest

from echolm.data.io import read_jsonl
from echolm.data.io import write_jsonl


def test_roundtrip_keeps_unicode(tmp_path):
    rows = [{"text": "bhai 😂"}, {"text": "acha"}]
    path = tmp_path / "sub" / "x.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    assert "😂" in path.read_text(encoding="utf-8")


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        read_jsonl(path)
