import pytest

from echolm.irc.validate import read_links
from echolm.irc.validate import validate

# line numbers below are 0-based positions in DAY (line 0 is the join message)
LINKS = """1 1 -
1 2 -
2 3 -
2 4 -
4 5 -
3 7 -
9 9 -
9 10 -
"""


def write_split(folder, text, links):
    folder.mkdir()
    (folder / "2024-03-01_09.raw.txt").write_text(text, encoding="utf-8")
    (folder / "2024-03-01_09.annotation.txt").write_text(links, encoding="utf-8")
    return folder


def test_read_links_orders_parent_first(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("5 3 -\n4 4 -\n", encoding="utf-8")
    assert read_links(p) == {5: {3}, 4: {4}}


def test_validate_against_labels(tmp_path, day_text):
    out = validate(write_split(tmp_path / "test", day_text, LINKS))
    assert out["precision"] == 1.0
    assert out["attached"] == 5
    assert out["reply_edges"] == 3
    assert out["recall"] == 1.0


def test_wrong_labels_lower_precision(tmp_path, day_text):
    out = validate(write_split(tmp_path / "test", day_text, "1 2 -\n1 10 -\n"), rules=("addressed",))
    assert out["precision_addressed"] == 0.5


def test_skip_and_missing_data(tmp_path, day_text):
    folder = write_split(tmp_path / "test", day_text, LINKS)
    with pytest.raises(ValueError, match="no \\*.raw.txt"):
        validate(folder, skip=("2024-03-01_09",))
