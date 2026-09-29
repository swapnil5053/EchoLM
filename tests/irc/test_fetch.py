import json
import sys
import types
from datetime import date

import pytest

from echolm.irc.fetch import doc_day
from echolm.irc.fetch import fetch
from echolm.irc.fetch import keep


def row(channel="#ubuntu", created="2019-05-04T00:00:00", text="[10:00] <a> b: hi"):
    return {"id": "x", "text": text, "created": created, "metadata": {"channel": channel}}


def test_doc_day_falls_back_to_the_url():
    assert doc_day(row()) == date(2019, 5, 4)
    r = {"id": "x", "created": None, "metadata": json.dumps({"url": "https://irclogs.ubuntu.com/2011/02/03/"})}
    assert doc_day(r) == date(2011, 2, 3)
    r["metadata"] = json.dumps({"url": "https://irclogs.ubuntu.com/"})
    assert doc_day(r) is None


def test_keep_filters_channel_and_year():
    assert keep(row(), "#ubuntu", 2019)["day"] == "2019-05-04"
    assert keep(row(channel="#ubuntu-fr"), "#ubuntu", 2019) is None
    assert keep(row(created="2012-01-01"), "#ubuntu", 2019) is None


@pytest.fixture
def fake_datasets(monkeypatch):
    rows = [row(), row(channel="#kubuntu"), row(created="2020-01-02")]
    mod = types.SimpleNamespace(load_dataset=lambda name, split, streaming: iter(rows))
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_fetch_writes_the_channel_slice(tmp_path, fake_datasets):
    out = tmp_path / "irc" / "logs.jsonl"
    assert fetch(out, "#ubuntu", 2019) == 2
    days = [json.loads(line)["day"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert days == ["2019-05-04", "2020-01-02"]
    assert not out.with_suffix(".part").exists()


def test_fetch_stops_at_max_docs(tmp_path, fake_datasets):
    assert fetch(tmp_path / "logs.jsonl", "#ubuntu", 2019, max_docs=1) == 1


def test_fetch_names_busy_channels_when_nothing_matches(tmp_path, fake_datasets):
    with pytest.raises(ValueError, match="#kubuntu"):
        fetch(tmp_path / "logs.jsonl", "#ubuntu-de", 2000)
