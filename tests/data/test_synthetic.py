from echolm.config import DataConfig
from echolm.data.synthetic import write_synthetic
from echolm.data.window import build_windows
from echolm.parse.load import load_export


def test_deterministic(tmp_path):
    write_synthetic(tmp_path / "a", 7)
    write_synthetic(tmp_path / "b", 7)
    for name in ("whatsapp_rohan.txt", "telegram_meera.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_yields_about_fifty_windows(tmp_path):
    write_synthetic(tmp_path, 7)
    msgs = load_export(tmp_path / "whatsapp_rohan.txt", "Kabir")
    msgs += load_export(tmp_path / "telegram_meera.json", "Kabir")
    wins = build_windows(msgs, DataConfig())
    assert 45 <= len(wins) <= 65
    assert any(w.quoted for w in wins)
    assert any(not w.context for w in wins)
