from echolm.data.models import DELETED
from echolm.data.models import FORWARDED
from echolm.data.models import MEDIA
from echolm.data.models import Msg
from echolm.data.models import Turn
from echolm.data.models import Window


def test_msg_roundtrip(mk):
    m = mk(3, True, "acha", reply_to="7")
    assert Msg.from_dict(m.to_dict()) == m


def test_turn_render_and_own_text(mk):
    turn = Turn(True, [mk(0, True, "dekh"), mk(1, True, "", kind=MEDIA),
                       mk(2, True, "news", kind=FORWARDED), mk(3, True, "x", kind=DELETED)])
    assert turn.render() == "dekh\n[media]\n[forwarded] news\n[deleted]"
    assert turn.own_text() == "dekh"


def test_window_roundtrip():
    w = Window("abc", "c1", 0, "2025-01-01T10:00:00", [{"role": "them", "text": "hi"}], "yo")
    assert Window.from_dict(w.to_dict()) == w
