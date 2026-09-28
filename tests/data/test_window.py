import pytest

from echolm.config import DataConfig
from echolm.data.models import MEDIA
from echolm.data.models import Turn
from echolm.data.window import build_context
from echolm.data.window import build_windows
from echolm.data.window import split_sessions
from echolm.data.window import to_turns


def test_split_sessions_on_gap(mk):
    msgs = [mk(0, False), mk(10, True), mk(50, False), mk(55, True)]
    assert [len(s) for s in split_sessions(msgs, 30)] == [2, 2]


def test_turns_merge_consecutive_sender(mk):
    turns = to_turns([mk(0, False), mk(1, True, "a"), mk(2, True, "b"), mk(3, False)])
    assert [t.is_me for t in turns] == [False, True, False]
    assert turns[1].own_text() == "a\nb"


def test_context_respects_turn_and_char_budget(mk):
    turns = [Turn(i % 2 == 1, [mk(i, i % 2 == 1, "x" * 10)]) for i in range(6)]
    assert len(build_context(turns, 3, 1000, 600)) == 3
    assert len(build_context(turns, 8, 25, 600)) == 2


def test_context_truncates_single_long_turn(mk):
    turns = [Turn(False, [mk(0, False, "a" * 50 + "END")])]
    ctx = build_context(turns, 8, 10, 600)
    assert ctx[0][1] == "aaaaaaaEND"


def test_long_context_turn_is_collapsed(mk):
    turns = [Turn(False, [mk(0, False, "paste " * 200)]), Turn(True, [mk(1, True, "ye kya hai")])]
    ctx = build_context(turns, 8, 1500, 600)
    assert [text for _, text in ctx] == ["[long message]", "ye kya hai"]


def test_windows_only_for_my_turns(mk):
    msgs = [mk(0, False, "kya kar raha"), mk(1, True, "kuch nahi"), mk(2, True, "tu bata"),
            mk(3, False, "chal"), mk(4, True, "haan")]
    wins = build_windows(msgs, DataConfig())
    assert [w.target for w in wins] == ["kuch nahi\ntu bata", "haan"]
    assert wins[1].context[-1] == {"role": "them", "text": "chal"}
    assert wins[0].id != wins[1].id


def test_opener_has_empty_context_and_can_be_disabled(mk):
    msgs = [mk(0, True, "bhai"), mk(1, False, "haan"), mk(2, True, "chal")]
    assert build_windows(msgs, DataConfig())[0].context == []
    assert len(build_windows(msgs, DataConfig(include_openers=False))) == 1


def test_media_only_reply_is_skipped(mk):
    msgs = [mk(0, False, "dekh"), mk(1, True, "", kind=MEDIA), mk(2, False, "?"), mk(3, True, "ok")]
    assert [w.target for w in build_windows(msgs, DataConfig())] == ["ok"]


def test_quote_outside_context_is_attached(mk):
    msgs = [mk(0, False, "movie chalein?", msg_id="1"), mk(1, True, "haan", msg_id="2"),
            mk(300, False, "kal?", msg_id="3"), mk(301, True, "wo movie", msg_id="4", reply_to="1")]
    wins = build_windows(msgs, DataConfig())
    assert wins[1].quoted == "movie chalein?"
    assert wins[0].quoted is None


def test_skip_reasons_are_counted_separately(mk, caplog):
    msgs = [mk(0, False, "dekh"), mk(1, True, "", kind=MEDIA), mk(2, False, "?"),
            mk(3, True, "x" * 700), mk(4, False, "??"), mk(5, True, "ok")]
    with caplog.at_level("INFO"):
        build_windows(msgs, DataConfig())
    assert "'empty_target': 1" in caplog.text
    assert "'target_too_long': 1" in caplog.text


def test_link_replies_are_skipped_unless_disabled(mk):
    msgs = [mk(0, False, "kya bhej"), mk(1, True, "Songspot guess the song <URL>"), mk(2, False, "?"),
            mk(3, True, "haan")]
    assert [w.target for w in build_windows(msgs, DataConfig())] == ["haan"]
    assert len(build_windows(msgs, DataConfig(drop_link_targets=False))) == 2


def test_no_windows_raises(mk):
    with pytest.raises(ValueError, match="no training windows"):
        build_windows([mk(0, False, "hi")], DataConfig())
