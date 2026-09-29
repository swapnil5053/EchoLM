from echolm.data.models import Window
from echolm.data.split import cap_chat_share
from echolm.data.split import n_held
from echolm.data.split import session_splits
from echolm.data.split import split_windows


def win(i, chat="a", session=None, split="train"):
    return Window(f"{chat}{i}", chat, i if session is None else session, "", [], "t", split=split)


def test_n_held():
    assert n_held(2, 0.1) == 0
    assert n_held(3, 0.1) == 1
    assert n_held(40, 0.1) == 4
    assert n_held(40, 0.0) == 0


def test_latest_sessions_are_held_out():
    labels = session_splits(list(range(20)), 0.05, 0.10)
    assert [labels[s] for s in (19, 18)] == ["test", "test"]
    assert labels[17] == "val"
    assert labels[0] == "train"


def test_split_is_per_chat_and_never_inside_a_session():
    wins = [win(i, "a", session=i // 2) for i in range(20)]
    wins += [win(i, "b") for i in range(10)]
    out = split_windows(wins, 0.1, 0.1, None, 0)
    by_session = {}
    for w in out:
        by_session.setdefault((w.chat_id, w.session), set()).add(w.split)
    assert all(len(s) == 1 for s in by_session.values())
    assert {w.split for w in out if w.chat_id == "b"} == {"train", "val", "test"}


def test_cap_chat_share():
    wins = [win(i, "big") for i in range(30)] + [win(i, "small") for i in range(10)]
    out = cap_chat_share(wins, 0.5, 0)
    assert sum(w.chat_id == "big" for w in out) == 10
    assert cap_chat_share(wins, None, 0) == wins


def test_cap_ignores_held_out_and_single_chat():
    wins = [win(i, "big") for i in range(10)] + [win(i, "big", split="test") for i in range(10, 12)]
    assert cap_chat_share(wins, 0.3, 0) == wins


def test_time_split_holds_out_latest_sessions_across_chats():
    wins = [Window(f"c{i}", f"chat{i}", 0, f"2024-01-{i + 1:02d}T10:00", [], "t") for i in range(20)]
    out = split_windows(wins, 0.1, 0.1, None, 0, by="time")
    labels = [w.split for w in sorted(out, key=lambda w: w.ts)]
    assert labels[-2:] == ["test", "test"]
    assert labels[-4:-2] == ["val", "val"]
    assert set(labels[:-4]) == {"train"}
    assert {w.split for w in split_windows(wins, 0.1, 0.1, None, 0)} == {"train"}
