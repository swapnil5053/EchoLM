from echolm.data.synthetic_dialogues import DIALOGUES


def test_every_dialogue_has_both_speakers():
    for d in DIALOGUES:
        assert {who for who, _ in d} == {"me", "them"}


def test_texts_are_non_empty():
    assert all(text.strip() for d in DIALOGUES for _, text in d)
