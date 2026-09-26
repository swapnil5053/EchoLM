from datetime import datetime
from datetime import timedelta

import pytest

from echolm.data.models import TEXT
from echolm.data.models import Msg

T0 = datetime(2025, 3, 1, 20, 0)


def make_msg(minute: float, is_me: bool, text: str = "hi", chat: str = "c1", kind: str = TEXT,
             msg_id: str | None = None, reply_to: str | None = None) -> Msg:
    return Msg(T0 + timedelta(minutes=minute), chat, "me" if is_me else "them", text, is_me, kind,
               msg_id, reply_to)


@pytest.fixture
def mk():
    return make_msg
