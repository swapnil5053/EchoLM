from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime

TEXT = "text"
MEDIA = "media"
DELETED = "deleted"
FORWARDED = "forwarded"


@dataclass
class Msg:
    ts: datetime
    chat_id: str
    sender: str
    text: str
    is_me: bool
    kind: str = TEXT
    msg_id: str | None = None
    reply_to: str | None = None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["ts"] = self.ts.isoformat()
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Msg":
        return cls(**{**d, "ts": datetime.fromisoformat(d["ts"])})


def render_msg(m: Msg) -> str:
    if m.kind == TEXT:
        return m.text
    if m.kind == FORWARDED:
        return f"[forwarded] {m.text}".rstrip()
    return f"[{m.kind}]"


@dataclass
class Turn:
    is_me: bool
    msgs: list[Msg] = field(default_factory=list)

    @property
    def ts(self) -> datetime:
        return self.msgs[-1].ts

    def render(self) -> str:
        return "\n".join(render_msg(m) for m in self.msgs)

    def own_text(self) -> str:
        return "\n".join(m.text for m in self.msgs if m.kind == TEXT and m.text)


@dataclass
class Window:
    id: str
    chat_id: str
    session: int
    ts: str
    context: list[dict]
    target: str
    quoted: str | None = None
    split: str = "train"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Window":
        return cls(**d)
