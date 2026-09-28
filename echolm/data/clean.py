import logging
import re
from dataclasses import replace

from echolm.data.models import Msg

log = logging.getLogger(__name__)

# masking only; no case folding or language filtering, Hinglish passes through untouched
URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?<![\w])\+?\d[\d -]{8,}\d(?![\w])")
OTP = re.compile(r"\b\d{4,8}\b(?=.{0,20}\b(?:otp|code|pin)\b)", re.IGNORECASE)


def mask_phone(m: re.Match) -> str:
    digits = sum(ch.isdigit() for ch in m.group(0))
    return "<PHONE>" if digits >= 10 else m.group(0)


def mask_pii(text: str, blocked: list[str]) -> str:
    text = URL.sub("<URL>", text)
    text = EMAIL.sub("<EMAIL>", text)
    text = PHONE.sub(mask_phone, text)
    text = OTP.sub("<CODE>", text)
    for word in blocked:
        text = re.sub(re.escape(word), "<REDACTED>", text, flags=re.IGNORECASE)
    return text


def dedupe(msgs: list[Msg]) -> list[Msg]:
    # only messages with a platform id can be proven duplicates; WhatsApp has minute timestamps,
    # so two identical "haha"s in the same minute are real messages and must both stay
    seen = set()
    out = []
    for m in msgs:
        key = (m.chat_id, m.msg_id, m.kind)
        if m.msg_id is not None and key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def clean(msgs: list[Msg], blocked: list[str]) -> list[Msg]:
    # deleted and media messages stay as placeholders so turn boundaries survive
    kept = dedupe(sorted(msgs, key=lambda m: (m.chat_id, m.ts)))
    out = [replace(m, text=mask_pii(m.text, blocked)) for m in kept]
    log.info("clean: %d in, %d duplicates dropped", len(msgs), len(msgs) - len(out))
    return out
