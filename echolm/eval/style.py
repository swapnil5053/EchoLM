import math
import re
import statistics

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF]")
ELONGATED = re.compile(r"([a-zA-Z])\1{2,}")
PLACEHOLDER = re.compile(r"<(?:URL|EMAIL|PHONE|CODE|REDACTED)>|\[(?:media|deleted|long message|forwarded)\]")

CHATBOT = re.compile(
    r"as an ai|i'?m (?:just )?an ai|language model|i'?m sorry,? but|i can'?t (?:assist|help) with"
    r"|how can i (?:help|assist)|i'?d be (?:happy|glad) to|feel free to|let me know if"
    r"|is there anything (?:else|specific)|certainly!|great question",
    re.IGNORECASE,
)

# log-scaled so one long reply doesn't dominate the average
LOG_FEATURES = ("chars", "words", "lines")


def features(text: str) -> dict:
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    caps = [w for w in words if len(w) > 1 and w.isalpha() and w.isupper()]
    return {
        "chars": len(text),
        "words": len(words),
        "lines": text.count("\n") + 1,
        "emoji": float(bool(EMOJI.search(text))),
        "lowercase_start": float(bool(letters) and letters[0].islower()),
        "ends_with_punct": float(text.rstrip()[-1:] in (".", "!", "?")),
        "question": float("?" in text),
        "elongated": float(bool(ELONGATED.search(text))),
        "caps_words": len(caps) / max(1, len(words)),
    }


def feature_table(texts: list[str]) -> dict[str, list[float]]:
    rows = [features(t) for t in texts]
    table = {k: [r[k] for r in rows] for k in rows[0]}
    for k in LOG_FEATURES:
        table[k] = [math.log1p(v) for v in table[k]]
    return table


def style_gap(generated: list[str], real: list[str]) -> dict[str, float]:
    gen, ref = feature_table(generated), feature_table(real)
    gaps = {}
    for k, ref_vals in ref.items():
        spread = statistics.pstdev(ref_vals) or 0.5
        gaps[k] = round(abs(statistics.mean(gen[k]) - statistics.mean(ref_vals)) / spread, 3)
    gaps["mean"] = round(statistics.mean(gaps.values()), 3)
    return gaps


def chatbot_rate(texts: list[str]) -> float:
    return round(sum(bool(CHATBOT.search(t)) for t in texts) / len(texts), 3)


def strip_placeholders(text: str) -> str:
    return " ".join(PLACEHOLDER.sub(" ", text).split())
