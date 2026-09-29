import json
import logging
import re
from collections import Counter
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

DATASET = "common-pile/ubuntu_irc"
DATE = re.compile(r"(\d{4})[-/](\d{2})[-/](\d{2})")


def as_dict(meta) -> dict:
    if isinstance(meta, str):
        return json.loads(meta)
    return meta or {}


def doc_day(row: dict) -> date | None:
    for value in (row.get("created"), row.get("id"), as_dict(row.get("metadata")).get("url")):
        m = DATE.search(str(value or ""))
        if m:
            return date(*map(int, m.groups()))
    return None


def keep(row: dict, channel: str, since: int) -> dict | None:
    if as_dict(row.get("metadata")).get("channel") != channel:
        return None
    day = doc_day(row)
    if day is None or day.year < since:
        return None
    return {"day": day.isoformat(), "channel": channel, "text": row["text"]}


def fetch(out: Path, channel: str, since: int, max_docs: int | None = None) -> int:
    """Stream the dataset once and keep one channel's daily logs; the full dump is 6 GB."""
    from datasets import load_dataset

    rows = load_dataset(DATASET, split="train", streaming=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    seen, kept = Counter(), 0
    with tmp.open("w", encoding="utf-8") as f:
        for n, row in enumerate(rows, start=1):
            seen[as_dict(row.get("metadata")).get("channel")] += 1
            doc = keep(row, channel, since)
            if doc:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1
            if n % 20000 == 0:
                log.info("scanned %d docs, kept %d from %s", n, kept, channel)
            if max_docs and kept >= max_docs:
                break
    if not kept:
        top = ", ".join(f"{c} ({k})" for c, k in seen.most_common(10))
        raise ValueError(f"no logs for {channel} since {since}; busiest channels seen: {top}")
    tmp.replace(out)
    log.info("kept %d daily logs of %s since %d in %s", kept, channel, since, out)
    return kept
