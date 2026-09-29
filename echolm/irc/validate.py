import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

from echolm.irc.log import parse_day
from echolm.irc.threads import RULES
from echolm.irc.threads import assign

log = logging.getLogger(__name__)


def read_links(path: Path) -> dict[int, set[int]]:
    parents = defaultdict(set)
    for row in path.read_text(encoding="utf-8").splitlines():
        parts = row.split()
        if len(parts) >= 2:
            a, b = sorted((int(parts[0]), int(parts[1])))
            parents[b].add(a)
    return parents


def score_file(raw: Path, ann: Path, rules: tuple[str, ...], gap_min: float) -> dict:
    day = date.fromisoformat(raw.name[:10])
    lines = parse_day(raw.read_text(encoding="utf-8", errors="replace"), day)
    nick = {ln.idx: ln.nick for ln in lines}
    parents = read_links(ann)
    got = {a.line.idx: a for a in assign(lines, gap_min, rules)}
    out = defaultdict(int)
    for j, a in got.items():
        links = parents.get(j, set()) - {j}
        if not links:
            continue
        ok = any(nick.get(i) == a.partner
                 or (nick.get(i) == a.line.nick and i in got and got[i].partner == a.partner)
                 for i in links)
        out[f"{a.rule}_n"] += 1
        out[f"{a.rule}_ok"] += ok
    edges = [(i, j) for j, ps in parents.items() for i in ps
             if i != j and i in nick and j in nick and nick[i] != nick[j]]
    out["edges"] = len(edges)
    out["edges_found"] = sum(j in got and got[j].partner == nick[i] for i, j in edges)
    return out


def summarize(total: dict) -> dict:
    n = sum(total[f"{r}_n"] for r in RULES)
    ok = sum(total[f"{r}_ok"] for r in RULES)
    out = {"precision": ok / n if n else 0.0, "attached": n,
           "recall": total["edges_found"] / total["edges"] if total["edges"] else 0.0,
           "reply_edges": total["edges"]}
    for r in RULES:
        if total[f"{r}_n"]:
            out[f"precision_{r}"] = total[f"{r}_ok"] / total[f"{r}_n"]
            out[f"n_{r}"] = total[f"{r}_n"]
    return out


def validate(data_dir: Path, rules: tuple[str, ...] = RULES, gap_min: float = 1.0,
             skip: tuple[str, ...] = ()) -> dict:
    """Score thread reconstruction against the hand-labelled reply links of Kummerfeld et al. (ACL 2019)."""
    raws = [p for p in sorted(data_dir.rglob("*.raw.txt")) if p.name[:-len(".raw.txt")] not in skip]
    if not raws:
        raise ValueError(f"no *.raw.txt files under {data_dir}; point --data at irc-disentanglement/data")
    total = defaultdict(int)
    for raw in raws:
        ann = raw.with_name(raw.name.replace(".raw.txt", ".annotation.txt"))
        if ann.exists():
            for k, v in score_file(raw, ann, rules, gap_min).items():
                total[k] += v
    out = summarize(total)
    log.info("%d files: precision %.3f over %d attached lines, recall %.3f of %d reply links",
             len(raws), out["precision"], out["attached"], out["recall"], out["reply_edges"])
    return out
