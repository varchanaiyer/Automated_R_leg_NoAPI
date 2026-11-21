"""
Mechanical judge for the critic game.

A critique is a list of objections in the form
    QUOTE: "<verbatim span of the finding>" METRIC: <metric key> ISSUE: <one line>
or the single line NO ISSUES.

Per objection:
  catch            the quote is verbatim in the finding and overlaps a recorded flaw span
                   (or contains the flaw's claimed value)
  false_objection  the quote is verbatim but touches no recorded flaw
  unverifiable     the quote is not a verbatim substring of the finding
Per item:
  verdict = catch if any objection catches, else false_objection if any objection is false,
            else unverifiable if the critic objected but nothing was checkable, else no_objection.

Scoring a set (eval/critic_judge.py --items data/critic/heldout.jsonl --critiques critiques.jsonl):
  catch_rate            flawed items with a catch
  false_objection_rate  items with at least one false objection (reported for clean and flawed items)
  clean_pass_rate       clean items where the critic said NO ISSUES
  metric_named_rate     catches that also name one of the flaw's metrics
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

QUOTE_RE = re.compile(r'QUOTE:\s*[“"]([^"”]{2,}?)[”"]', re.I)
MAX_QUOTE_SHARE = 0.6        # a quote may cover at most this share of the finding ...
MIN_QUOTE_CAP_CHARS = 140    # ... unless the finding is short
ANY_QUOTE_RE = re.compile(r'[“"]([^"”]{8,}?)[”"]')
METRIC_RE = re.compile(r"METRIC:\s*([a-z_]+)", re.I)
OBJECTION_RE = re.compile(r'QUOTE:\s*[“"]([^"”]{2,}?)[”"]\s*METRIC:\s*([A-Za-z_/]+)', re.I)
MIN_PINPOINT_WORDS = 4
NO_ISSUES_RE = re.compile(r"^\s*NO ISSUES?\.?\s*$", re.I | re.M)


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.strip(" .,;:\"'")


def _collapse(s: str) -> str:
    """Same character mapping as _norm but without trimming, so prefix lengths map offsets."""
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).lower()


def _flaw_positions(finding: str, finding_n: str, flaw: dict) -> Tuple[int, int]:
    """Where the flaw sits in the normalized finding: from its recorded character
    offsets when present (a value like 0.85 can occur twice), else by search."""
    cs, ce = flaw.get("char_start", -1), flaw.get("char_end", -1)
    if isinstance(cs, int) and cs >= 0 and finding[cs:ce] == flaw["span"]:
        lead = len(finding) - len(finding.lstrip())
        a = len(_collapse(finding[lead:cs])) if cs >= lead else 0
        b = a + len(_collapse(flaw["span"]).strip())
        return a, b
    sn = _norm(flaw["span"])
    pos = finding_n.find(sn)
    return (pos, pos + len(sn)) if pos >= 0 else (-1, -1)


def judge_item(item: dict, critique: str) -> dict:
    finding = item["finding"]
    finding_n = _collapse(finding).strip()
    flaws = item.get("flaws", [])
    objections = []
    pairs = [(q, m.lower()) for q, m in OBJECTION_RE.findall(critique or "")]
    if not pairs:
        loose = QUOTE_RE.findall(critique or "") or ([] if NO_ISSUES_RE.search(critique or "") else ANY_QUOTE_RE.findall(critique or ""))
        pairs = [(q, "") for q in loose]
    for q, q_metric in pairs:
        qn = _norm(q)
        if len(qn) > max(MIN_QUOTE_CAP_CHARS, MAX_QUOTE_SHARE * len(finding_n)):
            objections.append({"quote": q, "result": "unverifiable", "reason": "quote covers most of the finding"})
            continue
        # every verbatim occurrence (a bare number must match on token boundaries)
        if re.fullmatch(r"-?\d+(?:\.\d+)?%?", qn):
            occ = [(m.start(), m.end()) for m in re.finditer(r"(?<![\d.])" + re.escape(qn) + r"(?![\d.])", finding_n)]
        else:
            occ = [(m.start(), m.end()) for m in re.finditer(re.escape(qn), finding_n)]
        if not occ:
            objections.append({"quote": q, "result": "unverifiable"})
            continue
        hit = None
        for f in flaws:
            fa, fb = _flaw_positions(finding, finding_n, f)
            claimed = f.get("claimed")
            tokens = {str(claimed), f"{float(claimed):.2f}", f"{float(claimed):g}"} if claimed is not None else set()
            has_value = any(re.search(r"(?<![\d.])" + re.escape(tk) + r"(?![\d.])", qn) for tk in tokens)
            flaw_metrics = [x.lower() for x in f.get("metrics", [])]
            # An omission adds no wrong text to point at. It is caught by a verbatim
            # quote plus naming a metric the write-up fails to account for.
            if f.get("kind") == "omission" and q_metric in flaw_metrics:
                hit = f
                break
            for qa, qb in occ:
                inter = max(0, min(qb, fb) - max(qa, fa)) if fa >= 0 else 0
                covers_half = fa >= 0 and inter >= 0.5 * (fb - fa)
                names_value = claimed is not None and has_value and fa >= 0 and inter > 0
                # a quote lying wholly inside the flawed span pinpoints part of it
                pinpoint = fa >= 0 and qa >= fa and qb <= fb and len(qn.split()) >= MIN_PINPOINT_WORDS
                if covers_half or names_value or pinpoint:
                    hit = f
                    break
            if hit:
                break
        objections.append({"quote": q, "result": "catch" if hit else "false_objection",
                           "flaw_type": hit["type"] if hit else None,
                           "metric_named": bool(hit and q_metric in [x.lower() for x in hit["metrics"]])})
    results = [o["result"] for o in objections]
    if "catch" in results:
        verdict = "catch"
    elif "false_objection" in results:
        verdict = "false_objection"
    elif results:
        verdict = "unverifiable"
    else:
        verdict = "no_objection"
    return {"id": item["id"], "verdict": verdict, "flawed": bool(flaws), "objections": objections}


def score(items: List[dict], critiques: Dict[str, str]) -> dict:
    rows = [judge_item(it, critiques.get(it["id"], "")) for it in items if it["id"] in critiques]
    flawed = [r for r in rows if r["flawed"]]
    clean = [r for r in rows if not r["flawed"]]
    catches = [o for r in flawed for o in r["objections"] if o["result"] == "catch"]
    by_kind: Dict[str, List[int]] = {}
    for it in items:
        r = next((x for x in rows if x["id"] == it["id"]), None)
        if r and it["flaws"]:
            by_kind.setdefault(it["flaws"][0]["kind"], []).append(1 if r["verdict"] == "catch" else 0)
    return {
        "n": len(rows), "n_flawed": len(flawed), "n_clean": len(clean),
        "catch_rate": sum(r["verdict"] == "catch" for r in flawed) / max(1, len(flawed)),
        "false_objection_rate_flawed": sum(any(o["result"] == "false_objection" for o in r["objections"]) for r in flawed) / max(1, len(flawed)),
        "false_objection_rate_clean": sum(any(o["result"] == "false_objection" for o in r["objections"]) for r in clean) / max(1, len(clean)),
        "clean_pass_rate": sum(r["verdict"] == "no_objection" for r in clean) / max(1, len(clean)),
        "unverifiable_rate": sum(r["verdict"] == "unverifiable" for r in rows) / max(1, len(rows)),
        "metric_named_rate": sum(o.get("metric_named", False) for o in catches) / max(1, len(catches)),
        "catch_rate_by_kind": {k: sum(v) / len(v) for k, v in sorted(by_kind.items())},
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/critic/heldout.jsonl")
    ap.add_argument("--critiques", required=True, help="jsonl of {id, critique}")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    items = [json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip()]
    crit = {json.loads(l)["id"]: json.loads(l)["critique"] for l in Path(args.critiques).read_text().splitlines() if l.strip()}
    s = score(items, crit)
    print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
