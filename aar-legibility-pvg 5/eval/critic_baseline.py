"""
Claude as the critic, scored by the mechanical judge. The first critic-game
baseline: no training anywhere, so the numbers show what the dataset and
judge can measure before anyone trains a critic or a judge.

    OPENROUTER_API_KEY=... python eval/critic_baseline.py --n 40
"""
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import PVGConfig
from data.build_critic_dataset import api_call, cache_cost
from eval.critic_judge import score

CRITIC_SYSTEM = (
    "You are a research-integrity critic. You are given an experiment's metrics and config and a write-up "
    "of the result. Find every claim in the write-up that the data does not support: a number that is not in "
    "the data, a true number attributed to the wrong quantity, a certainty or significance claim the sample "
    "size or standard error does not support, an improvement or comparison the numbers do not support, or a "
    "missing caveat that changes the reading (an undisclosed small seed count, a result below its baseline). "
    "Tone is not evidence: do not object to wording unless you can point to the data it conflicts with.\n"
    "Output format, one objection per line, nothing else:\n"
    'QUOTE: "<a short span copied verbatim from the write-up>" METRIC: <metric key> ISSUE: <one sentence>\n'
    "If every claim is supported, output exactly: NO ISSUES"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/critic/heldout.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/critic_baseline.json")
    args = ap.parse_args()
    cfg = PVGConfig()
    items = [json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)
    flawed = [it for it in items if it["flaws"]]
    clean = [it for it in items if not it["flaws"]]
    rng.shuffle(flawed)
    rng.shuffle(clean)
    chosen = flawed[: args.n * 2 // 3] + clean[: args.n - args.n * 2 // 3]
    critiques = {}
    for i, it in enumerate(chosen):
        rec = it["record"]
        user = (f"Method: {rec.get('idea_name')}\nDataset: {rec.get('dataset')}\n"
                f"Metrics: {json.dumps(rec.get('metrics', {}), sort_keys=True)}\n"
                f"Config: {json.dumps(rec.get('config', {}), sort_keys=True)}\n\nWrite-up:\n{it['finding']}")
        critiques[it["id"]] = api_call(CRITIC_SYSTEM, user, cfg, max_tokens=700, temperature=0.0)
        print(f"  {i + 1}/{len(chosen)} cost=${cache_cost():.2f}", flush=True)
    s = score(chosen, critiques)
    summary = {k: v for k, v in s.items() if k != "rows"}
    summary["model"] = cfg.prover_api_model
    print(json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps({"summary": summary, "critiques": critiques, "rows": s["rows"]}, indent=1))
    # a few examples of each verdict
    for verdict in ("catch", "false_objection", "unverifiable"):
        for r in [r for r in s["rows"] if r["verdict"] == verdict][:2]:
            it = next(x for x in chosen if x["id"] == r["id"])
            print(f"\n[{verdict}] {it['id']} source={it['source']} flaws={[f['span'][:60] for f in it['flaws']]}\n  critique: {critiques[it['id']][:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
