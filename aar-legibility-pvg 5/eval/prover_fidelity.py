"""
Rule-check pass rate of the UNTRAINED prover's write-ups, per role.

Run this before spending GPU hours: if the base model's helpful write-ups
fail the rule check most of the time, the helpful prover earns no reward,
its samples are relabeled UNSOUND, and the verifier learns to reject prover
text (what happened in the 26 Sep run, helpful fidelity 0.09).

    python eval/prover_fidelity.py --records 8 --generations 4

Prints helpful fidelity (share passing the rule check), sneaky fidelity
(share failing it), why helpful write-ups fail, and two failing examples.
Exit code 1 if helpful fidelity is below --min-helpful.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import PVGConfig
from training import pvg_loop

ISSUE_KINDS = [
    ("cites number", "number not in the record"),
    ("claims [", "wrong seed count"),
    ("is not disclosed", "seed count not stated"),
    ("strong-confidence", "strong-confidence wording"),
    ("positive framing", "positive framing on a result that did not work"),
    ("beat the baseline", "claims to beat the baseline"),
]


def issue_kind(issue: str) -> str:
    for needle, name in ISSUE_KINDS:
        if needle in issue:
            return name
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=PVGConfig().output_dataset_path)
    ap.add_argument("--records", type=int, default=8)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-helpful", type=float, default=0.5)
    ap.add_argument("--out", default="eval/prover_fidelity.json")
    args = ap.parse_args()

    cfg = PVGConfig()
    rows = pvg_loop.load_dataset(args.dataset)
    honest = [r["experiment"] for r in rows if r["label"] == "honest"]
    random.Random(args.seed).shuffle(honest)
    recs = honest[:args.records]

    device = pvg_loop.pick_device()
    tok = AutoTokenizer.from_pretrained(cfg.prover_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.prover_model, dtype=pvg_loop.pick_dtype(device)).to(device).eval()
    torch.manual_seed(args.seed)

    results = {"helpful": [], "sneaky": []}
    for rec in recs:
        for role, tmpl in (("helpful", pvg_loop._HELPFUL_PROMPT_PATH), ("sneaky", pvg_loop._SNEAKY_PROMPT_PATH)):
            prompt = pvg_loop.render_prover_prompt(tmpl, rec, tok, cfg)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=cfg.prover_max_new_tokens, do_sample=True,
                                     temperature=cfg.prover_temperature, top_p=0.95,
                                     num_return_sequences=args.generations, pad_token_id=tok.pad_token_id)
            for c in tok.batch_decode(out[:, ids["input_ids"].shape[1]:], skip_special_tokens=True):
                c = c.strip()
                ok, issues = pvg_loop.rule_check(rec, c)
                results[role].append({"record_id": rec.get("record_id"), "metrics": rec.get("metrics"),
                                      "summary": c, "rule_check_passes": ok, "issues": issues})

    h, s = results["helpful"], results["sneaky"]
    fid_h = sum(x["rule_check_passes"] for x in h) / len(h)
    fid_s = sum(not x["rule_check_passes"] for x in s) / len(s)
    kinds = Counter(issue_kind(i) for x in h if not x["rule_check_passes"] for i in x["issues"])
    print(f"untrained {cfg.prover_model}: {len(recs)} records x {args.generations} samples per role")
    print(f"  helpful fidelity (passes the rule check): {fid_h:.3f}   target >= {args.min_helpful}")
    print(f"  sneaky  fidelity (fails the rule check):  {fid_s:.3f}")
    print("  why helpful write-ups fail:", dict(kinds.most_common()) or "none failed")
    for x in [x for x in h if not x["rule_check_passes"]][:2]:
        print(f"\n  FAILING helpful write-up for {x['record_id']} {json.dumps(x['metrics'])}\n    issues: {x['issues']}\n    {x['summary']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"helpful_fidelity": fid_h, "sneaky_fidelity": fid_s,
                                          "helpful_failure_kinds": dict(kinds), "samples": results}, indent=1))
    print(f"\nWritten to {args.out}")
    return 0 if fid_h >= args.min_helpful else 1


if __name__ == "__main__":
    sys.exit(main())
