"""
Baselines on the held-out split (step 5):
  (a) the rule check — data/build_findings_dataset.check_internal_consistency,
      which compares the numbers in the prose against the record's metrics
      and applies the seed/framing rules. Predicts SOUND iff no issue;
      the AUROC score is -(number of issues).
  (b) the same Qwen verifier, zero-shot (fresh LoRA adapter = base model),
      scored by the log-prob comparison in training/pvg_loop.py.

Both are also run on the 20-item hand-checked spot set.

    python eval/baselines.py --out eval/baselines_heldout.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import PVGConfig
from data.schema import ExperimentRecord
from data.build_findings_dataset import check_internal_consistency
from eval.metrics import compute_auroc


def _rows_metrics_config(row: dict):
    exp = row.get("experiment") or {}
    return row.get("metrics", exp.get("metrics", {})), row.get("config", exp.get("config", {})), exp


def _is_sound(row: dict) -> bool:
    if "is_sound" in row:
        return bool(row["is_sound"])
    return row.get("label") in ("honest", "helpful", "sound")


def rule_check_baseline(rows: List[dict]) -> dict:
    scores, labels, per_pert = [], [], {}
    honest_false_rejects = 0
    for row in rows:
        metrics, config, exp = _rows_metrics_config(row)
        rec = ExperimentRecord(idea_name=exp.get("idea_name", "x"), metrics=metrics, config=config,
                               worked=exp.get("worked"), dataset=exp.get("dataset"), record_id=exp.get("record_id"))
        ok, issues = check_internal_consistency(rec, row["summary"])
        sound = _is_sound(row)
        scores.append(-float(len(issues)))
        labels.append(sound)
        if sound and not ok:
            honest_false_rejects += 1
        if not sound:
            per_pert.setdefault(row.get("perturbation") or "unlabeled", []).append(not ok)
    n = len(rows)
    preds = [s >= 0 for s in scores]  # no issues -> SOUND
    return {
        "n": n,
        "accuracy": sum(1 for p, l in zip(preds, labels) if p == l) / n,
        "accept_rate": sum(preds) / n,
        "auroc": compute_auroc(scores, labels),
        "sneaky_reject_rate": 1 - (sum(1 for p, l in zip(preds, labels) if not l and p) / max(1, sum(1 for l in labels if not l))),
        "honest_false_reject_rate": honest_false_rejects / max(1, sum(labels)),
        "per_perturbation_reject_rate": {k: sum(v) / len(v) for k, v in sorted(per_pert.items())},
    }


def zero_shot_baseline(rows: List[dict], cfg: PVGConfig) -> dict:
    from training import pvg_loop
    ev = pvg_loop.evaluate_verifier_on_rows(rows, cfg)
    labels = ev["labels"]
    preds = [s >= pvg_loop.ACCEPT_THRESHOLD for s in ev["scores"]]
    return {
        "n": ev["n"],
        "accuracy": ev["accuracy"],
        "accept_rate": ev["accept_rate"],
        "auroc": ev["auroc"],
        "unparseable_share": ev["unparseable_share"],
        "sneaky_reject_rate": 1 - (sum(1 for p, l in zip(preds, labels) if not l and p) / max(1, sum(1 for l in labels if not l))),
        "honest_false_reject_rate": sum(1 for p, l in zip(preds, labels) if l and not p) / max(1, sum(labels)),
        "per_perturbation_reject_rate": dict(sorted(ev["per_perturbation_reject_rate"].items())),
    }


def _fmt(d: dict) -> str:
    keys = ["n", "accuracy", "auroc", "accept_rate", "sneaky_reject_rate", "honest_false_reject_rate", "unparseable_share"]
    return "  ".join(f"{k}={d[k]:.3f}" if isinstance(d.get(k), float) else f"{k}={d.get(k)}" for k in keys if k in d)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default=PVGConfig().heldout_dataset_path)
    parser.add_argument("--spot-set", default=PVGConfig().spot_check_set_path)
    parser.add_argument("--out", default="eval/baselines_heldout.json")
    parser.add_argument("--skip-zero-shot", action="store_true")
    args = parser.parse_args()
    cfg = PVGConfig()

    sets = {}
    for name, path in (("heldout", args.heldout), ("spot_check_set", args.spot_set)):
        p = Path(path)
        if p.exists():
            sets[name] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if "heldout" not in sets:
        print(f"No held-out set at {args.heldout}; run data/build_findings_dataset.py first.")
        return 1

    results: Dict[str, dict] = {}
    for name, rows in sets.items():
        results[name] = {"rule_check": rule_check_baseline(rows)}
        print(f"[{name}] rule check : {_fmt(results[name]['rule_check'])}")
        print(f"          per-perturbation reject rate: {results[name]['rule_check']['per_perturbation_reject_rate']}")
    if not args.skip_zero_shot:
        for name, rows in sets.items():
            results[name]["qwen_zero_shot"] = zero_shot_baseline(rows, cfg)
            print(f"[{name}] qwen 0-shot: {_fmt(results[name]['qwen_zero_shot'])}")
            print(f"          per-perturbation reject rate: {results[name]['qwen_zero_shot']['per_perturbation_reject_rate']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"verifier_model": cfg.verifier_model, "results": results}, indent=2))
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
