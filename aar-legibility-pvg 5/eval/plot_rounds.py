"""
Plots helpful-prover accuracy and verifier robustness per round, with
intervals across seeds, from training/checkpoint/seed_*/round_history.jsonl.
Also prints the per-round table with the three diagnostics.

    python eval/plot_rounds.py --checkpoint-dir training/checkpoint --out training/checkpoint/pvg_curves.png
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

METRICS_MAIN = ["helpful_prover_accuracy", "verifier_robustness"]
METRICS_DIAG = ["heldout_accuracy", "heldout_accept_rate", "reward_gap", "unparseable_share"]


def load_histories(ckpt_root: Path) -> Dict[int, List[dict]]:
    hist = {}
    for f in sorted(ckpt_root.glob("seed_*/round_history.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if rows:
            seed = rows[0].get("seed")
            if seed is None:
                seed = int(f.parent.name.split("_", 1)[1])
            hist[seed] = rows
    return hist


def aggregate(hist: Dict[int, List[dict]], metric: str) -> Dict[int, dict]:
    per_round = defaultdict(list)
    for rows in hist.values():
        for r in rows:
            v = r.get(metric)
            if v is not None:
                per_round[r["round_number"]].append(v)
    out = {}
    for rn, vals in sorted(per_round.items()):
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
        out[rn] = {"mean": mean, "std": std, "min": min(vals), "max": max(vals), "n": len(vals)}
    return out


def print_table(hist: Dict[int, List[dict]]) -> None:
    cols = METRICS_MAIN + METRICS_DIAG
    aggs = {m: aggregate(hist, m) for m in cols}
    rounds = sorted({rn for a in aggs.values() for rn in a})
    seeds = sorted(hist)
    print(f"\nPer-round table, mean ± std across seeds {seeds}:")
    print("| round | " + " | ".join(cols) + " | aborted |")
    print("|---|" + "---|" * (len(cols) + 1))
    for rn in rounds:
        cells = []
        for m in cols:
            a = aggs[m].get(rn)
            cells.append(f"{a['mean']:.3f} ± {a['std']:.3f}" if a else "—")
        aborted = [s for s, rows in hist.items() for r in rows if r["round_number"] == rn and r.get("aborted")]
        print(f"| {rn} | " + " | ".join(cells) + f" | {aborted if aborted else ''} |")


def plot(hist: Dict[int, List[dict]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"helpful_prover_accuracy": "#1f77b4", "verifier_robustness": "#d62728",
              "heldout_accuracy": "#2ca02c", "heldout_accept_rate": "#9467bd",
              "reward_gap": "#ff7f0e", "unparseable_share": "#7f7f7f"}
    for ax, metrics, title in ((axes[0], METRICS_MAIN, "Prover accuracy vs. verifier robustness"),
                               (axes[1], METRICS_DIAG, "Round diagnostics (held-out set)")):
        for m in metrics:
            agg = aggregate(hist, m)
            if not agg:
                continue
            xs = sorted(agg)
            mean = [agg[x]["mean"] for x in xs]
            lo = [agg[x]["min"] for x in xs]
            hi = [agg[x]["max"] for x in xs]
            ax.plot(xs, mean, marker="o", color=colors.get(m), label=f"{m} (mean, n={agg[xs[0]]['n']} seeds)")
            ax.fill_between(xs, lo, hi, color=colors.get(m), alpha=0.15)
            for rows in hist.values():
                ax.plot([r["round_number"] for r in rows if r.get(m) is not None],
                        [r[m] for r in rows if r.get(m) is not None],
                        color=colors.get(m), alpha=0.25, linewidth=0.8)
        ax.set_xlabel("PVG round")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("score")
    axes[1].axhline(0.9, color="#9467bd", linestyle=":", linewidth=1)
    axes[1].axhline(0.0, color="#ff7f0e", linestyle=":", linewidth=1)
    axes[1].set_ylabel("value (abort lines: accept rate 0.9, gap 0)")
    fig.suptitle("Legibility PVG training — shaded = min/max across seeds")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"Plot written to {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="training/checkpoint")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.checkpoint_dir)
    hist = load_histories(root)
    if not hist:
        print(f"No seed_*/round_history.jsonl under {root}")
        return 1
    print_table(hist)
    plot(hist, Path(args.out) if args.out else root / "pvg_curves.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
