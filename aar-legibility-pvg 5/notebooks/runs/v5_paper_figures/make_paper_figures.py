"""
Paper figures for the v5 run (5 seeds x 10 rounds), cropped to rounds 1-7 for the main text and
uncropped for the appendix. Built from the round histories rebuilt from the notebook log, because the
per-round sample files were not saved. Sneaky success is only available as a per-round mean and std
across seeds (printed by the notebook), so its band is +/- 1 std; every other band is min/max over seeds.

    python make_paper_figures.py <pvg_v5_results_from_notebook dir> <out dir>
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
h = pd.read_csv(src / "round_history_all_seeds.csv")
tab = {k: {int(r): v for r, v in d.items()} for k, d in json.loads((src / "per_round_means_from_table.json").read_text()).items()}
TOTAL = int(h.round_number.max())
LINES = [  # (column or derived, label, colour, panel)
    ("helpful_prover_accuracy", "helpful accepted (helpful accuracy)", "tab:blue", 0),
    ("sneaky_accept", "sneaky accepted (1 - robustness)", "tab:orange", 0),
    ("role_fidelity_helpful", "helpful passes rule check", "tab:green", 1),
    ("role_fidelity_sneaky", "sneaky is flawed", "tab:purple", 1),
    ("reward_gap", "reward gap (mean P(sound) helpful - sneaky)", "tab:gray", 1),
]
h["sneaky_accept"] = 1 - h["verifier_robustness"]


def fig2(max_round, path):
    d = h[h.round_number <= max_round]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True)
    for col, label, color, k in LINES:
        g = d.groupby("round_number")[col]
        m = g.mean()
        axes[k].plot(m.index, m.values, marker="o", color=color, label=label)
        axes[k].fill_between(m.index, g.min().values, g.max().values, color=color, alpha=0.15)
    rs = [r for r in range(1, max_round + 1) if r in tab["sneaky_success_mean"]]
    mu = [tab["sneaky_success_mean"][r] for r in rs]; sd = [tab["sneaky_success_std"][r] for r in rs]
    axes[0].plot(rs, mu, marker="o", color="tab:red", label="sneaky success (flawed AND accepted)")
    axes[0].fill_between(rs, [max(0, a - b) for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)], color="tab:red", alpha=0.15)
    for ax, t in zip(axes, ("Verifier decisions", "Prover behaviour")):
        ax.set_ylim(-0.02, 1.08); ax.set_xlabel("PVG round"); ax.set_title(t); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False); ax.set_xticks(range(1, max_round + 1))
    axes[0].set_ylabel(f"rate (mean over {h.seed.nunique()} seeds)")
    span = f"rounds 1-{max_round} of {TOTAL}" if max_round < TOTAL else f"rounds 1-{TOTAL}"
    fig.suptitle(f"Verifier decisions and prover behaviour, {span}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


fig2(7, out / "fig2_verifier_and_provers_rounds1-7.png")
fig2(TOTAL, out / f"fig2_verifier_and_provers_rounds1-{TOTAL}_appendix.png")

# Figure 1 with the repo's own script, from per-seed round histories
ck = out / "_ckpt"
for s, g in h.groupby("seed"):
    p = ck / f"seed_{s}"; p.mkdir(parents=True, exist_ok=True)
    (p / "round_history.jsonl").write_text("".join(json.dumps({k: (None if pd.isna(v) else v) for k, v in r.items()}) + "\n"
                                                   for r in g.drop(columns=["sneaky_accept"]).to_dict("records")))
plot = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("eval/plot_rounds.py")
for args, name in ((["--max-round", "7"], "fig1_training_rounds1-7.png"), ([], f"fig1_training_rounds1-{TOTAL}_appendix.png")):
    subprocess.run([sys.executable, str(plot), "--checkpoint-dir", str(ck), "--out", str(out / name)] + args,
                   check=True, stdout=subprocess.DEVNULL)
print(sorted(p.name for p in out.glob("*.png")))
