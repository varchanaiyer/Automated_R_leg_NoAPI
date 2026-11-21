"""
Drives the full experiment: N rounds x K seeds, one subprocess per round so a
Colab disconnect loses at most one round. Re-run the same command to resume.

    python run_experiment.py --seeds 42 43 44 --rounds 10 --fresh

Exit codes: 0 all seeds finished; 2 the spot-check gate refused to train;
3 at least one seed was aborted by the round diagnostics (see its last
round_history.jsonl row). After the loop, eval/plot_rounds.py is run.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def next_round_for(ckpt_root: Path, seed: int) -> int:
    state = ckpt_root / f"seed_{seed}" / "state.json"
    if state.exists():
        return json.loads(state.read_text())["next_round"]
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="training/checkpoint")
    parser.add_argument("--dataset", default="data/findings_pvg_dataset.jsonl")
    parser.add_argument("--heldout", default="data/findings_pvg_heldout.jsonl")
    parser.add_argument("--spot-set", default="data/spot_check_set.jsonl")
    parser.add_argument("--fresh", action="store_true", help="archive existing per-seed checkpoints first")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ckpt_root = Path(args.checkpoint_dir)
    outcomes = {}
    for seed in args.seeds:
        first = True
        while True:
            nr = next_round_for(ckpt_root, seed)
            if nr > args.rounds and not (first and args.fresh):
                print(f"[seed {seed}] complete ({args.rounds} rounds).")
                outcomes[seed] = "complete"
                break
            cmd = [sys.executable, str(HERE / "run_one_round.py"),
                   "--dataset", args.dataset, "--heldout", args.heldout, "--spot-set", args.spot_set,
                   "--checkpoint-dir", str(ckpt_root), "--seed", str(seed), "--total-rounds", str(args.rounds)]
            if first and args.fresh:
                cmd.append("--fresh")
            first = False
            rc = subprocess.run(cmd).returncode
            if rc == 2:
                print(f"[seed {seed}] spot-check gate REFUSED to train. Stopping everything.")
                return 2
            if rc == 3:
                print(f"[seed {seed}] aborted by round diagnostics; moving to the next seed.")
                outcomes[seed] = "aborted"
                break
            if rc != 0:
                print(f"[seed {seed}] round failed with exit {rc}; stopping this seed.")
                outcomes[seed] = f"failed({rc})"
                break

    print("\nOutcomes:", outcomes)
    if not args.no_plot:
        subprocess.run([sys.executable, str(HERE / "eval" / "plot_rounds.py"),
                        "--checkpoint-dir", str(ckpt_root)])
    return 3 if any(v == "aborted" for v in outcomes.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
