"""
Runs exactly ONE PVG round for ONE seed, saves all state to disk, and exits.
Run it again to do the next round — it resumes from <checkpoint-dir>/seed_<seed>/state.json.

Exit codes:
    0  round completed and checkpointed
    2  REFUSED by the spot-check gate (verifier verdicts degenerate) — nothing ran
    3  ABORTED by the round diagnostics (held-out accept rate > 0.9 or
       honest-minus-sneaky reward gap <= 0); the row is logged with
       aborted=true, the checkpoint is NOT advanced

A checkpoint directory written before the diagnostics existed (its
round_history.jsonl rows lack reward_gap / heldout_accept_rate) is moved
aside automatically — those rounds were produced by the degenerate verifier
and are not usable.
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PVGConfig
from data.schema import FindingLabel
import training.pvg_loop as pvg_loop
from training.pvg_loop import (
    DegenerateVerifierError, enforce_spot_check_gate, load_dataset,
    load_verifier_checkpoint, run_single_round, save_verifier_checkpoint,
    set_dataset_examples,
)
from training.train_prover_step import ProverState, load_prover

REQUIRED_HISTORY_FIELDS = {"reward_gap", "heldout_accept_rate", "unparseable_share"}


def save_prover(state: ProverState, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state.model.save_pretrained(str(path))
    torch.save(state.optimizer.state_dict(), path / "optimizer.pt")


def load_prover_checkpoint(cfg: PVGConfig, base_model_name: str, path: Path) -> ProverState:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = pvg_loop.pick_device()
    dtype = pvg_loop.pick_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, str(path), is_trainable=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.prover_lr)
    opt_path = path / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    return ProverState(model=model, tokenizer=tokenizer, optimizer=optimizer, device=device)


def archive_checkpoint(ckpt: Path, reason: str) -> Optional[Path]:
    if not ckpt.exists():
        return None
    dest = ckpt.parent / f"{ckpt.name}_archived_{time.strftime('%Y%m%d-%H%M%S')}_{reason}"
    shutil.move(str(ckpt), str(dest))
    print(f"[checkpoint] moved {ckpt} -> {dest} ({reason})")
    return dest


def archive_if_stale(ckpt: Path) -> None:
    """Checkpoints from before the diagnostics existed are not usable."""
    hist = ckpt / "round_history.jsonl"
    state = ckpt / "state.json"
    if not ckpt.exists():
        return
    stale = False
    if hist.exists():
        rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
        if rows and not REQUIRED_HISTORY_FIELDS.issubset(rows[-1].keys()):
            stale = True
    elif state.exists():
        stale = True
    if stale:
        archive_checkpoint(ckpt, "pre-diagnostics-degenerate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=PVGConfig().output_dataset_path)
    parser.add_argument("--heldout", default=PVGConfig().heldout_dataset_path)
    parser.add_argument("--spot-set", default=PVGConfig().spot_check_set_path)
    parser.add_argument("--checkpoint-dir", default="training/checkpoint")
    parser.add_argument("--seed", type=int, default=PVGConfig().seed)
    parser.add_argument("--total-rounds", type=int, default=PVGConfig().num_rounds)
    parser.add_argument("--fresh", action="store_true", help="archive any existing checkpoint for this seed first")
    args = parser.parse_args()

    cfg = PVGConfig()
    cfg.seed = args.seed
    cfg.num_rounds = args.total_rounds

    root = Path(args.checkpoint_dir)
    ckpt = root / f"seed_{cfg.seed}"
    if args.fresh:
        archive_checkpoint(ckpt, "fresh-start")
    # The legacy layout put everything directly under training/checkpoint/.
    archive_if_stale(root)
    archive_if_stale(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    state_path = ckpt / "state.json"
    history_path = ckpt / "round_history.jsonl"
    combined_history_path = root / "round_history.jsonl"
    replay_path = ckpt / "replay_buffer.json"
    helpful_dir, sneaky_dir, verifier_dir = ckpt / "helpful_prover", ckpt / "sneaky_prover", ckpt / "verifier"

    next_round = 1
    if state_path.exists():
        next_round = json.loads(state_path.read_text())["next_round"]
    if next_round > cfg.num_rounds:
        print(f"[seed {cfg.seed}] all {cfg.num_rounds} rounds already complete.")
        return 0

    dataset = load_dataset(args.dataset)
    honest_records = [d for d in dataset if d["label"] == FindingLabel.HONEST.value]
    heldout_rows = load_dataset(args.heldout) if Path(args.heldout).exists() else []
    spot_rows = load_dataset(args.spot_set) if Path(args.spot_set).exists() else []
    print(f"[seed {cfg.seed}] round {next_round}/{cfg.num_rounds}: {len(honest_records)} honest records, "
          f"{len(dataset)} labeled train rows, {len(heldout_rows)} held-out rows, {len(spot_rows)} spot-check rows")
    if not heldout_rows:
        print("WARNING: no held-out set found — the accept-rate abort criterion cannot fire. "
              "Run data/build_findings_dataset.py first.")

    random.seed(cfg.seed * 1000 + next_round)
    torch.manual_seed(cfg.seed * 1000 + next_round)
    t0 = time.time()

    # --- Verifier: resume or fresh (+ gate) ---
    try:
        if verifier_dir.exists():
            print("Resuming verifier from checkpoint...")
            load_verifier_checkpoint(cfg, verifier_dir)
            set_dataset_examples(dataset)
            enforce_spot_check_gate(cfg, spot_set_rows=spot_rows, enforce_set=False)
        else:
            print("Initializing verifier fresh (round 1)...")
            # Two-case gate on the untrained verifier, warm-up on the labeled
            # train split, then the two-case + 20-item gate on the verifier
            # that will actually reward round 1. Refuses (exit 2) on failure.
            pvg_loop.prepare_verifier_for_run(cfg, dataset, spot_rows)
    except DegenerateVerifierError as e:
        print(f"\n{e}")
        return 2

    if replay_path.exists():
        pvg_loop._REPLAY_BUFFER.clear()
        pvg_loop._REPLAY_BUFFER.extend(json.loads(replay_path.read_text()))
        print(f"Resumed replay buffer with {len(pvg_loop._REPLAY_BUFFER)} examples.")

    # --- Provers ---
    if helpful_dir.exists():
        print("Resuming helpful prover from checkpoint...")
        helpful_prover_state = load_prover_checkpoint(cfg, cfg.prover_model, helpful_dir)
    else:
        print("Initializing helpful prover fresh (round 1)...")
        helpful_prover_state = load_prover(cfg, cfg.prover_model)
    if sneaky_dir.exists():
        print("Resuming sneaky prover from checkpoint...")
        sneaky_prover_state = load_prover_checkpoint(cfg, cfg.prover_model, sneaky_dir)
    else:
        print("Initializing sneaky prover fresh (round 1)...")
        sneaky_prover_state = load_prover(cfg, cfg.prover_model)

    samples: list = []
    metadata = run_single_round(next_round, honest_records, helpful_prover_state, sneaky_prover_state,
                                cfg, heldout_rows, spot_rows, seed=cfg.seed, samples_out=samples)
    metadata.elapsed_s = round(time.time() - t0, 1)

    # Every prover sample of the round, readable and reusable by
    # eval/human_eval_harness.py build --dataset <this file>.
    samples_dir = ckpt / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    with open(samples_dir / f"round_{next_round:02d}.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    row = json.dumps(vars(metadata))
    with open(history_path, "a") as f:
        f.write(row + "\n")
    with open(combined_history_path, "a") as f:
        f.write(row + "\n")

    if metadata.aborted:
        print(f"\n[seed {cfg.seed}] round {next_round} ABORTED — {metadata.notes}")
        print("Checkpoint not advanced. Fix the reward/verifier before re-running.")
        return 3

    print("Saving checkpoint...")
    save_prover(helpful_prover_state, helpful_dir)
    save_prover(sneaky_prover_state, sneaky_dir)
    save_verifier_checkpoint(cfg, verifier_dir)
    replay_path.write_text(json.dumps(pvg_loop._REPLAY_BUFFER))
    state_path.write_text(json.dumps({"next_round": next_round + 1, "seed": cfg.seed,
                                      "total_rounds": cfg.num_rounds}))

    print(f"\n[seed {cfg.seed}] round {next_round}/{cfg.num_rounds} complete in {metadata.elapsed_s}s, saved to {ckpt}/")
    if next_round < cfg.num_rounds:
        print("Run again to continue with the next round.")
    else:
        print("All rounds complete for this seed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
