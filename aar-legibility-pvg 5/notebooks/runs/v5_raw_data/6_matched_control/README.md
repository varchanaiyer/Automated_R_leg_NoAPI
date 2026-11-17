# Matched control: the v5 verifier trained with no provers (seeds 42-46)

Same verifier warm-up, calibration and seeding as the v5 game, then 7 rounds trained only on the fixed example pool,
using the game's per-round example counts (seed 42's: 454, 566, 716, 872, 928, 928, 928). Round 0 is before any
round training. Run on Colab A100, 2026-11-17, about 9 minutes per seed, notebook `Control_v5_no_provers_parallel.ipynb`.

- `control_vs_game_per_round_means.csv`: mean (and sd) over the 5 seeds per round, control next to the game.
  Game held-out numbers are from the v5 log (`1_training_rounds`); the game's heldout-48 numbers are the lower
  bounds from `3_heldout_48_only`, since the game was never scored on the 48 alone.
- `control_by_seed_round.csv`: every seed and round, accuracy / AUROC / accept rate / sneaky rejected on
  heldout_100, heldout_48 and spot_20.
- `matched_control_v5_all_seeds.json`: the raw output of the notebook.

Caveat: the control's setup re-sampled the Qwen-written held-out rows, so its "heldout_100" set is 104 rows
(48 template + 56 Qwen-written), not the game's exact 100. heldout_48 and spot_20 are the same template rows as v5.
All 5 control seeds share one copy of the data (hash e98da24d38b2).
