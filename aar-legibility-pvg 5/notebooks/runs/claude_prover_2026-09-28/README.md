# Claude-prover run, 28 Sep 2026 (frozen Claude Opus 5 writes both roles, only the Qwen verifier trains; seeds 42-44, 10 rounds)

- `seed_NN_round_history.jsonl`, `all_seeds_round_history.csv`: one row per seed and round, rebuilt from the notebook's printed log (3-decimal precision; role fidelity to 2 decimals). `api_cost_usd_cumulative` is the per-seed OpenRouter spend after that round.
- Verifier labels came from the role in this run (`verifier_labels_from_rule_check: false`), so nothing was relabeled or dropped.

The per-round sample files and the Claude response cache were not saved from this run; the notebook had no download step.
