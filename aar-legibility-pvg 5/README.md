# AAR Legibility via Prover-Verifier Games

## Status (23 Sep 2026): verifier fixed, gates in, dataset grown — training re-run pending on Colab

**What went wrong.** The verifier (Qwen2.5-1.5B-Instruct) was fed the raw
prompt text with no chat template, so it *continued* the document instead
of answering: both spot-check cases returned the prompt's own answer menu
(`"SOUND - The write-up accurately reflects the data.\nUNSURE - ..."`), the
substring parser found `SOUND` in that echo and accepted everything. Rounds
1–4 of the first run (accuracy 0.75–1.00, robustness 0.15–0.40, sneaky
reward > helpful reward) were noise and have been discarded.

**What changed.**
- `training/pvg_loop.py` — verdict = `sigmoid(logp("SOUND") - logp("UNSOUND"))`
  under the chat template, one batched forward pass, no free text. Verifier
  training is binary cross-entropy on that same logit. Prover prompts also go
  through the chat template.
- Spot-check **gate**: `run_one_round.py` / `pvg_loop.py` refuse to train
  (exit 2) unless the two fixed cases split the right way and the verifier
  that plays round 1 is >0.70 accurate with accept rate in [0.3, 0.7] on
  `data/spot_check_set.jsonl` (20 hand-checked items).
- Per-round **diagnostics** in `round_history.jsonl`: `unparseable_share`,
  `heldout_accept_rate` / `heldout_accuracy` / `heldout_auroc`, `reward_gap`
  (mean helpful P(sound) − mean sneaky P(sound)), role fidelity vs. the rule
  check. A run **aborts itself** (exit 3) if held-out accept rate > 0.9 or
  reward gap ≤ 0.
- **Dataset**: `data/build_findings_dataset.py` builds 120 honest/sneaky
  pairs (5 real + 115 synthetic weak-to-strong records), 20% held out *by
  record* with seed 42, and records the perturbation behind every sneaky
  row. ~40% of sneaky rows are true-numbers/misleading-framing cases the
  rule check cannot catch.
- **Baselines**: `eval/baselines.py` (rule check + zero-shot Qwen on the
  held-out split). **Runs**: `run_experiment.py --seeds 42 43 44 --rounds 10`,
  curves via `eval/plot_rounds.py`. Notebook: `notebooks/Experiments_v2.ipynb`.

**24 Sep, first Colab run.** The gate refused: after warm-up the verifier
ranked the spot set at AUROC 0.86 but accepted only 25-30% of it (threshold
drift). Now: a scalar logit bias is calibrated on the train split after every
verifier update and saved with the checkpoint; the verifier LoRA covers all
linear layers; a third of synthetic records are sparse (pgr / se / seeds
only) like the real findings and the spot-check cases.

**Run order.**
```
python data/build_findings_dataset.py --target-pairs 120 --seed 42
python eval/baselines.py
python run_experiment.py --seeds 42 43 44 --rounds 10 --fresh
```

## Goal

Apply Kirchner et al.'s Prover-Verifier Game (PVG) method
(https://arxiv.org/abs/2407.13692) directly to an Automated Alignment
Researcher's (AAR) own research output — findings, hypotheses, result
write-ups — instead of to chain-of-thought math solutions (the original
paper's domain) or to chat-answer critiques (an earlier, now-abandoned
direction in this project).

The concrete question this repo is built to answer:

    Does training a "verifier" to distinguish sound AAR findings from
    unsound-but-plausible-looking ones, and training the AAR (the
    "prover") against that verifier, produce AAR research output that
    is more checkable — without making the AAR's actual findings less
    useful or less frequently correct?

This is a full redesign, not an extension of the earlier
critique-based `critic_legibility` work in this project. That work
(training a critic to write persuasive-vs-legible critiques of chat
answers, scored via a single blended GRPO reward) is set aside
entirely. Nothing here depends on it.

## Why the original paper's method, not the simplified blend

The earlier direction used one scalar reward:
`reward = alpha*task_reward + (1-alpha)*legibility_score`. That is a
simplification of PVG, not PVG itself — it has no adversarial
component, so the verifier is never stress-tested against an opponent
actively trying to fool it. Kirchner et al.'s actual result rests on
three roles trained in alternation:

- **Helpful prover** — produces solutions/findings that are correct
  AND that the verifier accepts.
- **Sneaky prover** — produces solutions/findings that are INCORRECT
  but constructed to fool the verifier into accepting them anyway.
- **Verifier** — a small model trained to predict correctness, updated
  against both provers' current behavior.

This repo implements that three-role structure, retargeted at research
findings instead of math solutions.

## Domain mapping (math → AAR findings)

| Kirchner et al. (math CoT)              | This repo (AAR findings)                                   |
|------------------------------------------|--------------------------------------------------------------|
| Grade-school math problem                | A completed experiment (metrics + config already known)      |
| Chain-of-thought solution                | A research finding write-up (summary, claimed result)         |
| Ground-truth correct/incorrect answer    | Whether the finding's claims are actually supported by its own cited metrics/config |
| Helpful prover                           | AAR producing an accurate, well-supported finding             |
| Sneaky prover                            | AAR (or a trained adversary) producing a finding that misrepresents, overstates, or fabricates support for its claim |
| Small verifier                           | A model trained to accept/reject a finding as sound            |
| Human checking accuracy                  | Human reviewers' accuracy judging findings, before vs. after training |

## What "ground truth" means here

Kirchner et al. have an unambiguous ground truth: the math answer is
right or wrong. AAR findings don't have that automatically — a
finding is a claim ABOUT data that already exists (metrics, configs,
logs from an experiment run). We construct ground truth by checking
internal consistency: does the finding's narrative claim match what
its own cited numbers actually show. See `data/build_findings_dataset.py`
for exactly how this is operationalized, and `LIMITATIONS.md` for why
this is a real, non-trivial simplification worth being upfront about.

## Repo layout

```
aar-legibility-pvg/
  README.md                        - this file
  LIMITATIONS.md                   - honest scope and known gaps
  config.py                        - all tunable settings
  requirements.txt
  data/
    schema.py                      - the Finding / ExperimentRecord data model
    build_findings_dataset.py      - constructs (experiment, honest write-up,
                                      fabricated write-up) training pairs
  prompts/
    helpful_prover_prompt.jinja2
    sneaky_prover_prompt.jinja2
    verifier_prompt.jinja2
  training/
    pvg_loop.py                    - the main iterative training loop
    train_verifier_step.py         - one verifier update
    train_prover_step.py           - one prover update (helpful or sneaky)
  eval/
    metrics.py                     - the three core metrics (see below)
    human_eval_harness.py          - structured human-eval CLI/export
  integration/
    share_finding_gate.py          - hook the trained verifier into the
                                      AAR's existing share_finding tool
  scripts/
    run_pvg_training.sh
    run_eval.sh
```

## Metrics (directly mirroring the paper)

1. **Helpful prover accuracy** — of findings the helpful prover
   produces, what fraction does the verifier accept, and (separately,
   the metric that actually matters) what fraction are ACTUALLY
   correct per the ground-truth check.
2. **Verifier robustness** — of findings the sneaky prover produces
   (which are, by construction, unsupported/misrepresented), what
   fraction does the verifier correctly reject. Tracked over training
   rounds — should increase as the verifier adapts to the sneaky
   prover's current tricks.
3. **Human checking accuracy** — time-boxed human reviewers judge a
   mixed batch of helpful-prover and sneaky-prover findings as
   sound/unsound, WITHOUT seeing the verifier's own verdict. Tracked
   at the start and end of training. The paper's headline result is
   that this rises for helpful-prover output and falls for
   sneaky-prover output over training — this repo's eval harness is
   built to produce that same comparison, but it requires you to
   actually recruit and run reviewers; nothing here fakes it
   automatically. See `eval/human_eval_harness.py`.

## What "proving legibility improved research quality" requires

To make that claim rigorously you need, at minimum:
- Metric 1 showing helpful-prover accuracy holds or improves (legibility
  training didn't come at the cost of usefulness)
- Metric 2 showing verifier robustness increases over rounds (the
  verifier is actually getting harder to fool, not just static)
- Metric 3 (the real payoff metric) showing human accuracy rises on
  helpful output and falls on sneaky output post-training — this is
  the part that actually says "legibility improved," since 1 and 2
  are about the automated game, not about whether humans (the actual
  target of legibility) benefit

Skipping metric 3 and only reporting 1 and 2 would not, by itself,
establish that AAR research became more legible to a human reader —
only that the automated verifier game converged. Say this explicitly
in any write-up.
