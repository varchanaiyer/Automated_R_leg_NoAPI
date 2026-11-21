# Limitations and open problems

Read this before trusting any metric this repo produces. Several parts
of the design are real simplifications relative to Kirchner et al.,
not just implementation details.

## 1. Ground truth for findings is constructed, not given

In math, "is this answer correct" is checkable by a calculator. "Is
this research finding sound" has no equivalent oracle. This repo
builds ground truth by checking a finding's narrative against its own
cited metrics/config (see `data/build_findings_dataset.py`) — e.g.
does the write-up's claimed effect size match the actual numbers, does
it claim significance with too few seeds, does it cite a metric that
isn't in the data at all.

This catches INTERNAL inconsistency (the finding contradicts its own
data) but NOT external validity (the finding could be internally
consistent and still describe a bad experiment, a p-hacked result, or
a real-but-non-replicable effect). Sneaky-prover findings constructed
this way are "plausible misrepresentations of real data," not
"convincing lies about a genuinely correct result" — the paper's math
domain more cleanly separates "wrong answer" from "convincing
argument for it." Be honest about this gap in any published claim.

## 2. The sneaky prover needs a way to generate misrepresentations

Kirchner et al. can invert a correct math solution into a plausible
wrong one relatively mechanically. There's no equivalent mechanical
inversion for a research finding. `data/build_findings_dataset.py`
uses prompted generation (ask a model to write a misleading-but-fluent
version of a real finding) rather than a principled inversion
procedure. This means sneaky-prover training data quality depends
entirely on how well that generation prompt works, and should be
spot-checked by a human before trusting it as adversarial training
signal.

## 3. Human evaluation is not automated, on purpose

`eval/human_eval_harness.py` is a structured interface for running a
real human study — it does not simulate one. Any run of this pipeline
that skips human evaluation and reports only automated verifier
metrics has NOT established that findings became more legible to a
human reader. It has only shown the automated adversarial game
converged. State this plainly if you report results without running
the human-eval step.

## 4. This is not connected to the earlier critic_legibility work

Nothing here reuses trained weights, prompts, or data from the earlier
chat-critique legibility experiment in this project. The domains
(chat answer critique vs. research finding write-up) are different
enough that no direct transfer was attempted. If you want lessons from
that earlier work, the only transferable thing is a design principle,
not a component — see that project's own EXPERIMENTS.md for what it
found (or would have found, pending a real run).

## 5. Integration with the live AAR loop is a gate, not a training signal

`integration/share_finding_gate.py` hooks the TRAINED verifier into
the AAR's existing share_finding tool as an accept/reject gate on
submissions. It does NOT retrain the AAR's own weights based on
verifier feedback — the AAR in the underlying `automated-w2s-research`
codebase is a frozen model called via API, not a fine-tunable model in
this training loop. If the actual goal is to change the AAR's
underlying behavior (not just gate its outputs), that requires the AAR
itself to be a trainable model in a loop like `training/pvg_loop.py`,
which is a substantially larger infrastructure undertaking than a
gate.

## 7. The fine-tunable prover mode changes WHO the claim is about

Setting `use_finetunable_prover=True` (see `config.py`, `training/train_prover_step.py`)
makes prover fine-tuning real — both prover roles update via GRPO-style
policy gradients against the verifier's score, round over round. This is
what makes "legibility training" literally true for the prover, not just
the verifier (see the earlier discussion this project's conversation log
flagged: the frozen-API mode only trains the verifier).

But this requires replacing the real AAR (Claude, called via API, frozen)
with a small local open-weights model (e.g. Qwen2.5-7B-Instruct) as the
stand-in prover. That changes the scope of any result:

- **What you CAN claim**: "PVG training makes a Qwen-based researcher-like
  model's findings more legible, as judged by a trained verifier and
  (if you run the human-eval study) by human reviewers."
- **What you CANNOT claim from this alone**: "this improves the real,
  Claude-based AAR." The stand-in model's behavior, capacity to be
  convincing, and failure modes are not the same as Claude's — a result
  here is evidence the METHOD can work on a research-writeup domain, not
  a demonstrated improvement to the actual production AAR.

Closing that gap would require either (a) treating this as a method
validation step before proposing changes to how the real AAR is prompted/
scaffolded (informed by, not identical to, this experiment), or (b) if
Claude fine-tuning access becomes available for this purpose, redoing the
prover role with the real model — infrastructure this project does not
have.

## 8. Small-scale validation still required

Everything here is scaffolding. Before trusting any metric at scale,
run the pipeline on a handful of synthetic examples end-to-end and
manually inspect: does the verifier's accept/reject actually track
what a careful human would say; does the sneaky prover produce
findings that are subtly wrong rather than obviously broken (if
they're obviously broken, the verifier's "robustness" metric is
meaningless — it's not being tested against anything hard).

## 9. The verifier must be shown to discriminate before any round is trusted

The first training run reported accuracy ~0.95 and robustness ~0.3 for four
rounds while the verifier was returning the same echoed text for every
input. Automated metrics from this pipeline are only meaningful if (a) the
spot-check gate passed for the verifier that produced the rewards and (b)
the round's `reward_gap` is positive and `heldout_accept_rate` is not ~1.
Both are now enforced in code, but they are necessary, not sufficient: the
20-item spot set and the 48-row held-out split are small, and the synthetic
records share templates with the training split, so held-out accuracy
overstates what the verifier would do on real AAR write-ups. Zero-shot
Qwen2.5-1.5B accepts ~75% of the hand-checked set; the verifier that clears
the bar is the one warmed up on the labeled train split.
