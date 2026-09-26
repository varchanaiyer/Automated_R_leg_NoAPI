"""
Central configuration for the AAR legibility PVG pipeline.
Single source of truth — all modules import from here.
"""
import json
import os
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple


@dataclass
class PVGConfig:
    # --- Models ---
    # Two modes, chosen by use_finetunable_prover below:
    #   False: prover_model is a frozen API model (e.g. Claude). Only the
    #       verifier trains (see LIMITATIONS.md #5). Out of scope for now —
    #       no paid API is available.
    #   True: prover_model is a LOCAL, LoRA-fine-tunable stand-in model
    #       trained via GRPO against the verifier's score, mirroring
    #       Kirchner et al.'s actual method (see LIMITATIONS.md #7).
    use_finetunable_prover: bool = True
    # --- Frozen API prover (use_finetunable_prover=False) ---
    # The real AAR is a frozen Claude model, so this mode is the one that
    # matches LIMITATIONS.md #5: Claude writes the helpful and sneaky
    # findings from the same two prompts, its weights never change, and only
    # the verifier trains. Calls go through OpenRouter's OpenAI-compatible
    # chat endpoint (the key we have is an OpenRouter key). Completions are
    # cached on disk under the checkpoint dir, keyed by seed/round/role/
    # record/index, so a resumed round never re-spends.
    prover_api_model: str = "anthropic/claude-opus-5"
    prover_api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    prover_api_key_env: str = "OPENROUTER_API_KEY"
    prover_api_max_tokens: int = 1200          # includes the model's reasoning tokens
    prover_api_reasoning_effort: str = "low"   # OpenRouter unified `reasoning.effort`; "" to omit
    # Same length band for both roles: in a 16-sample check Claude's honest
    # write-ups averaged 226 words and its sneaky ones 146, a shortcut a small
    # verifier could learn instead of checking the numbers.
    prover_api_system: str = ("Reply with the finding write-up only: one paragraph of plain prose, "
                              "between 110 and 160 words, no heading, no preamble, no bullet points, "
                              "no closing remark.")
    # Small model so a free-tier Colab T4 (16GB) can hold both prover
    # adapters + the verifier at once.
    prover_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # The verifier should be small relative to the prover, per the paper's
    # design. Same base here; the verifier gets its own LoRA adapter.
    verifier_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # --- Fine-tunable prover settings ---
    prover_lora_r: int = 16
    prover_lora_alpha: int = 16
    prover_lr: float = 2e-5
    prover_num_generations: int = 4       # samples per experiment, per role, per round
    # Findings are ~60-120 tokens; 512 was mostly wasted generation time.
    prover_max_new_tokens: int = 200
    prover_temperature: float = 0.9
    prover_kl_penalty: float = 0.02       # penalize drifting too far from the base model
    # Wrap prover prompts in the model's chat template (Instruct models
    # complete raw text instead of following it — the same bug class that
    # broke the verifier).
    prover_use_chat_template: bool = True
    # GRPO advantage: (r - mean) / std amplifies tiny reward noise into O(1)
    # advantages when a whole group scores nearly the same (which is common
    # with a continuous P(sound) reward). Default is the Dr.GRPO-style
    # unnormalized advantage (r - mean); set True for the classic form.
    grpo_normalize_std: bool = False
    # Prover reward, with the rule check (data/build_findings_dataset.py's
    # check_internal_consistency) as the ground-truth oracle:
    #   "convincingness":    reward = verifier P(sound) for both roles. Round 1
    #                        on Colab showed the helpful prover passing the rule
    #                        check only 16% of the time under this reward: it
    #                        learns to be convincing, not accurate.
    #   "helpful_gated":     helpful reward = P(sound) if the completion passes
    #                        the rule check else 0; sneaky reward = P(sound)
    #                        (free to be subtle). Default.
    #   "correctness_gated": Kirchner et al.'s CGC — both roles gated (sneaky
    #                        must FAIL the rule check to earn P(sound)).
    prover_reward_mode: str = "helpful_gated"

    # --- PVG training loop ---
    num_rounds: int = 10                 # outer alternation rounds
    # (experiment) records sampled per round. Each yields
    # prover_num_generations completions per role, all of which feed the
    # verifier. 8 records -> 64 completions/round; 64 records would be ~50
    # min/round on a T4.
    findings_per_round: int = 8
    verifier_train_epochs_per_round: int = 1
    # LoRA, so 1e-4 (a full-fine-tune 1e-5 barely moves an adapter).
    verifier_lr: float = 1e-4
    verifier_batch_size: int = 16
    verifier_lora_r: int = 16
    verifier_lora_alpha: int = 16
    # q/v-only LoRA left the warm-up underfit (BCE 0.44 after 6 epochs on
    # 192 rows); all linear layers fit far faster for negligible memory.
    verifier_lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj")
    # After every verifier update, shift the SOUND/UNSOUND logit by a scalar
    # bias so the TRAIN split's honest and sneaky rows sit symmetrically
    # around the 0.5 threshold (midpoint of the two class medians). The
    # first Colab warm-up ranked the spot set at AUROC 0.86 but accepted only
    # 25% of it — a threshold drift, not a discrimination failure. The bias
    # is saved with the checkpoint and never fitted on the spot/held-out sets.
    verifier_calibrate_bias: bool = True
    verifier_calibration_rows: int = 96   # balanced subsample of the train split scored per calibration
    # Before round 1 the verifier does N epochs on the TRAIN split of the
    # labeled dataset, so round-1 provers see a non-degenerate reward.
    verifier_warmup_epochs: int = 2
    verifier_warmup_max_epochs: int = 6   # extra epochs allowed while the 20-item gate bar is unmet
    # The verifier trains each round on a sliding window of recent prover
    # samples plus the labeled dataset. Capped for memory/time.
    verifier_replay_max: int = 600
    # Include the labeled TRAIN split in every verifier update (not just
    # the prover-sample replay window). Turn off only for plumbing tests.
    verifier_train_on_dataset: bool = True
    # Labels for the provers' samples in verifier training come from the rule
    # check, not the role: helpful & fails -> UNSOUND (relabeled), sneaky &
    # passes -> dropped (unknown: subtle misrepresentation or plain honest).
    # Role labels alone fed 64 mostly-wrong 'SOUND' rows per round.
    verifier_labels_from_rule_check: bool = True

    # --- Data construction ---
    source_findings_dir: str = "data/source_findings"
    output_dataset_path: str = "data/findings_pvg_dataset.jsonl"
    heldout_dataset_path: str = "data/findings_pvg_heldout.jsonl"
    spot_check_set_path: str = "data/spot_check_set.jsonl"
    dataset_target_pairs: int = 120       # honest/sneaky pairs before the split
    heldout_fraction: float = 0.2         # split is BY RECORD so pairs stay together
    dataset_seed: int = 42

    # Minimum internal-consistency checks a "helpful" finding must pass to be
    # used as ground-truth-honest training data (see data/schema.py).
    require_cited_metrics_match: bool = True
    require_num_seeds_disclosed: bool = True

    # --- Gates and abort criteria (training refuses to run / stops itself) ---
    # Step-2 gate: the two fixed spot-check cases must get DIFFERENT
    # verdicts and both must be parseable, every time a training process
    # starts. At round 1 (untrained verifier) the 20-item hand-checked set
    # must also clear these bars.
    gate_min_spot_accuracy: float = 0.70
    gate_accept_rate_range: Tuple[float, float] = (0.3, 0.7)
    # Step-3 abort: after each round's verifier update, on the held-out
    # balanced set.
    abort_if_heldout_accept_rate_above: float = 0.9
    abort_if_reward_gap_at_or_below: float = 0.0

    # --- Evaluation ---
    human_eval_batch_size: int = 40
    human_eval_time_limit_seconds: int = 120
    human_eval_output_dir: str = "eval/human_eval_sessions"

    # --- Integration with the live AAR loop ---
    gate_reject_threshold: float = 0.3
    # Where run_one_round.py leaves the trained verifier adapter (per seed).
    trained_verifier_dir: str = "training/checkpoint/seed_42/verifier"

    seed: int = 42

    def __post_init__(self):
        # Optional JSON overrides for quick experiments / smoke tests without
        # editing this file, e.g.
        #   PVG_CONFIG_OVERRIDES='{"findings_per_round": 2, "num_rounds": 1}'
        raw = os.environ.get("PVG_CONFIG_OVERRIDES")
        if raw:
            known = {f.name for f in fields(self)}
            for k, v in json.loads(raw).items():
                if k not in known:
                    raise KeyError(f"PVG_CONFIG_OVERRIDES: unknown field {k!r}")
                setattr(self, k, tuple(v) if isinstance(v, list) else v)


@dataclass
class RunMetadata:
    """Attached to every PVG training round's output for later analysis."""
    round_number: int
    seed: Optional[int] = None
    helpful_prover_accuracy: Optional[float] = None
    verifier_robustness: Optional[float] = None
    # --- diagnostics added after the degenerate-verifier incident ---
    unparseable_share: Optional[float] = None   # scoring calls with no usable verdict / all calls
    heldout_accept_rate: Optional[float] = None  # P(sound)>=0.5 rate on the held-out balanced set
    heldout_accuracy: Optional[float] = None
    heldout_auroc: Optional[float] = None
    reward_gap: Optional[float] = None           # mean helpful P(sound) - mean sneaky P(sound)
    helpful_mean_reward: Optional[float] = None
    sneaky_mean_reward: Optional[float] = None
    role_fidelity_helpful: Optional[float] = None  # helpful completions passing the rule check
    role_fidelity_sneaky: Optional[float] = None   # sneaky completions failing the rule check
    spot_check_accuracy: Optional[float] = None    # 20-item hand-checked set
    spot_check_accept_rate: Optional[float] = None
    spot_check_auroc: Optional[float] = None
    verifier_relabeled_helpful: Optional[int] = None  # helpful samples relabeled UNSOUND by the rule check
    verifier_dropped_sneaky: Optional[int] = None     # sneaky samples dropped (passed the rule check)
    verifier_train_examples: Optional[int] = None
    verifier_train_loss: Optional[float] = None
    verifier_logit_bias: Optional[float] = None    # calibration offset in force for this round
    aborted: bool = False
    elapsed_s: Optional[float] = None
    human_eval_helpful_accuracy: Optional[float] = None
    human_eval_sneaky_accuracy: Optional[float] = None
    notes: str = ""
