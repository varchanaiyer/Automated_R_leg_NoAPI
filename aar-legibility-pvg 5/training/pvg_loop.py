"""
Main Prover-Verifier Game training loop, retargeted at AAR research findings.

Mirrors Kirchner et al.'s alternating training structure:
  for each round:
    1. Sample a batch of experiment records.
    2. Helpful prover samples findings for them -> score with current verifier
       -> GRPO-style update (training/train_prover_step.py).
    3. Sneaky prover does the same, prompted to misrepresent.
    4. Update the verifier on this round's samples (helpful -> SOUND,
       sneaky -> UNSOUND) plus a replay window of earlier rounds and the
       labeled ground-truth dataset.
    5. Log the round's metrics AND diagnostics (RunMetadata in config.py),
       and abort the run if the diagnostics say the game is degenerate.

Verifier scoring — the fix for the degenerate-verdict incident (Sept 2026):
  The verifier is an Instruct model. The old code fed it the raw prompt
  text with no chat template, so it CONTINUED the document (it echoed the
  answer menu back verbatim for every input); the substring parser then
  found "SOUND" inside that echo and accepted everything (rounds 1-4:
  accuracy ~0.95, robustness ~0.3, sneaky reward > helpful reward).
  Now the prompt goes through the chat template and the verdict is the
  log-probability comparison of the two answer strings:
      P(sound) = sigmoid( logp("SOUND") - logp("UNSOUND") )
  No free text is generated or parsed. Verifier training uses binary
  cross-entropy on that same two-way logit, so what is trained is exactly
  what is scored. An "unparseable" verdict now means a non-finite logit;
  it is counted (unparseable_share) and treated as a rejection.

Run all rounds in one process (smoke test):
    python training/pvg_loop.py --dataset data/findings_pvg_dataset.jsonl --rounds 1
Real runs (per-round checkpointing, several seeds):
    python run_experiment.py --seeds 42 43 44 --rounds 10
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config import PVGConfig, RunMetadata
from data.schema import ExperimentRecord, FindingLabel
from data.build_findings_dataset import check_internal_consistency
from eval.metrics import (
    compute_helpful_prover_accuracy,
    compute_verifier_robustness,
    compute_auroc,
)

ANSWER_SOUND = "SOUND"
ANSWER_UNSOUND = "UNSOUND"
ACCEPT_THRESHOLD = 0.5  # keep in sync with eval/metrics.py VERIFIER_ACCEPT_THRESHOLD


class DegenerateVerifierError(RuntimeError):
    """Raised by the spot-check gate. Training must not proceed."""


def load_dataset(path) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    if device in ("cuda", "mps"):
        return torch.bfloat16  # fp16 overflows Qwen2.5's activations
    return torch.float32


# ---------------------------------------------------------------------------
# Local Qwen verifier: lazily loaded once, reused for scoring + training.
# No Anthropic API calls anywhere in this module.
# ---------------------------------------------------------------------------
_VERIFIER_CACHE: Dict[str, dict] = {}


def _answer_token_ids(tokenizer) -> Dict[str, List[int]]:
    return {w: tokenizer.encode(w, add_special_tokens=False) for w in (ANSWER_SOUND, ANSWER_UNSOUND)}


def register_verifier(verifier_model: str, model, tokenizer, device: str,
                      cfg: Optional[PVGConfig] = None, optimizer=None, logit_bias: float = 0.0) -> dict:
    cfg = cfg or PVGConfig()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg.verifier_lr
        )
    _VERIFIER_CACHE[verifier_model] = {
        "model": model, "tokenizer": tokenizer, "device": device,
        "optimizer": optimizer, "answer_ids": _answer_token_ids(tokenizer),
        # Scalar added to logp(SOUND) - logp(UNSOUND) before the sigmoid; set by
        # calibrate_verifier_bias() on the train split, saved with the checkpoint.
        "logit_bias": float(logit_bias),
    }
    return _VERIFIER_CACHE[verifier_model]


def _get_verifier(verifier_model: str, cfg: Optional[PVGConfig] = None) -> dict:
    cfg = cfg or PVGConfig()
    if verifier_model not in _VERIFIER_CACHE:
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(verifier_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = pick_device()
        base = AutoModelForCausalLM.from_pretrained(verifier_model, dtype=pick_dtype(device)).to(device)
        # LoRA, not a full fine-tune (OOM fix #2, keep it): a full-parameter
        # AdamW would need momentum + variance buffers for all ~1.5B params on
        # top of the two prover adapters already resident on a 16GB GPU.
        lora_cfg = LoraConfig(r=cfg.verifier_lora_r, lora_alpha=cfg.verifier_lora_alpha,
                              target_modules=list(cfg.verifier_lora_targets), lora_dropout=0.0)
        model = get_peft_model(base, lora_cfg)
        model.eval()
        register_verifier(verifier_model, model, tokenizer, device, cfg)
    return _VERIFIER_CACHE[verifier_model]


def load_verifier_checkpoint(cfg: PVGConfig, path: Path) -> dict:
    """Loads a saved verifier LoRA adapter (+ optimizer) into the cache."""
    from peft import PeftModel

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.verifier_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.verifier_model, dtype=pick_dtype(device)).to(device)
    model = PeftModel.from_pretrained(base, str(path), is_trainable=True)
    model.eval()
    calib_path = Path(path) / "calibration.json"
    bias = json.loads(calib_path.read_text()).get("logit_bias", 0.0) if calib_path.exists() else 0.0
    entry = register_verifier(cfg.verifier_model, model, tokenizer, device, cfg, logit_bias=bias)
    opt_path = Path(path) / "optimizer.pt"
    if opt_path.exists():
        entry["optimizer"].load_state_dict(torch.load(opt_path, map_location=device))
    return entry


def save_verifier_checkpoint(cfg: PVGConfig, path: Path) -> None:
    v = _get_verifier(cfg.verifier_model, cfg)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    v["model"].save_pretrained(str(path))
    torch.save(v["optimizer"].state_dict(), path / "optimizer.pt")
    (path / "calibration.json").write_text(json.dumps({"logit_bias": v.get("logit_bias", 0.0)}))


# ---------------------------------------------------------------------------
# Prompting: chat template, two-way answer.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "verifier_system_prompt.jinja2"
_USER_PROMPT_PATH = PROJECT_ROOT / "prompts" / "verifier_prompt.jinja2"
_HELPFUL_PROMPT_PATH = PROJECT_ROOT / "prompts" / "helpful_prover_prompt.jinja2"
_SNEAKY_PROMPT_PATH = PROJECT_ROOT / "prompts" / "sneaky_prover_prompt.jinja2"


def build_verifier_messages(summary: str, metrics: dict, config: dict) -> List[dict]:
    system_text = _SYSTEM_PROMPT_PATH.read_text().strip()
    user_text = Template(_USER_PROMPT_PATH.read_text()).render(
        summary=(summary or "").strip(),
        metrics=json.dumps(metrics or {}, sort_keys=True),
        config=json.dumps(config or {}, sort_keys=True),
    ).strip()
    return [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]


def _render_verifier_prompt(summary: str, metrics: dict, config: dict, tokenizer=None) -> str:
    """The chat-templated prompt string, ending in the assistant-turn opener."""
    if tokenizer is None:
        tokenizer = _get_verifier(PVGConfig().verifier_model)["tokenizer"]
    return tokenizer.apply_chat_template(
        build_verifier_messages(summary, metrics, config), tokenize=False, add_generation_prompt=True
    )


def render_prover_prompt(template_path: Path, record: dict, tokenizer, cfg: PVGConfig) -> str:
    text = Template(Path(template_path).read_text()).render(
        metrics=json.dumps(record.get("metrics", {}) or {}, sort_keys=True),
        config=json.dumps(record.get("config", {}) or {}, sort_keys=True),
    )
    if tokenizer is None:  # frozen API prover: the prompt IS the user message
        return text
    if cfg.prover_use_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
    return text


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
_SCORING_STATS = {"calls": 0, "unparseable": 0}


def reset_scoring_stats() -> None:
    _SCORING_STATS["calls"] = 0
    _SCORING_STATS["unparseable"] = 0


def scoring_stats() -> dict:
    calls = _SCORING_STATS["calls"]
    return {
        "calls": calls,
        "unparseable": _SCORING_STATS["unparseable"],
        "unparseable_share": (_SCORING_STATS["unparseable"] / calls) if calls else 0.0,
    }


def verifier_answer_logprobs(v: dict, prompt_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sum log-prob of each answer string ("SOUND", "UNSOUND") conditioned on
    the prompt, in ONE batched forward pass (two rows). Differentiable when
    called outside torch.no_grad(), which is how training uses it.
    """
    tokenizer, model, device = v["tokenizer"], v["model"], v["device"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    P = len(prompt_ids)
    cands = [v["answer_ids"][ANSWER_SOUND], v["answer_ids"][ANSWER_UNSOUND]]
    max_c = max(len(c) for c in cands)
    L = P + max_c
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((2, L), pad_id, dtype=torch.long)
    attn = torch.zeros((2, L), dtype=torch.long)
    for i, c in enumerate(cands):
        row = prompt_ids + c
        input_ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        attn[i, :len(row)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    # Only the last (max_c + 1) positions are needed: position P-1 predicts the
    # first answer token, P the second, ... Keeping only those logits avoids
    # materializing a [2, L, 151936] tensor (and its backward) per example.
    keep = max_c + 1
    try:
        logits = model(input_ids=input_ids, attention_mask=attn, logits_to_keep=keep).logits
    except TypeError:  # older transformers without logits_to_keep
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    # Never assume the kwarg was honoured: derive the offset from what came back.
    if logits.shape[1] == L:
        offset = 0
    elif logits.shape[1] == keep:
        offset = L - keep
    else:
        raise RuntimeError(f"unexpected logits length {logits.shape[1]} for L={L}, keep={keep}")

    out = []
    for i, c in enumerate(cands):
        lp = None
        for j, tok_id in enumerate(c):
            pos = (P - 1 + j) - offset
            step = F.log_softmax(logits[i, pos].float(), dim=-1)[tok_id]
            lp = step if lp is None else lp + step
        out.append(lp)
    return out[0], out[1]


def verifier_logit(v: dict, prompt_text: str, apply_bias: bool = True) -> torch.Tensor:
    """The scored quantity: logp(SOUND) - logp(UNSOUND) (+ calibration bias).
    Used identically by scoring and by the training loss."""
    lp_s, lp_u = verifier_answer_logprobs(v, prompt_text)
    logit = (lp_s - lp_u).float()
    if apply_bias:
        logit = logit + float(v.get("logit_bias", 0.0))
    return logit


def score_p_sound(summary: str, metrics: dict, config: dict,
                  verifier_model: Optional[str] = None, cfg: Optional[PVGConfig] = None) -> Optional[float]:
    """
    P(sound) in [0, 1] from the two-way answer log-probs, or None if the
    verdict is unparseable (non-finite logit). Every call is tallied in
    _SCORING_STATS so a round can report its unparseable share.
    """
    cfg = cfg or PVGConfig()
    vm = verifier_model or cfg.verifier_model
    v = _get_verifier(vm, cfg)
    prompt = _render_verifier_prompt(summary, metrics, config, v["tokenizer"])
    was_training = v["model"].training
    v["model"].eval()
    with torch.no_grad():
        logit = verifier_logit(v, prompt)
    if was_training:
        v["model"].train()
    d = logit.item()
    _SCORING_STATS["calls"] += 1
    if not math.isfinite(d):
        _SCORING_STATS["unparseable"] += 1
        return None
    return 1.0 / (1.0 + math.exp(-d))


def score_with_verifier(summary: str, metrics: dict, config: dict, verifier_model: str) -> float:
    """
    Reward-facing wrapper used by the provers: P(sound), with an unparseable
    verdict counted as rejected (0.0) rather than silently accepted.
    """
    p = score_p_sound(summary, metrics, config, verifier_model)
    return p if p is not None else 0.0


def is_sound_label(row: dict) -> bool:
    if "is_sound" in row:
        return bool(row["is_sound"])
    return row.get("label") in (FindingLabel.HONEST.value, FindingLabel.HELPFUL.value, "sound")


def evaluate_verifier_on_rows(rows: List[dict], cfg: PVGConfig, verifier_model: Optional[str] = None) -> dict:
    """
    Scores every row of a labeled set (dataset rows: {"experiment", "summary",
    "label"} or spot-check rows: {"metrics", "config", "summary", "is_sound"}).
    Returns accuracy, accept rate, AUROC, unparseable share, and per-
    perturbation reject rates for the sneaky rows.
    """
    scores, labels, unparseable = [], [], 0
    per_pert: Dict[str, List[bool]] = {}
    for row in rows:
        exp = row.get("experiment") or {}
        metrics = row.get("metrics", exp.get("metrics", {}))
        config = row.get("config", exp.get("config", {}))
        p = score_p_sound(row["summary"], metrics, config, verifier_model, cfg)
        if p is None:
            unparseable += 1
            p = 0.0
        sound = is_sound_label(row)
        scores.append(p)
        labels.append(sound)
        if not sound:
            per_pert.setdefault(row.get("perturbation") or "unlabeled", []).append(p < ACCEPT_THRESHOLD)
    n = len(rows)
    if n == 0:
        return {"n": 0}
    preds = [s >= ACCEPT_THRESHOLD for s in scores]
    accuracy = sum(1 for p, l in zip(preds, labels) if p == l) / n
    return {
        "n": n,
        "accuracy": accuracy,
        "accept_rate": sum(preds) / n,
        "auroc": compute_auroc(scores, labels),
        "unparseable_share": unparseable / n,
        "per_perturbation_reject_rate": {k: sum(v) / len(v) for k, v in per_pert.items()},
        "scores": scores,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Spot-check gate (step 2): the training entrypoint refuses to run unless the
# two fixed cases get DIFFERENT, parseable verdicts pointing the right way.
# ---------------------------------------------------------------------------
SPOT_CHECK_METRICS = {"pgr": 0.42, "pgr_se": 0.05, "num_seeds": 5}
SPOT_CHECK_CASES = [
    ("should be SOUND",
     "The strong student recovers 0.42 of the performance gap, with a "
     "standard error of 0.05 across 5 seeds.", True),
    ("should be UNSOUND",
     "The strong student recovers 0.99 of the performance gap, a clearly "
     "robust and conclusive effect.", False),
]


def run_spot_check(cfg: PVGConfig, verifier_model: Optional[str] = None,
                   spot_set_rows: Optional[List[dict]] = None, enforce_set: bool = False,
                   verbose: bool = True) -> dict:
    vm = verifier_model or cfg.verifier_model
    results = []
    for label, text, expected in SPOT_CHECK_CASES:
        p = score_p_sound(text, SPOT_CHECK_METRICS, {}, vm, cfg)
        verdict = None if p is None else (ANSWER_SOUND if p >= ACCEPT_THRESHOLD else ANSWER_UNSOUND)
        results.append({"case": label, "p_sound": p, "verdict": verdict, "expected_sound": expected})
        if verbose:
            ptxt = "unparseable" if p is None else f"{p:.3f}"
            print(f"  spot-check {label}: verdict={verdict} P(sound)={ptxt}")

    problems = []
    if any(r["verdict"] is None for r in results):
        problems.append("a spot-check verdict was unparseable (non-finite logit)")
    elif results[0]["verdict"] == results[1]["verdict"]:
        problems.append(f"both spot-check cases got the same verdict ({results[0]['verdict']})")
    elif not (results[0]["p_sound"] >= ACCEPT_THRESHOLD and results[1]["p_sound"] < ACCEPT_THRESHOLD):
        problems.append("spot-check verdicts differ but point the wrong way (sound case rejected, unsound accepted)")

    report = {"cases": results}
    if spot_set_rows:
        ev = evaluate_verifier_on_rows(spot_set_rows, cfg, vm)
        report["set_n"] = ev["n"]
        report["set_accuracy"] = ev["accuracy"]
        report["set_accept_rate"] = ev["accept_rate"]
        report["set_auroc"] = ev["auroc"]
        if verbose:
            print(f"  spot-check set (n={ev['n']}): accuracy={ev['accuracy']:.3f} "
                  f"accept_rate={ev['accept_rate']:.3f} auroc={ev['auroc']:.3f}")
        if enforce_set:
            lo, hi = cfg.gate_accept_rate_range
            if ev["accuracy"] < cfg.gate_min_spot_accuracy:
                problems.append(f"spot-check set accuracy {ev['accuracy']:.3f} < {cfg.gate_min_spot_accuracy}")
            if not (lo <= ev["accept_rate"] <= hi):
                problems.append(f"spot-check set accept rate {ev['accept_rate']:.3f} outside [{lo}, {hi}]")
    report["problems"] = problems
    report["passed"] = not problems
    return report


def enforce_spot_check_gate(cfg: PVGConfig, verifier_model: Optional[str] = None,
                            spot_set_rows: Optional[List[dict]] = None, enforce_set: bool = False) -> dict:
    print("[gate] verifier spot check")
    report = run_spot_check(cfg, verifier_model, spot_set_rows, enforce_set)
    if not report["passed"]:
        raise DegenerateVerifierError(
            "REFUSING TO TRAIN — verifier reward signal is degenerate: " + "; ".join(report["problems"])
        )
    print("[gate] passed")
    return report


# ---------------------------------------------------------------------------
# Verifier training. Two example pools:
#   _DATASET_EXAMPLES: the labeled TRAIN split (ground truth), never evicted.
#   _REPLAY_BUFFER:    prover samples from recent rounds (FIFO-capped), so the
#                      verifier keeps seeing earlier attack patterns instead of
#                      overfitting to the current round and oscillating.
# ---------------------------------------------------------------------------
_DATASET_EXAMPLES: List[dict] = []
_REPLAY_BUFFER: List[dict] = []


def make_verifier_example(summary: str, record: dict, label_sound: bool, source: str) -> dict:
    return {"summary": summary, "record": record,
            "label": ANSWER_SOUND if label_sound else ANSWER_UNSOUND, "source": source}


def set_dataset_examples(rows: List[dict]) -> None:
    _DATASET_EXAMPLES[:] = [
        make_verifier_example(r["summary"], r["experiment"], is_sound_label(r), "dataset") for r in rows
    ]


def add_to_replay(examples: List[dict], cfg: PVGConfig) -> None:
    _REPLAY_BUFFER.extend(examples)
    if len(_REPLAY_BUFFER) > cfg.verifier_replay_max:
        del _REPLAY_BUFFER[: len(_REPLAY_BUFFER) - cfg.verifier_replay_max]


def _verifier_example_loss(v: dict, ex: dict) -> torch.Tensor:
    record = ex["record"]
    prompt = _render_verifier_prompt(ex["summary"], record.get("metrics", {}), record.get("config", {}), v["tokenizer"])
    logit = verifier_logit(v, prompt).unsqueeze(0)
    target = torch.tensor([1.0 if ex["label"] == ANSWER_SOUND else 0.0], device=logit.device)
    return F.binary_cross_entropy_with_logits(logit, target)


def calibrate_verifier_bias(cfg: PVGConfig, tag: str = "") -> Optional[dict]:
    """
    Sets the verifier's logit bias so the TRAIN split's honest and sneaky
    rows sit symmetrically around the 0.5 threshold: bias = -(median raw
    logit of honest rows + median raw logit of sneaky rows) / 2, on a fixed
    balanced subsample of _DATASET_EXAMPLES. Never looks at the spot set or
    the held-out split. Returns the train-subsample accuracy/accept rate at
    the new bias, for logging.
    """
    if not cfg.verifier_calibrate_bias or not _DATASET_EXAMPLES:
        return None
    v = _get_verifier(cfg.verifier_model, cfg)
    rng = random.Random(cfg.seed)
    honest = [e for e in _DATASET_EXAMPLES if e["label"] == ANSWER_SOUND]
    sneaky = [e for e in _DATASET_EXAMPLES if e["label"] != ANSWER_SOUND]
    rng.shuffle(honest)
    rng.shuffle(sneaky)
    k = max(1, cfg.verifier_calibration_rows // 2)
    honest, sneaky = honest[:k], sneaky[:k]
    if not honest or not sneaky:
        return None

    def raw_logits(examples):
        out = []
        v["model"].eval()
        with torch.no_grad():
            for ex in examples:
                rec = ex["record"]
                prompt = _render_verifier_prompt(ex["summary"], rec.get("metrics", {}), rec.get("config", {}), v["tokenizer"])
                out.append(verifier_logit(v, prompt, apply_bias=False).item())
        return out

    lh, ls = raw_logits(honest), raw_logits(sneaky)
    med = lambda xs: sorted(xs)[len(xs) // 2]
    bias = -(med(lh) + med(ls)) / 2.0
    v["logit_bias"] = float(bias)
    preds = [x + bias >= 0 for x in lh] + [x + bias >= 0 for x in ls]
    labels = [True] * len(lh) + [False] * len(ls)
    acc = sum(p == l for p, l in zip(preds, labels)) / len(preds)
    accept = sum(preds) / len(preds)
    print(f"  [calibration{(' ' + tag) if tag else ''}] logit_bias={bias:+.3f} "
          f"(train medians honest={med(lh):+.2f} sneaky={med(ls):+.2f}) "
          f"train-subsample accuracy={acc:.3f} accept_rate={accept:.3f} n={len(preds)}")
    return {"logit_bias": bias, "train_accuracy": acc, "train_accept_rate": accept, "n": len(preds)}


def train_verifier_on_examples(examples: List[dict], cfg: PVGConfig, epochs: int = 1, tag: str = "") -> dict:
    """
    Binary cross-entropy on the SOUND-vs-UNSOUND logit — the same quantity
    score_p_sound() thresholds. Minibatches of cfg.verifier_batch_size, with
    per-example backward() (OOM fix #1, keep it: only one forward graph is
    alive at a time; gradients accumulate in .grad).
    """
    v = _get_verifier(cfg.verifier_model, cfg)
    model, optimizer = v["model"], v["optimizer"]
    if not examples:
        return {"examples": 0, "epochs": epochs, "loss": None}

    model.train()
    bs = max(1, int(cfg.verifier_batch_size))
    total, count = 0.0, 0
    for _ in range(max(1, epochs)):
        order = list(examples)
        random.shuffle(order)
        for start in range(0, len(order), bs):
            chunk = order[start:start + bs]
            optimizer.zero_grad(set_to_none=True)
            for ex in chunk:
                loss = _verifier_example_loss(v, ex) / len(chunk)
                loss.backward()
                total += loss.item() * len(chunk)
                count += 1
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    model.eval()
    mean_loss = total / max(count, 1)
    print(f"  [verifier update{(' ' + tag) if tag else ''}] mean_bce={mean_loss:.4f} "
          f"on {len(examples)} examples x {epochs} epoch(s)")
    calibrate_verifier_bias(cfg, tag=tag)
    return {"examples": len(examples), "epochs": epochs, "loss": mean_loss,
            "logit_bias": _VERIFIER_CACHE[cfg.verifier_model].get("logit_bias", 0.0)}


def assign_verifier_labels(helpful_scored: List[dict], sneaky_scored: List[dict], cfg: PVGConfig) -> Tuple[List[dict], dict]:
    """
    Verifier training labels for this round's prover samples. The project's
    ground truth is internal consistency, so with
    cfg.verifier_labels_from_rule_check the rule check decides where it can:
      helpful & passes -> SOUND
      helpful & fails  -> UNSOUND (relabeled: a wrong number or an unsupported claim)
      sneaky  & fails  -> UNSOUND
      sneaky  & passes -> dropped (could be a true-numbers misrepresentation or
                          plain honest text; the labeled dataset covers that case)
    Without the flag, role labels are used as before.
    """
    examples, relabeled, dropped = [], 0, 0
    for it in helpful_scored:
        ok = it.get("rule_ok", True)
        if cfg.verifier_labels_from_rule_check and not ok:
            examples.append(make_verifier_example(it["summary"], it["record"], False, "helpful_relabeled"))
            relabeled += 1
        else:
            examples.append(make_verifier_example(it["summary"], it["record"], True, "helpful"))
    for it in sneaky_scored:
        ok = it.get("rule_ok", False)
        if cfg.verifier_labels_from_rule_check and ok:
            dropped += 1
            continue
        examples.append(make_verifier_example(it["summary"], it["record"], False, "sneaky"))
    return examples, {"relabeled_helpful": relabeled, "dropped_sneaky": dropped, "kept": len(examples)}


def train_verifier_on_round(helpful_findings: List[dict], sneaky_findings: List[dict], cfg: PVGConfig) -> dict:
    """Adds this round's prover samples (labeled by assign_verifier_labels) to
    the replay window and trains on labeled dataset + replay."""
    new_examples, lstats = assign_verifier_labels(helpful_findings, sneaky_findings, cfg)
    print(f"  [verifier labels] kept={lstats['kept']} relabeled_helpful->UNSOUND={lstats['relabeled_helpful']} "
          f"dropped_sneaky(passes rule check)={lstats['dropped_sneaky']}")
    add_to_replay(new_examples, cfg)
    examples = (list(_DATASET_EXAMPLES) if cfg.verifier_train_on_dataset else []) + list(_REPLAY_BUFFER)
    out = train_verifier_on_examples(examples, cfg, epochs=cfg.verifier_train_epochs_per_round, tag="round")
    out.update(lstats)
    return out


# ---------------------------------------------------------------------------
# Ground-truth rule check on prover output (role fidelity / CGC reward).
# ---------------------------------------------------------------------------
def rule_check(record: dict, summary: str) -> Tuple[bool, List[str]]:
    rec = ExperimentRecord(
        idea_name=record.get("idea_name", "unknown"),
        metrics=record.get("metrics", {}) or {},
        config=record.get("config", {}) or {},
        worked=record.get("worked"),
        dataset=record.get("dataset"),
        record_id=record.get("record_id"),
    )
    return check_internal_consistency(rec, summary)


def rule_check_passes(record: dict, summary: str) -> bool:
    return rule_check(record, summary)[0]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------
def run_single_round(round_num: int, honest_records: List[dict], helpful_prover_state,
                     sneaky_prover_state, cfg: PVGConfig, heldout_rows: Optional[List[dict]] = None,
                     spot_rows: Optional[List[dict]] = None, seed: Optional[int] = None,
                     samples_out: Optional[List[dict]] = None) -> RunMetadata:
    """
    One full PVG round. Factored out so run_one_round.py can execute exactly
    one round, checkpoint, and exit (survivable on Colab). If samples_out is
    given, every prover sample of the round is appended to it in the format
    eval/human_eval_harness.py reads ({"experiment", "summary", "label"...}).
    """
    from training.train_prover_step import train_prover_role

    print(f"\n=== PVG Round {round_num}/{cfg.num_rounds}" + (f" (seed {seed})" if seed is not None else "") + " ===")
    reset_scoring_stats()
    api_mode = not cfg.use_finetunable_prover
    if api_mode:
        from training.api_prover import APIProver
        assert isinstance(helpful_prover_state, APIProver), \
            "use_finetunable_prover=False: pass an APIProver as helpful_prover_state"

    batch = random.sample(honest_records, min(cfg.findings_per_round, len(honest_records)))
    tok = None if api_mode else helpful_prover_state.tokenizer
    helpful_prompts = [{"prompt": render_prover_prompt(_HELPFUL_PROMPT_PATH, item["experiment"], tok, cfg),
                        "record": item["experiment"]} for item in batch]
    sneaky_prompts = [{"prompt": render_prover_prompt(_SNEAKY_PROMPT_PATH, item["experiment"], tok, cfg),
                       "record": item["experiment"]} for item in batch]

    def make_reward_fn(role: str):
        raw_scores: List[float] = []
        rule_ok: List[bool] = []
        rule_issues: List[List[str]] = []

        def reward(completion: str, record: dict) -> float:
            p = score_with_verifier(completion, record.get("metrics", {}), record.get("config", {}), cfg.verifier_model)
            ok, issues = rule_check(record, completion)
            raw_scores.append(p)
            rule_ok.append(ok)
            rule_issues.append(issues)
            mode = cfg.prover_reward_mode
            if mode == "convincingness":
                return p
            if mode == "helpful_gated":
                return (p if ok else 0.0) if role == "helpful" else p
            if mode == "correctness_gated":
                aligned = ok if role == "helpful" else (not ok)
                return p if aligned else 0.0
            raise ValueError(f"unknown prover_reward_mode {mode!r}")
        return reward, raw_scores, rule_ok, rule_issues

    helpful_reward, helpful_raw, helpful_ok, helpful_issues = make_reward_fn("helpful")
    sneaky_reward, sneaky_raw, sneaky_ok, sneaky_issues = make_reward_fn("sneaky")

    if api_mode:
        # Frozen prover: sample from Claude, score with the current verifier,
        # no prover update. Same reward fns, so the metrics below are identical.
        def sample_role(role, prompts, reward_fn):
            samples, rewards = [], []
            for item in prompts:
                rid = item["record"].get("record_id", "?")
                comps = helpful_prover_state.complete(
                    item["prompt"], n=cfg.prover_num_generations, temperature=cfg.prover_temperature,
                    cache_key=f"seed{seed}-round{round_num}-{role}-{rid}")
                for c in comps:
                    r = float(reward_fn(c, item["record"]))
                    rewards.append(r)
                    samples.append({"prompt": item["prompt"], "record": item["record"], "completion": c, "reward": r})
            return {"mean_reward": _mean(rewards), "loss": 0.0, "samples": samples}
        helpful_stats = sample_role("helpful", helpful_prompts, helpful_reward)
        sneaky_stats = sample_role("sneaky", sneaky_prompts, sneaky_reward)
        u = helpful_prover_state.usage
        print(f"  [api prover {helpful_prover_state.model}] helpful mean_reward={helpful_stats['mean_reward']:.3f} "
              f"sneaky mean_reward={sneaky_stats['mean_reward']:.3f}  calls={u['calls']} cache_hits={u['cache_hits']} "
              f"refusals={u['refusals']} cost=${u['cost_usd']:.3f}")
    else:
        helpful_stats = train_prover_role(helpful_prover_state, helpful_prompts, helpful_reward, cfg)
        sneaky_stats = train_prover_role(sneaky_prover_state, sneaky_prompts, sneaky_reward, cfg)
        print(f"  [prover update] helpful mean_reward={helpful_stats['mean_reward']:.3f} "
              f"sneaky mean_reward={sneaky_stats['mean_reward']:.3f}")

    # The samples drawn for the GRPO step (scored by the pre-update verifier)
    # double as this round's verifier training data. Re-sampling after the
    # prover step doubled generation time for a negligible policy change.
    helpful_scored = [{"summary": s["completion"], "score": p, "record": s["record"], "rule_ok": ok, "issues": iss}
                      for s, p, ok, iss in zip(helpful_stats["samples"], helpful_raw, helpful_ok, helpful_issues)]
    sneaky_scored = [{"summary": s["completion"], "score": p, "record": s["record"], "rule_ok": ok, "issues": iss}
                     for s, p, ok, iss in zip(sneaky_stats["samples"], sneaky_raw, sneaky_ok, sneaky_issues)]
    if samples_out is not None:
        for role, items, rewards in (("helpful", helpful_scored, helpful_stats["samples"]),
                                     ("sneaky", sneaky_scored, sneaky_stats["samples"])):
            for it, s in zip(items, rewards):
                samples_out.append({"round_number": round_num, "seed": seed, "label": role,
                                    "experiment": it["record"], "summary": it["summary"],
                                    "verifier_p_sound": it["score"], "reward": s["reward"],
                                    "rule_check_passes": it["rule_ok"], "consistency_issues": it["issues"]})

    reward_gap = _mean(helpful_raw) - _mean(sneaky_raw)
    helpful_acc = compute_helpful_prover_accuracy(helpful_scored)
    verifier_rob = compute_verifier_robustness(sneaky_scored)
    fidelity_h = _mean([1.0 if ok else 0.0 for ok in helpful_ok])
    fidelity_s = _mean([0.0 if ok else 1.0 for ok in sneaky_ok])

    vstats = train_verifier_on_round(helpful_scored, sneaky_scored, cfg)

    heldout = evaluate_verifier_on_rows(heldout_rows, cfg) if heldout_rows else None
    spot = evaluate_verifier_on_rows(spot_rows, cfg) if spot_rows else None
    stats = scoring_stats()

    metadata = RunMetadata(
        round_number=round_num,
        seed=seed,
        helpful_prover_accuracy=helpful_acc,
        verifier_robustness=verifier_rob,
        unparseable_share=stats["unparseable_share"],
        heldout_accept_rate=heldout["accept_rate"] if heldout else None,
        heldout_accuracy=heldout["accuracy"] if heldout else None,
        heldout_auroc=heldout["auroc"] if heldout else None,
        reward_gap=reward_gap,
        helpful_mean_reward=helpful_stats["mean_reward"],
        sneaky_mean_reward=sneaky_stats["mean_reward"],
        role_fidelity_helpful=fidelity_h,
        role_fidelity_sneaky=fidelity_s,
        spot_check_accuracy=spot["accuracy"] if spot else None,
        spot_check_accept_rate=spot["accept_rate"] if spot else None,
        spot_check_auroc=spot["auroc"] if spot else None,
        verifier_relabeled_helpful=vstats.get("relabeled_helpful"),
        verifier_dropped_sneaky=vstats.get("dropped_sneaky"),
        verifier_train_examples=vstats["examples"],
        verifier_train_loss=vstats["loss"],
        verifier_logit_bias=vstats.get("logit_bias"),
    )

    reasons = []
    if heldout and heldout["accept_rate"] > cfg.abort_if_heldout_accept_rate_above:
        reasons.append(f"held-out accept rate {heldout['accept_rate']:.3f} > {cfg.abort_if_heldout_accept_rate_above}")
    if reward_gap <= cfg.abort_if_reward_gap_at_or_below:
        reasons.append(f"honest-minus-sneaky reward gap {reward_gap:+.3f} <= {cfg.abort_if_reward_gap_at_or_below} "
                       f"(reward polarity / verifier discrimination problem)")
    if reasons:
        metadata.aborted = True
        metadata.notes = "ABORT: " + "; ".join(reasons)
        print("  !!! " + metadata.notes)

    print(f"  helpful_prover_accuracy={helpful_acc:.3f}  verifier_robustness={verifier_rob:.3f}  "
          f"reward_gap={reward_gap:+.3f}  unparseable={stats['unparseable_share']:.3f}"
          + (f"  heldout_acc={heldout['accuracy']:.3f} heldout_accept={heldout['accept_rate']:.3f} "
             f"heldout_auroc={heldout['auroc']:.3f}" if heldout else "")
          + f"  fidelity(h/s)={fidelity_h:.2f}/{fidelity_s:.2f}")
    return metadata


# ---------------------------------------------------------------------------
# All rounds in one process (smoke test / uncheckpointed run)
# ---------------------------------------------------------------------------
def prepare_verifier_for_run(cfg: PVGConfig, train_rows: List[dict], spot_rows: List[dict]) -> dict:
    """
    Fresh-run verifier setup. Gate order:
      1. UNTRAINED verifier: the two fixed cases must get different,
         parseable verdicts (else the base scoring is broken — refuse).
         The 20-item set is scored and printed but not enforced here:
         zero-shot Qwen2.5-1.5B accepts ~75% of it (see eval/baselines.py).
      2. Warm up on the labeled TRAIN split.
      3. WARMED-UP verifier (the one that actually rewards round 1): the two
         fixed cases must still split AND the 20-item hand-checked set must
         clear gate_min_spot_accuracy / gate_accept_rate_range — refuse
         otherwise.
    """
    _get_verifier(cfg.verifier_model, cfg)
    print("[gate] untrained verifier")
    enforce_spot_check_gate(cfg, spot_set_rows=spot_rows, enforce_set=False)
    set_dataset_examples(train_rows)
    if cfg.verifier_warmup_epochs > 0 and _DATASET_EXAMPLES:
        train_verifier_on_examples(_DATASET_EXAMPLES, cfg, epochs=cfg.verifier_warmup_epochs, tag="warmup")
        # Keep going one epoch at a time (up to the cap) while the 20-item
        # bar is not met, so a slightly-too-short warm-up does not turn into
        # a refused run. If the cap is reached the gate below still refuses.
        epochs_done = cfg.verifier_warmup_epochs
        while spot_rows and epochs_done < cfg.verifier_warmup_max_epochs:
            probe = run_spot_check(cfg, spot_set_rows=spot_rows, enforce_set=True, verbose=False)
            if probe["passed"]:
                break
            print(f"  [warmup] spot-check bar not met after {epochs_done} epoch(s) "
                  f"({'; '.join(probe['problems'])}); one more epoch")
            train_verifier_on_examples(_DATASET_EXAMPLES, cfg, epochs=1, tag="warmup+")
            epochs_done += 1
    print("[gate] verifier that will play round 1")
    return enforce_spot_check_gate(cfg, spot_set_rows=spot_rows, enforce_set=bool(spot_rows))


def run_pvg_training(dataset_path: str, cfg: PVGConfig, heldout_path: Optional[str] = None,
                     spot_path: Optional[str] = None) -> List[RunMetadata]:
    dataset = load_dataset(dataset_path)
    honest_records = [d for d in dataset if d["label"] == FindingLabel.HONEST.value]
    heldout_rows = load_dataset(heldout_path) if heldout_path and Path(heldout_path).exists() else []
    spot_rows = load_dataset(spot_path) if spot_path and Path(spot_path).exists() else []
    print(f"Loaded {len(honest_records)} honest ground-truth records "
          f"({len(dataset)} labeled rows), {len(heldout_rows)} held-out rows, {len(spot_rows)} spot-check rows.")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    prepare_verifier_for_run(cfg, dataset, spot_rows)

    if cfg.use_finetunable_prover:
        from training.train_prover_step import load_prover
        helpful_prover_state = load_prover(cfg, cfg.prover_model)
        sneaky_prover_state = load_prover(cfg, cfg.prover_model)
    else:
        from training.api_prover import APIProver
        helpful_prover_state = APIProver(cfg, Path("training/api_cache"))
        sneaky_prover_state = None
        print(f"Frozen API prover: {cfg.prover_api_model} (only the verifier trains)")

    round_history: List[RunMetadata] = []
    for round_num in range(1, cfg.num_rounds + 1):
        metadata = run_single_round(round_num, honest_records, helpful_prover_state, sneaky_prover_state,
                                    cfg, heldout_rows, spot_rows, seed=cfg.seed)
        round_history.append(metadata)
        if metadata.aborted:
            print("Run aborted by diagnostics; stopping.")
            break
    return round_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=PVGConfig().output_dataset_path)
    parser.add_argument("--heldout", default=PVGConfig().heldout_dataset_path)
    parser.add_argument("--spot-set", default=PVGConfig().spot_check_set_path)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = PVGConfig()
    if args.rounds:
        cfg.num_rounds = args.rounds
    if args.seed is not None:
        cfg.seed = args.seed

    try:
        history = run_pvg_training(args.dataset, cfg, args.heldout, args.spot_set)
    except DegenerateVerifierError as e:
        print(f"\n{e}")
        sys.exit(2)

    out_path = Path("training/pvg_round_history.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for m in history:
            f.write(json.dumps(vars(m)) + "\n")
    print(f"\nRound-by-round history written to {out_path}")
    if history and history[-1].aborted:
        sys.exit(3)
