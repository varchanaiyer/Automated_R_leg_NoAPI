"""
Fine-tunes a STAND-IN prover model (not the real AAR — see LIMITATIONS.md #7)
via a GRPO-style policy update, using the verifier's score as reward.

This is what makes "legibility training" literally true for the prover, not
just the verifier: the prover's own weights change, round over round, to
produce output the verifier scores as sound (helpful role) or as
convincingly-sound-but-actually-unsound (sneaky role).

Method (mirrors GRPO):
  1. For each experiment record, sample `num_generations` findings from the
     CURRENT prover policy (temperature > 0, so they differ).
  2. Score each with the current verifier -> reward.
  3. Group-relative advantage: reward - group_mean (Dr.GRPO style; dividing
     by the group std as well is available via cfg.grpo_normalize_std, but
     with a continuous P(sound) reward it turns tiny within-group noise into
     O(1) advantages).
  4. Policy-gradient update per group: maximize advantage-weighted
     log-probability of the generated tokens, with a KL penalty against the
     frozen base model (the same model with the LoRA adapter disabled).

One optimizer step PER GROUP (per record), not one per round: with a single
step per round at lr 1e-5 the prover was effectively frozen for the whole
10-round run.

This trains ONE role at a time (helpful or sneaky) — call it once for each
per round from pvg_loop.py, since they optimize toward different targets.

Two CUDA-OOM fixes live here and must be preserved:
  - per-completion backward() (never accumulate a live graph across the loop)
  - no separate frozen reference copy; the KL reference is the same model
    with the adapter disabled.
"""
from dataclasses import dataclass
from typing import Callable, List
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PVGConfig


@dataclass
class ProverState:
    """Holds the loaded model/tokenizer/optimizer across rounds, so the
    prover's weights persist and accumulate updates round over round."""
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    optimizer: torch.optim.Optimizer
    device: str


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _dtype(device: str):
    return torch.bfloat16 if device in ("cuda", "mps") else torch.float32


def load_prover(cfg: PVGConfig, base_model_name: str) -> ProverState:
    """
    Loads the stand-in prover with a LoRA adapter (only the adapter trains).
    No separate frozen reference-model copy is loaded: the reference
    (pre-LoRA) log-probs needed for the KL penalty are computed by
    temporarily disabling the adapter on this SAME model
    (model.disable_adapter() in train_prover_role). This halves memory vs.
    a second full copy per role — required on a single 16GB GPU where
    helpful + sneaky roles and the verifier all need to fit at once.
    """
    from peft import LoraConfig, get_peft_model

    device = _device()
    dtype = _dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype).to(device)

    lora_cfg = LoraConfig(r=cfg.prover_lora_r, lora_alpha=cfg.prover_lora_alpha,
                          target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    model = get_peft_model(base, lora_cfg)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.prover_lr)

    return ProverState(model=model, tokenizer=tokenizer, optimizer=optimizer, device=device)


@torch.no_grad()
def generate_group(state: ProverState, prompt: str, cfg: PVGConfig) -> List[str]:
    """Samples cfg.prover_num_generations completions for one prompt. The
    prompt is expected to be already chat-templated (see pvg_loop.render_prover_prompt)."""
    state.model.eval()
    inputs = state.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(state.device)
    outputs = state.model.generate(
        **inputs,
        max_new_tokens=cfg.prover_max_new_tokens,
        do_sample=True,
        temperature=cfg.prover_temperature,
        top_p=0.95,
        num_return_sequences=cfg.prover_num_generations,
        pad_token_id=state.tokenizer.pad_token_id,
    )
    completions = state.tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return [c.strip() for c in completions]


def compute_sequence_logprob(model, tokenizer, prompt: str, completion: str, device: str) -> torch.Tensor:
    """Sum of token log-probs for `completion` conditioned on `prompt`, under `model`."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        return torch.zeros((), device=device)
    full_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)

    logits = model(full_ids).logits[:, :-1, :]
    targets = full_ids[:, 1:]

    # Slice to the completion BEFORE the softmax. A full-sequence
    # log_softmax materializes two [L, 151936] tensors first — ~600MB of
    # scratch at L=1024, thrown away immediately. That's where the OOM lands.
    logits = logits[:, prompt_len - 1:, :]
    targets = targets[:, prompt_len - 1:]

    # cross_entropy is fused — no separate log_softmax tensor. It returns
    # -log p(target), so negate.
    token_log_probs = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none"
    )
    return token_log_probs.sum()


def train_prover_role(
    state: ProverState,
    prompts_and_records: List[dict],
    score_fn: Callable[[str, dict], float],
    cfg: PVGConfig,
) -> dict:
    """
    One GRPO-style update pass for a single prover role (helpful or sneaky):
    one optimizer step per prompt group.

    Returns {"mean_reward", "loss", "samples"}; samples is the list of
    {"prompt", "record", "completion", "reward"} in generation order, so
    the caller can reuse them as verifier training data without re-sampling.
    """
    total_loss_value = 0.0
    all_rewards: List[float] = []
    samples: List[dict] = []
    trainable = [p for p in state.model.parameters() if p.requires_grad]
    G = max(cfg.prover_num_generations, 1)

    for item in prompts_and_records:
        prompt, record = item["prompt"], item["record"]
        completions = generate_group(state, prompt, cfg)

        rewards = [float(score_fn(c, record)) for c in completions]
        all_rewards.extend(rewards)
        for c, r in zip(completions, rewards):
            samples.append({"prompt": prompt, "record": record, "completion": c, "reward": r})

        rewards_t = torch.tensor(rewards, device=state.device, dtype=torch.float32)
        advantages = rewards_t - rewards_t.mean()  # group-relative advantage
        if cfg.grpo_normalize_std:
            advantages = advantages / (rewards_t.std() + 1e-4)
        if advantages.abs().max().item() < 1e-6:
            continue  # whole group scored identically: no learning signal

        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)
        for completion, advantage in zip(completions, advantages):
            if not completion.strip():
                continue
            logprob = compute_sequence_logprob(state.model, state.tokenizer, prompt, completion, state.device)
            with torch.no_grad(), state.model.disable_adapter():
                ref_logprob = compute_sequence_logprob(state.model, state.tokenizer, prompt, completion, state.device)
            kl = logprob - ref_logprob  # per-sequence KL proxy

            # Policy gradient loss: push up logprob of high-advantage completions,
            # down for low-advantage ones, penalized by drift from the reference.
            loss = -(advantage.detach() * logprob) + cfg.prover_kl_penalty * kl
            scaled_loss = loss / G
            # Backward NOW, per-completion, instead of accumulating a live
            # tensor across the loop — that kept every completion's forward
            # graph in memory until one final backward(), which is what OOM'd
            # a single 16GB GPU. Gradients accumulate in .grad across calls.
            scaled_loss.backward()
            total_loss_value += scaled_loss.item()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        state.optimizer.step()
        state.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release generation's reserved-but-unused KV cache

    return {
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "loss": total_loss_value,
        "samples": samples,
    }
