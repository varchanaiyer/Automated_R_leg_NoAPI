#!/usr/bin/env python3
"""
apply_fixes.py: every fix for the collapse, in one file.

Lives in the repo at  aar-legibility-pvg 5/apply_fixes.py.  In Colab, run it once right
after the clone cell and before the data cell:

    !python apply_fixes.py            # pin code + patch + rebuild dataset + model-written rows (GPU)
    !python apply_fixes.py --no-rows  # same, without generating model-written rows

Works whichever repo the notebook clones (gunasti2002 main or the varchanaiyer fork): it
first pins the project folder to the fork commit the fixes were written for (51940d1),
fetching it from GitHub if needed, then patches. Re-running gives the same result. If an
edit point has moved it stops with a message instead of half-applying.

What it fixes (see the run of 3 Oct: round-2 crash in all seeds, helpful fidelity 0.06-0.20,
logit bias drifting to +7.5, sneaky reward 0.000 from round 7, seed 42 aborted):

  0. Preflight: that run used commit 8616874 (26 Sep), which predates the class-balanced
     verifier and the helpful-prompt rules the notebook describes. This script refuses to
     patch code that old.
  1. Labels: helpful samples that fail the rule check are DROPPED instead of relabeled
     UNSOUND, and each round's prover samples are class-balanced by subsampling.
     (~110 of 128 helpful samples per round were becoming UNSOUND -> verifier learned
     "prover text = UNSOUND" -> round-2 crash and the logit-bias drift.)
  2. Model-written rows: Qwen-written SOUND write-ups plus perturbed UNSOUND twins of the
     same text, added to the verifier's labeled pool and held-out eval, so writing style
     no longer predicts the label.
  3. Real data: 5 more real records (blog headline + 4 human baselines from the
     researcher's own repo; the first records that did not work). 10 total, 3 land in held-out.
  4. Warm start: before round 1, both provers are fine-tuned on honest write-ups and start
     from that merged model, which is also their KL reference. (Base Qwen's helpful
     fidelity is 0.06-0.20; it needed 7 rounds of RL to reach ~0.6.)
  5. Role bonus: +0.1 reward when a sample does its role (helpful passes / sneaky fails the
     rule check), so neither prover's signal vanishes when the verifier rejects everything.

Every fix has a config switch; the script prints the overrides that restore the old behaviour.
"""
import argparse
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

BUILT_AGAINST = "51940d1"  # varchanaiyer/Automated_R_leg_NoAPI, branch fix/verifier-scoring-and-gates

SFT_WARMSTART = '"""\nSupervised warm start for BOTH provers (fix 4 of apply_fixes.py).\n\nProblem: the base Qwen2.5-1.5B helpful prover passes the rule check only\n6-20% of the time in round 1 (markdown lists, LaTeX, invented numbers, missing\nseed counts), so for several rounds most helpful text is unusable and the\ngame starts from a broken prover.\n\nFix: before round 1, fine-tune one LoRA on honest write-ups (the templated\nhonest rows + the model-written SOUND rows), merge it into the base weights,\nand save that merged model once per seed. Both the helpful and the sneaky\nprover then start from it, and its weights are the KL reference for the RL\nupdates (otherwise the KL penalty would pull the provers back to the\nunformatted base model). Starting both roles from the same model matters: if\nonly the helpful prover were warm-started, the two roles would differ in style\nand the verifier could separate them by style again.\n\nWrites <ckpt>/prover_sft_base/ (merged model + tokenizer + sft_report.json).\n"""\nimport json\nimport random\nfrom pathlib import Path\nfrom typing import List\n\nimport torch\n\nfrom config import PVGConfig\n\n\ndef sft_base_dir(ckpt: Path) -> Path:\n    return Path(ckpt) / "prover_sft_base"\n\n\ndef _examples(rows: List[dict]) -> List[dict]:\n    out, seen = [], set()\n    for r in rows:\n        sound = r.get("is_sound") if "is_sound" in r else (r.get("label") in ("honest", "helpful", "sound"))\n        exp = r.get("experiment")\n        if not sound or not exp or not r.get("summary"):\n            continue\n        key = (exp.get("record_id"), r["summary"])\n        if key in seen:\n            continue\n        seen.add(key)\n        out.append({"record": exp, "summary": r["summary"].strip()})\n    return out\n\n\ndef run_sft(cfg: PVGConfig, train_rows: List[dict], out_dir: Path, seed: int) -> dict:\n    from training import pvg_loop\n    from training.train_prover_step import compute_sequence_logprob, generate_group, load_prover\n\n    out_dir = Path(out_dir)\n    rng = random.Random(seed)\n    examples = _examples(train_rows)\n    if not examples:\n        raise RuntimeError("SFT warm start: no SOUND rows with an experiment record to train on")\n    print(f"[sft] warm start on {len(examples)} honest write-ups x {cfg.prover_sft_epochs} epoch(s), lr {cfg.prover_sft_lr}")\n\n    state = load_prover(cfg, cfg.prover_model)\n    tok, model = state.tokenizer, state.model\n    trainable = [p for p in model.parameters() if p.requires_grad]\n    opt = torch.optim.AdamW(trainable, lr=cfg.prover_sft_lr)\n    prompts = {}\n\n    def prompt_for(rec):\n        k = json.dumps(rec, sort_keys=True)\n        if k not in prompts:\n            prompts[k] = pvg_loop.render_prover_prompt(pvg_loop._HELPFUL_PROMPT_PATH, rec, tok, cfg)\n        return prompts[k]\n\n    bs, losses = 8, []\n    model.train()\n    for ep in range(max(1, int(cfg.prover_sft_epochs))):\n        order = list(examples)\n        rng.shuffle(order)\n        for start in range(0, len(order), bs):\n            chunk = order[start:start + bs]\n            opt.zero_grad(set_to_none=True)\n            for ex in chunk:\n                completion = ex["summary"] + (tok.eos_token or "")\n                n_tok = max(1, len(tok(completion, add_special_tokens=False)["input_ids"]))\n                lp = compute_sequence_logprob(model, tok, prompt_for(ex["record"]), completion, state.device)\n                loss = -(lp / n_tok) / len(chunk)       # mean token NLL\n                loss.backward()\n                losses.append(loss.item() * len(chunk))\n            torch.nn.utils.clip_grad_norm_(trainable, 1.0)\n            opt.step()\n        print(f"[sft] epoch {ep + 1}: mean token NLL {sum(losses[-len(order):]) / len(order):.3f}")\n    model.eval()\n\n    # Fidelity probe: does the warm-started model now write rule-check-passing text?\n    probe_recs = rng.sample([e["record"] for e in examples], min(8, len(examples)))\n    h_ok, s_fail, n = 0, 0, 0\n    for rec in probe_recs:\n        for c in generate_group(state, pvg_loop.render_prover_prompt(pvg_loop._HELPFUL_PROMPT_PATH, rec, tok, cfg), cfg):\n            h_ok += pvg_loop.rule_check(rec, c)[0]; n += 1\n    n_s = 0\n    for rec in probe_recs:\n        for c in generate_group(state, pvg_loop.render_prover_prompt(pvg_loop._SNEAKY_PROMPT_PATH, rec, tok, cfg), cfg):\n            s_fail += not pvg_loop.rule_check(rec, c)[0]; n_s += 1\n    report = {"examples": len(examples), "epochs": cfg.prover_sft_epochs, "lr": cfg.prover_sft_lr,\n              "helpful_fidelity_after_sft": h_ok / max(1, n), "sneaky_fidelity_after_sft": s_fail / max(1, n_s),\n              "probe_samples_per_role": n}\n    print(f"[sft] after warm start: helpful fidelity {report[\'helpful_fidelity_after_sft\']:.2f} "\n          f"(base Qwen was ~0.06-0.20), sneaky fidelity {report[\'sneaky_fidelity_after_sft\']:.2f}")\n    if report["helpful_fidelity_after_sft"] < 0.5:\n        print("[sft] WARNING: helpful fidelity is still below 0.5. Try prover_sft_epochs=3 "\n              "or check the failing samples before the full run.")\n\n    merged = model.merge_and_unload()\n    out_dir.mkdir(parents=True, exist_ok=True)\n    merged.save_pretrained(str(out_dir))\n    tok.save_pretrained(str(out_dir))\n    (out_dir / "sft_report.json").write_text(json.dumps(report, indent=2))\n    del merged, model, state, opt\n    if torch.cuda.is_available():\n        torch.cuda.empty_cache()\n    print(f"[sft] merged warm-start model saved to {out_dir}; both provers start from it")\n    return report\n'

MODEL_WRITTEN_ROWS = '"""\nModel-written verifier rows (fix for "SOUND only ever looks like the template").\n\nProblem this addresses: every SOUND row in findings_pvg_dataset.jsonl is\ntemplated text, while almost every prover-written row the verifier sees in\nthe game is labeled UNSOUND. The cheapest feature that separates the classes\nis "template vs. model prose", and the verifier learns that instead of\nsoundness (helpful accuracy collapses to ~0.1, held-out AUROC still ~0.99\nbecause held-out is templated too).\n\nThis script makes SOUND and UNSOUND rows that are BOTH written by the prover\'s\nown base model, so style no longer predicts the label:\n\n  SOUND   - Qwen rewrites of a record\'s honest write-up ("paraphrase"), plus\n            Qwen write-ups from the record alone ("freewrite"). Kept only if\n            they pass the rule check AND the extra faithfulness guards below\n            (cites the PGR, states the seed count, keeps the few-seed caveat,\n            keeps a negative conclusion when the record did not work).\n  UNSOUND - for each kept SOUND text, a twin made by applying the dataset\'s\n            own perturbations to THAT text (same wording, one number/claim\n            changed), kept only if the twin fails the rule check. Twins are\n            the best control: the pair differs only in the misrepresentation.\n\nTrain records -> data/findings_pvg_model_written.jsonl       (verifier training)\nHeld-out records -> data/findings_pvg_heldout_model_written.jsonl (evaluation)\nThe split is inherited from the existing files, so no held-out record leaks.\n\nRun on the GPU (Colab), after build_findings_dataset.py:\n    python data/build_model_written_rows.py\nTest the filtering/pairing logic without a model:\n    python data/build_model_written_rows.py --stub\n"""\nimport argparse\nimport json\nimport random\nimport re\nimport sys\nfrom collections import Counter\nfrom pathlib import Path\nfrom typing import Dict, List, Optional\n\nsys.path.insert(0, str(Path(__file__).resolve().parent.parent))\nfrom data.build_findings_dataset import PERTURBATIONS, check_internal_consistency, fmt  # noqa: E402\nfrom data.schema import ExperimentRecord  # noqa: E402\n\nSOURCE_TAG = "model_written"\n\nPARAPHRASE_PROMPT = """Rewrite the research finding below in your own words, as one short paragraph.\n\nRules:\n- Keep every number exactly as written (same decimals). Do not add, round, or compute any new number.\n- Keep the seed count and any caveat about it.\n- Keep the conclusion: if the original says the method did not beat the baseline, or that the result is within noise or preliminary, say so too.\n- No lists, no headings, no preamble. Output only the rewritten paragraph.\n\nOriginal finding:\n{summary}"""\n\nFREEWRITE_PROMPT = """Write a short, accurate one-paragraph summary of this weak-to-strong experiment for a research log.\n\nRules:\n- Cite numbers exactly as they appear below. Do not round, compute differences, or give percentages.\n- State the number of seeds.\n- If the record says it did not work, or PGR is not above the baseline PGR, say plainly that it did not improve on the baseline.\n- If there are fewer than 5 seeds, say the result is preliminary.\n- No lists, no headings, no preamble. Output only the paragraph.\n\nMetrics: {metrics}\nConfig: {config}\nWorked: {worked}\nTestbed: {dataset}"""\n\n_NEGATIVE_CUES = re.compile(r"\\b(not|no|did not|didn\'t|does not|fell short|falls short|short of|below|"\n                            r"within (?:the )?(?:standard )?error|noise|fail\\w*|n\'t|neither|nor)\\b", re.I)\n_CAVEAT_CUES = re.compile(r"\\b(preliminary|only|small|few|limited|too small|caution|tentative|"\n                          r"firm conclusion|more seeds|not enough)\\b", re.I)\n_PREAMBLE = re.compile(r"^\\s*(?:sure[,!.]?|here(?:\'s| is)[^:\\n]*:|rewritten (?:finding|paragraph):|summary:)\\s*", re.I)\n\n\n# ---------------------------------------------------------------------------\n# Filtering: does a model-written text deserve a SOUND label?\n# ---------------------------------------------------------------------------\ndef to_record(exp: dict) -> ExperimentRecord:\n    return ExperimentRecord(idea_name=exp.get("idea_name", "unknown"), metrics=exp.get("metrics") or {},\n                            config=exp.get("config") or {}, worked=exp.get("worked"),\n                            dataset=exp.get("dataset"), record_id=exp.get("record_id"))\n\n\ndef clean(text: str) -> str:\n    text = _PREAMBLE.sub("", text.strip())\n    text = text.strip().strip(\'"\').strip()\n    return re.sub(r"\\s+", " ", text)\n\n\ndef sound_problems(rec: ExperimentRecord, text: str) -> List[str]:\n    """Empty list = acceptable as SOUND. The rule check plus guards for the\n    omissions it cannot see (a dropped caveat or a dropped negative result\n    is exactly what several sneaky perturbations do)."""\n    probs: List[str] = []\n    if not (80 <= len(text) <= 900):\n        probs.append("length")\n    if re.search(r"^\\s*(?:\\d+[.)]|[-*•])\\s", text, re.M) or "Metrics:" in text or "Config:" in text:\n        probs.append("list/echo")\n    ok, issues = check_internal_consistency(rec, text)\n    if not ok:\n        probs.append("rule_check: " + "; ".join(issues))\n    m = rec.metrics\n    pgr, n, base = m.get("pgr"), m.get("num_seeds"), m.get("baseline_pgr")\n    if isinstance(pgr, (int, float)) and fmt(pgr) not in text and f"{pgr:g}" not in text:\n        probs.append("pgr not cited")\n    if n is not None and not re.search(rf"\\b{n}\\b", text):\n        probs.append("seed count not stated")\n    if n is not None and n < 5 and not _CAVEAT_CUES.search(text):\n        probs.append("few-seed caveat dropped")\n    # Only records with a baseline have a "did not beat it" conclusion to keep.\n    # On sparse records worked=False just means low PGR; the rule check\n    # already rejects positive framing there.\n    has_base = isinstance(pgr, (int, float)) and isinstance(base, (int, float))\n    if has_base and (rec.worked is False or pgr <= base) and not _NEGATIVE_CUES.search(text):\n        probs.append("negative conclusion dropped")\n    return probs\n\n\ndef make_twin(rec: ExperimentRecord, text: str, rng: random.Random) -> Optional[tuple]:\n    """Apply the dataset\'s perturbations to a model-written text. Returns\n    (twin_text, perturbation, family) for one that changes the text and fails\n    the rule check, or None. (The true-numbers perturbations match template\n    phrasing and rarely apply to free text; number/seed/framing ones do.)"""\n    options = []\n    for name, fn, _w in PERTURBATIONS:\n        try:\n            res = fn(rec, text, rng)\n        except Exception:\n            res = None\n        if not res or not res[0] or res[0].strip() == text.strip():\n            continue\n        twin, family = res\n        if not check_internal_consistency(rec, twin)[0]:\n            options.append((twin, name, family))\n    return rng.choice(options) if options else None\n\n\ndef row(exp: dict, summary: str, sound: bool, split: str, perturbation=None, family=None, origin=None) -> dict:\n    ok, issues = check_internal_consistency(to_record(exp), summary)\n    return {"experiment": exp, "summary": summary, "label": "honest" if sound else "sneaky",\n            "is_internally_consistent": sound, "rule_check_passes": ok, "consistency_issues": issues,\n            "perturbation": perturbation, "perturbation_family": family, "split": split,\n            "source": SOURCE_TAG, "generation": origin}\n\n\ndef build_rows(honest_rows: List[dict], generations: Dict[str, List[tuple]], split: str,\n               per_record: int, rng: random.Random, stats: Counter) -> List[dict]:\n    """generations: record_id -> [(origin, text), ...]. Keeps up to per_record\n    SOUND texts per record, each with its UNSOUND twin. A SOUND text with no\n    valid twin is dropped, so the output is exactly class-balanced."""\n    out = []\n    for hr in honest_rows:\n        exp = hr["experiment"]\n        rec = to_record(exp)\n        seen = {re.sub(r"\\W+", " ", hr["summary"].lower()).strip()}\n        kept = 0\n        for origin, raw in generations.get(exp["record_id"], []):\n            if kept >= per_record:\n                break\n            text = clean(raw)\n            key = re.sub(r"\\W+", " ", text.lower()).strip()\n            if key in seen:\n                stats["dup"] += 1\n                continue\n            seen.add(key)\n            probs = sound_problems(rec, text)\n            if probs:\n                stats["rejected"] += 1\n                for p in probs:\n                    stats["why: " + p.split(":")[0]] += 1\n                continue\n            twin = make_twin(rec, text, rng)\n            if twin is None:\n                stats["no_twin"] += 1\n                continue\n            out.append(row(exp, text, True, split, origin=origin))\n            out.append(row(exp, twin[0], False, split, twin[1], twin[2], origin=origin))\n            stats[f"kept_{origin}"] += 1\n            kept += 1\n        stats["records_with_rows" if kept else "records_without_rows"] += 1\n    return out\n\n\n# ---------------------------------------------------------------------------\n# Generation\n# ---------------------------------------------------------------------------\ndef prompts_for(hr: dict) -> List[tuple]:\n    exp = hr["experiment"]\n    return [("paraphrase", PARAPHRASE_PROMPT.format(summary=hr["summary"])),\n            ("freewrite", FREEWRITE_PROMPT.format(metrics=json.dumps(exp.get("metrics") or {}, sort_keys=True),\n                                                  config=json.dumps(exp.get("config") or {}, sort_keys=True),\n                                                  worked=exp.get("worked"), dataset=exp.get("dataset")))]\n\n\ndef generate_with_model(honest_rows: List[dict], model_name: str, n_samples: int, temperature: float,\n                        max_new_tokens: int, batch_size: int, seed: int) -> Dict[str, List[tuple]]:\n    import torch\n    from transformers import AutoModelForCausalLM, AutoTokenizer\n\n    torch.manual_seed(seed)\n    tok = AutoTokenizer.from_pretrained(model_name)\n    tok.padding_side = "left"\n    if tok.pad_token is None:\n        tok.pad_token = tok.eos_token\n    device = "cuda" if torch.cuda.is_available() else "cpu"\n    dtype = torch.bfloat16 if device == "cuda" else torch.float32\n    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()\n\n    jobs = []  # (record_id, origin, chat_prompt)\n    for hr in honest_rows:\n        for origin, text in prompts_for(hr):\n            chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,\n                                           add_generation_prompt=True)\n            jobs.append((hr["experiment"]["record_id"], origin, chat))\n\n    out: Dict[str, List[tuple]] = {}\n    for start in range(0, len(jobs), batch_size):\n        chunk = jobs[start:start + batch_size]\n        enc = tok([j[2] for j in chunk], return_tensors="pt", padding=True).to(device)\n        with torch.no_grad():\n            gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,\n                                 max_new_tokens=max_new_tokens, num_return_sequences=n_samples,\n                                 pad_token_id=tok.pad_token_id)\n        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)\n        for i, t in enumerate(texts):\n            rid, origin, _ = chunk[i // n_samples]\n            out.setdefault(rid, []).append((origin, t))\n        print(f"  generated {min(start + batch_size, len(jobs))}/{len(jobs)} prompts", flush=True)\n    # Interleave paraphrase/freewrite per record so per_record keeps a mix.\n    for rid, items in out.items():\n        para = [x for x in items if x[0] == "paraphrase"]\n        free = [x for x in items if x[0] == "freewrite"]\n        mixed = [x for pair in zip(para, free) for x in pair] + para[len(free):] + free[len(para):]\n        out[rid] = mixed\n    return out\n\n\ndef generate_stub(honest_rows: List[dict]) -> Dict[str, List[tuple]]:\n    """No model: fake \'model-written\' rewrites so the filter/twin logic can be\n    tested offline. Not for training."""\n    out = {}\n    for hr in honest_rows:\n        s = hr["summary"]\n        out[hr["experiment"]["record_id"]] = [\n            ("paraphrase", "Here is the rewritten finding: " + s.replace("PGR", "performance gap recovered (PGR)", 1)),\n            ("paraphrase", "1. " + s),                                   # list -> rejected\n            ("freewrite", s.replace(" seeds", " runs").replace(" seed", " run")),  # seed count lost -> rejected\n        ]\n    return out\n\n\ndef main() -> int:\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--train", default="data/findings_pvg_dataset.jsonl")\n    ap.add_argument("--heldout", default="data/findings_pvg_heldout.jsonl")\n    ap.add_argument("--train-out", default="data/findings_pvg_model_written.jsonl")\n    ap.add_argument("--heldout-out", default="data/findings_pvg_heldout_model_written.jsonl")\n    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")  # the provers\' base model\n    ap.add_argument("--samples", type=int, default=4, help="samples per prompt (2 prompts per record)")\n    ap.add_argument("--per-record", type=int, default=2, help="max SOUND/UNSOUND pairs kept per record")\n    ap.add_argument("--temperature", type=float, default=0.8)\n    ap.add_argument("--max-new-tokens", type=int, default=200)\n    ap.add_argument("--batch-size", type=int, default=16)\n    ap.add_argument("--seed", type=int, default=42)\n    ap.add_argument("--stub", action="store_true", help="no model; test the filtering logic only")\n    args = ap.parse_args()\n\n    load = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]\n    for split, src, dst in (("train", args.train, args.train_out), ("heldout", args.heldout, args.heldout_out)):\n        honest = [r for r in load(src) if r["label"] == "honest"]\n        print(f"[{split}] {len(honest)} honest records from {src}")\n        gens = generate_stub(honest) if args.stub else generate_with_model(\n            honest, args.model, args.samples, args.temperature, args.max_new_tokens, args.batch_size, args.seed)\n        stats: Counter = Counter()\n        rows = build_rows(honest, gens, split, args.per_record, random.Random(args.seed), stats)\n        Path(dst).write_text("".join(json.dumps(r) + "\\n" for r in rows))\n        n_sound = sum(r["label"] == "honest" for r in rows)\n        print(f"[{split}] wrote {len(rows)} rows ({n_sound} SOUND / {len(rows) - n_sound} UNSOUND) to {dst}")\n        print(f"[{split}] perturbations used: {dict(Counter(r[\'perturbation\'] for r in rows if r[\'perturbation\']))}")\n        print(f"[{split}] filter stats: {dict(sorted(stats.items()))}")\n    return 0\n\n\nif __name__ == "__main__":\n    sys.exit(main())\n'

REAL_RECORDS = {
 "aar_directed_headline": {
  "id": "aar_directed_headline",
  "idea_name": "nine_parallel_directed_aars",
  "pgr": 0.97,
  "config": {
   "method": "nine parallel automated researchers (Claude Opus 4.6), each seeded with a distinct research direction, running for 5 days (800 cumulative hours)",
   "testbed": "chat_preference",
   "num_researchers": 9,
   "days": 5,
   "cumulative_hours": 800
  },
  "worked": True,
  "summary": "Running 9 automated researchers in parallel, each seeded with a different research direction, the best idea reached a PGR of 0.97 on the chat preference testbed within 5 days and 800 cumulative hours.",
  "source": "Wen, Qiu, et al., 'Automated Weak-to-Strong Researcher', Anthropic Alignment Science Blog, 2026",
  "notes": "Headline result of the AAR system as a whole, not a single idea. Seed count and SE not reported."
 },
 "baseline_confident_labels": {
  "id": "baseline_confident_labels",
  "idea_name": "train_only_on_confident_labels",
  "pgr": 0.23,
  "pgr_se": 0.05,
  "num_seeds": 5,
  "transfer_acc": 0.58,
  "weak_acc": 0.53,
  "strong_acc": 0.73,
  "config": {
   "method": "training the strong student only on weak labels with confidence above 0.75 or below 0.25",
   "weak_model": "Qwen1.5-0.5B-Chat",
   "strong_model": "Qwen3-4B-Base",
   "testbed": "chat_preference",
   "conf_high": 0.75,
   "conf_low": 0.25
  },
  "worked": False,
  "summary": "Training Qwen3-4B-Base only on the weak labels where Qwen1.5-0.5B-Chat was confident (above 0.75 or below 0.25) gave a PGR of 0.23 (se 0.05) over 5 seeds on chat preference; transfer accuracy was 0.58 (weak supervisor 0.53, strong ceiling 0.73). Individual seeds ranged widely, so this is only a modest gain over weak supervision.",
  "source": "safety-research/automated-w2s-research, cache_results.tar.gz (chat-0115 runs); metrics computed from the per-seed JSONs, write-up by Claude",
  "notes": "Human-run baseline; matches the blog's 'best human PGR of 0.23'. Per-seed PGR: 0.40, 0.22, 0.21, 0.27, 0.07."
 },
 "baseline_critic_grpo": {
  "id": "baseline_critic_grpo",
  "idea_name": "critic_training",
  "pgr": -0.04,
  "pgr_se": 0.04,
  "num_seeds": 5,
  "transfer_acc": 0.52,
  "weak_acc": 0.53,
  "strong_acc": 0.73,
  "config": {
   "method": "training a critic with GRPO (KL penalty 0.02, 50 steps) on weak demonstrations, then using its judgments to label data for the strong student",
   "weak_model": "Qwen1.5-0.5B-Chat",
   "strong_model": "Qwen3-4B-Base",
   "testbed": "chat_preference",
   "grpo_kl_penalty": 0.02,
   "grpo_max_steps": 50
  },
  "worked": False,
  "summary": "Training a GRPO critic on weak demonstrations and using it to label data for Qwen3-4B-Base gave a PGR of -0.04 (se 0.04) across 5 seeds on chat preference; transfer accuracy was 0.52, slightly below the weak supervisor's 0.53. The critic did not help in this setting.",
  "source": "safety-research/automated-w2s-research, cache_results.tar.gz (chat-0115 runs); metrics computed from the per-seed JSONs, write-up by Claude",
  "notes": "Human-run baseline. Per-seed PGR: -0.12, -0.10, 0.06, 0.07, -0.11."
 },
 "baseline_ue_zeroshot": {
  "id": "baseline_ue_zeroshot",
  "idea_name": "ue_zeroshot",
  "pgr": 0.03,
  "pgr_se": 0.02,
  "num_seeds": 4,
  "transfer_acc": 0.53,
  "weak_acc": 0.53,
  "strong_acc": 0.73,
  "config": {
   "method": "zero-shot unsupervised elicitation: the strong base model's own label prior, with no weak labels",
   "strong_model": "Qwen3-4B-Base",
   "testbed": "chat_preference"
  },
  "worked": False,
  "summary": "Using the zero-shot label prior of Qwen3-4B-Base with no weak labels, PGR was 0.03 (se 0.02) over 4 seeds on chat preference, with transfer accuracy of 0.53, essentially no better than the weak supervisor. With only 4 seeds this result should be treated as preliminary.",
  "source": "safety-research/automated-w2s-research, cache_results.tar.gz (chat-0115 runs); metrics computed from the per-seed JSONs, write-up by Claude",
  "notes": "Human-run baseline. The 4 result files all record seed=42 but have different results; treated as 4 runs. Per-run PGR: -0.04, 0.07, 0.03, 0.04."
 },
 "baseline_vanilla_w2s": {
  "id": "baseline_vanilla_w2s",
  "idea_name": "vanilla_w2s",
  "pgr": 0.2,
  "pgr_se": 0.05,
  "num_seeds": 5,
  "transfer_acc": 0.57,
  "weak_acc": 0.53,
  "strong_acc": 0.73,
  "config": {
   "method": "fine-tuning the strong student directly on the weak teacher's labels",
   "weak_model": "Qwen1.5-0.5B-Chat",
   "strong_model": "Qwen3-4B-Base",
   "testbed": "chat_preference"
  },
  "worked": False,
  "summary": "Fine-tuning Qwen3-4B-Base directly on the labels of Qwen1.5-0.5B-Chat reached a PGR of 0.20 (se 0.05) across 5 seeds on the chat preference testbed, with transfer accuracy of 0.57 against a weak-supervisor accuracy of 0.53 and a strong ceiling of 0.73.",
  "source": "safety-research/automated-w2s-research, cache_results.tar.gz (chat-0115 runs); metrics computed from the per-seed JSONs, write-up by Claude",
  "notes": "Human-run baseline. Per-seed PGR: 0.37, 0.21, 0.12, 0.19, 0.13."
 }
}


# ===========================================================================
# Patch machinery
# ===========================================================================
MARK = "[fixes-v4]"


def find_root(arg: Optional[str]) -> Path:
    cands = [Path(arg)] if arg else [Path(__file__).resolve().parent, Path.cwd()]
    for c in cands:
        for d in [c] + list(c.glob("*")) + list(c.glob("*/*")):
            if (d / "config.py").exists() and (d / "training" / "pvg_loop.py").exists():
                return d.resolve()
    raise SystemExit("Could not find the project root (folder with config.py and training/pvg_loop.py). "
                     "Run from the project folder or pass --root.")


class Patcher:
    def __init__(self, root: Path):
        self.root, self.log, self.changed = root, [], set()

    def replace(self, rel: str, old: str, new: str, label: str) -> None:
        p = self.root / rel
        text = p.read_text()
        if f"{MARK} {label}" in text:
            self.log.append(f"  = {label}: already applied")
            return
        assert f"{MARK} {label}" in new, f"internal: replacement for {label!r} lacks its marker"
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"\n{label}: expected the anchor exactly once in {rel}, found {n}.\n"
                             f"This branch has changed since the fixes were written (built against commit {BUILT_AGAINST}).\n"
                             f"Nothing after this point was applied; send the error to get an updated script.")
        p.write_text(text.replace(old, new, 1))
        self.changed.add(rel)
        self.log.append(f"  + {label}: {rel}")

    def write(self, rel: str, content: str, label: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_text() == content:
            self.log.append(f"  = {label}: already present")
            return
        p.write_text(content)
        self.changed.add(rel)
        self.log.append(f"  + {label}: {rel}")


FORK_URL = "https://github.com/varchanaiyer/Automated_R_leg_NoAPI.git"
PINNED_SHA = "51940d1c10b229b6d735633a26bd2950bb632974"


def pin_code(root: Path) -> None:
    """Make the project folder match the code these fixes were written for (fork branch
    fix/verifier-scoring-and-gates at 51940d1), whichever repo/branch Colab cloned: your
    main is older (no assign_verifier_labels, no prompt rules) and the fork keeps moving.
    Only tracked files of that commit are overwritten; extra files (this script) stay."""
    def git(*a, check=True):
        return subprocess.run(["git", "-C", str(root), *a], text=True, capture_output=True, check=check)
    try:
        top = Path(git("rev-parse", "--show-toplevel").stdout.strip())
    except Exception:
        print("  (not a git checkout; skipping the version pin and relying on the preflight check)")
        return
    rel = root.relative_to(top).as_posix()
    if git("cat-file", "-e", PINNED_SHA + "^{commit}", check=False).returncode != 0:
        print(f"  fetching the pinned code ({BUILT_AGAINST}) from {FORK_URL} ...", flush=True)
        r = git("fetch", "--quiet", "--depth", "1", FORK_URL, PINNED_SHA, check=False)
        if r.returncode != 0:
            r = git("fetch", "--quiet", FORK_URL, "fix/verifier-scoring-and-gates", check=False)
        if git("cat-file", "-e", PINNED_SHA + "^{commit}", check=False).returncode != 0:
            raise SystemExit(f"Could not fetch commit {BUILT_AGAINST} from {FORK_URL}:\n{r.stderr}")
    r = subprocess.run(["git", "-C", str(top), "checkout", PINNED_SHA, "--", rel or "."], text=True, capture_output=True)
    if r.returncode:
        raise SystemExit(f"Could not check out the pinned code: {r.stderr.strip()}")
    print(f"  project code pinned to {BUILT_AGAINST} ({rel or '.'}); this script and other extra files untouched")


def preflight(root: Path) -> None:
    loop = (root / "training" / "pvg_loop.py").read_text()
    prompt = (root / "prompts" / "helpful_prover_prompt.jinja2").read_text()
    problems = []
    if "def class_weights" not in loop:
        problems.append("no class-balanced verifier loss (pvg_loop.class_weights)")
    if "def assign_verifier_labels" not in loop:
        problems.append("no assign_verifier_labels()")
    if "Rules your write-up is checked against" not in prompt:
        problems.append("helpful prompt without the rule list")
    if problems:
        raise SystemExit("This checkout is OLDER than the code these fixes are written for:\n  - "
                         + "\n  - ".join(problems)
                         + f"\nYour last run used commit 8616874 (26 Sep), which predates these. Clone the branch head "
                           f"(or `git checkout {BUILT_AGAINST}`) and run this script again.")


def apply_all(root: Path) -> Patcher:
    P = Patcher(root)

    # ---------------- config.py: new settings ----------------
    P.replace("config.py",
        "    verifier_labels_from_rule_check: bool = True\n",
        "    verifier_labels_from_rule_check: bool = True\n"
        f"    # {MARK} config\n"
        "    # Fix 1. What happens to a HELPFUL sample that fails the rule check:\n"
        "    #   \"drop\": left out of verifier training (the rule check has false positives on\n"
        "    #           accurate prose, so failing it is not reliable evidence of unsoundness)\n"
        "    #   \"relabel\": labeled UNSOUND (the old behaviour: ~100 of 128 helpful samples per\n"
        "    #           round became UNSOUND and the verifier learned 'prover text = UNSOUND')\n"
        "    verifier_failed_helpful: str = \"drop\"\n"
        "    # Fix 1. Per round, subsample the larger class of prover samples so neither label\n"
        "    # dominates the replay window. Labels are never changed, only how many are kept.\n"
        "    verifier_round_balance: bool = True\n"
        "    verifier_round_balance_floor: int = 16\n"
        "    # Fix 2. Model-written SOUND rows + perturbed UNSOUND twins (data/build_model_written_rows.py),\n"
        "    # added to the verifier's labeled pool and held-out eval when the files exist. \"\" disables.\n"
        "    model_written_train_path: str = \"data/findings_pvg_model_written.jsonl\"\n"
        "    model_written_heldout_path: str = \"data/findings_pvg_heldout_model_written.jsonl\"\n"
        "    # Fix 4. Supervised warm start of BOTH provers on honest write-ups before round 1\n"
        "    # (training/sft_warmstart.py). 0 disables.\n"
        "    prover_sft_epochs: int = 2\n"
        "    prover_sft_lr: float = 1e-4\n"
        "    # Fix 5. Bonus added to a gated reward when the sample does its role (helpful passes the\n"
        "    # rule check / sneaky fails it), so neither prover's signal vanishes when the verifier\n"
        "    # rejects everything (sneaky reward was 0.000 from round 7 on). 0 disables.\n"
        "    prover_role_bonus: float = 0.1\n",
        "config")

    # ---------------- pvg_loop.py ----------------
    L = "training/pvg_loop.py"
    P.replace(L, "def pick_device() -> str:",
        f"# {MARK} model-written loader\n"
        "def with_model_written(rows: List[dict], path: str, what: str) -> List[dict]:\n"
        "    \"\"\"rows + the model-written rows at `path` (if configured and present). Verifier\n"
        "    pool and held-out eval only; prover prompts keep using the original records.\"\"\"\n"
        "    if not path or not Path(path).exists():\n"
        "        print(f\"  [data] no model-written {what} rows ({path or 'disabled'}); templated rows only\")\n"
        "        return list(rows)\n"
        "    extra = load_dataset(path)\n"
        "    n_sound = sum(1 for r in extra if is_sound_label(r))\n"
        "    print(f\"  [data] + {len(extra)} model-written {what} rows ({n_sound} SOUND / {len(extra) - n_sound} UNSOUND)\")\n"
        "    return list(rows) + extra\n\n\n"
        "def pick_device() -> str:",
        "model-written loader")

    P.replace(L,
        "    examples, relabeled, dropped = [], 0, 0\n"
        "    for it in helpful_scored:\n"
        "        ok = it.get(\"rule_ok\", True)\n"
        "        if cfg.verifier_labels_from_rule_check and not ok:\n"
        "            examples.append(make_verifier_example(it[\"summary\"], it[\"record\"], False, \"helpful_relabeled\"))\n"
        "            relabeled += 1\n"
        "        else:\n"
        "            examples.append(make_verifier_example(it[\"summary\"], it[\"record\"], True, \"helpful\"))\n"
        "    for it in sneaky_scored:\n"
        "        ok = it.get(\"rule_ok\", False)\n"
        "        if cfg.verifier_labels_from_rule_check and ok:\n"
        "            dropped += 1\n"
        "            continue\n"
        "        examples.append(make_verifier_example(it[\"summary\"], it[\"record\"], False, \"sneaky\"))\n"
        "    return examples, {\"relabeled_helpful\": relabeled, \"dropped_sneaky\": dropped, \"kept\": len(examples)}\n",
        f"    # {MARK} labels\n"
        "    # Fix 1: a helpful sample failing the rule check is DROPPED (not relabeled UNSOUND) unless\n"
        "    # cfg.verifier_failed_helpful == \"relabel\"; then each round's two classes are balanced by\n"
        "    # random subsampling (labels never change, only how many of each are kept).\n"
        "    sound, unsound, relabeled, dropped, failed = [], [], 0, 0, 0\n"
        "    for it in helpful_scored:\n"
        "        ok = it.get(\"rule_ok\", True)\n"
        "        if cfg.verifier_labels_from_rule_check and not ok:\n"
        "            failed += 1\n"
        "            if cfg.verifier_failed_helpful == \"relabel\":\n"
        "                unsound.append(make_verifier_example(it[\"summary\"], it[\"record\"], False, \"helpful_relabeled\"))\n"
        "                relabeled += 1\n"
        "        else:\n"
        "            sound.append(make_verifier_example(it[\"summary\"], it[\"record\"], True, \"helpful\"))\n"
        "    for it in sneaky_scored:\n"
        "        ok = it.get(\"rule_ok\", False)\n"
        "        if cfg.verifier_labels_from_rule_check and ok:\n"
        "            dropped += 1\n"
        "            continue\n"
        "        unsound.append(make_verifier_example(it[\"summary\"], it[\"record\"], False, \"sneaky\"))\n"
        "    subsampled = 0\n"
        "    if cfg.verifier_round_balance:\n"
        "        cap_u = max(len(sound), cfg.verifier_round_balance_floor)\n"
        "        if len(unsound) > cap_u:\n"
        "            subsampled += len(unsound) - cap_u\n"
        "            unsound = random.sample(unsound, cap_u)\n"
        "        cap_s = max(len(unsound), cfg.verifier_round_balance_floor)\n"
        "        if len(sound) > cap_s:\n"
        "            subsampled += len(sound) - cap_s\n"
        "            sound = random.sample(sound, cap_s)\n"
        "    examples = sound + unsound\n"
        "    return examples, {\"relabeled_helpful\": relabeled, \"dropped_sneaky\": dropped, \"kept\": len(examples),\n"
        "                      \"failed_helpful\": failed, \"round_sound_added\": len(sound),\n"
        "                      \"round_unsound_added\": len(unsound), \"subsampled_for_balance\": subsampled}\n",
        "labels")

    P.replace(L,
        "    print(f\"  [verifier labels] kept={lstats['kept']} relabeled_helpful->UNSOUND={lstats['relabeled_helpful']} \"\n"
        "          f\"dropped_sneaky(passes rule check)={lstats['dropped_sneaky']}\")\n",
        f"    # {MARK} label log\n"
        "    print(f\"  [verifier labels] added SOUND={lstats['round_sound_added']} UNSOUND={lstats['round_unsound_added']} | \"\n"
        "          f\"helpful failing rule check={lstats['failed_helpful']} ({cfg.verifier_failed_helpful}) \"\n"
        "          f\"dropped_sneaky(passes rule check)={lstats['dropped_sneaky']} subsampled={lstats['subsampled_for_balance']}\")\n",
        "label log")

    P.replace(L,
        "            if mode == \"helpful_gated\":\n"
        "                return (p if ok else 0.0) if role == \"helpful\" else p\n"
        "            if mode == \"correctness_gated\":\n"
        "                aligned = ok if role == \"helpful\" else (not ok)\n"
        "                return p if aligned else 0.0\n",
        f"            # {MARK} role bonus\n"
        "            b = float(getattr(cfg, \"prover_role_bonus\", 0.0))\n"
        "            if mode == \"helpful_gated\":\n"
        "                return (p + b if ok else 0.0) if role == \"helpful\" else p\n"
        "            if mode == \"correctness_gated\":\n"
        "                aligned = ok if role == \"helpful\" else (not ok)\n"
        "                return p + b if aligned else 0.0\n",
        "role bonus")

    P.replace(L,
        "        verifier_train_sound_share=vstats.get(\"sound_share\"),\n    )\n",
        "        verifier_train_sound_share=vstats.get(\"sound_share\"),\n    )\n"
        f"    # {MARK} extra diagnostics\n"
        "    for _k in (\"failed_helpful\", \"round_sound_added\", \"round_unsound_added\", \"subsampled_for_balance\"):\n"
        "        setattr(metadata, \"verifier_\" + _k, vstats.get(_k))\n",
        "extra diagnostics")

    P.replace(L,
        "    random.seed(cfg.seed)\n    torch.manual_seed(cfg.seed)\n\n    prepare_verifier_for_run(cfg, dataset, spot_rows)\n",
        f"    # {MARK} run_pvg_training data\n"
        "    verifier_rows = with_model_written(dataset, cfg.model_written_train_path, \"train\")\n"
        "    if heldout_rows:\n"
        "        heldout_rows = with_model_written(heldout_rows, cfg.model_written_heldout_path, \"held-out\")\n"
        "    random.seed(cfg.seed)\n    torch.manual_seed(cfg.seed)\n\n    prepare_verifier_for_run(cfg, verifier_rows, spot_rows)\n",
        "run_pvg_training data")

    # ---------------- run_one_round.py ----------------
    R = "run_one_round.py"
    P.replace(R,
        "    spot_rows = load_dataset(args.spot_set) if Path(args.spot_set).exists() else []\n",
        "    spot_rows = load_dataset(args.spot_set) if Path(args.spot_set).exists() else []\n"
        f"    # {MARK} model-written rows\n"
        "    verifier_rows = pvg_loop.with_model_written(dataset, cfg.model_written_train_path, \"train\")\n"
        "    if heldout_rows:\n"
        "        heldout_rows = pvg_loop.with_model_written(heldout_rows, cfg.model_written_heldout_path, \"held-out\")\n",
        "model-written rows")
    P.replace(R, "            set_dataset_examples(dataset)\n", "            set_dataset_examples(verifier_rows)  # " + MARK + " pool\n", "pool")
    P.replace(R, "pvg_loop.prepare_verifier_for_run(cfg, dataset, spot_rows)",
              "pvg_loop.prepare_verifier_for_run(cfg, verifier_rows, spot_rows)  # " + MARK + " warmup pool", "warmup pool")
    P.replace(R,
        "    # --- Provers ---\n",
        f"    # {MARK} sft warm start\n"
        "    # Fix 4: both provers start from one supervised warm-start model per seed (made once,\n"
        "    # before round 1); it is also their KL reference. Old checkpoints without it keep the base model.\n"
        "    prover_base = cfg.prover_model\n"
        "    if cfg.use_finetunable_prover and cfg.prover_sft_epochs > 0:\n"
        "        from training.sft_warmstart import run_sft, sft_base_dir\n"
        "        sft_dir = sft_base_dir(ckpt)\n"
        "        if not (sft_dir / \"config.json\").exists() and not helpful_dir.exists():\n"
        "            run_sft(cfg, verifier_rows, sft_dir, seed=cfg.seed)\n"
        "        if (sft_dir / \"config.json\").exists():\n"
        "            prover_base = str(sft_dir)\n"
        "            print(f\"Provers start from the warm-start model {sft_dir}\")\n\n"
        "    # --- Provers ---\n",
        "sft warm start")
    P.replace(R, "load_prover_checkpoint(cfg, cfg.prover_model, helpful_dir)",
              "load_prover_checkpoint(cfg, prover_base, helpful_dir)  # " + MARK + " base h", "base h")
    P.replace(R, "load_prover_checkpoint(cfg, cfg.prover_model, sneaky_dir)",
              "load_prover_checkpoint(cfg, prover_base, sneaky_dir)  # " + MARK + " base s", "base s")
    P.replace(R,
        "        print(\"Initializing helpful prover fresh (round 1)...\")\n        helpful_prover_state = load_prover(cfg, cfg.prover_model)\n",
        "        print(\"Initializing helpful prover fresh (round 1)...\")\n        helpful_prover_state = load_prover(cfg, prover_base)  # " + MARK + " fresh h\n",
        "fresh h")
    P.replace(R,
        "            print(\"Initializing sneaky prover fresh (round 1)...\")\n            sneaky_prover_state = load_prover(cfg, cfg.prover_model)\n",
        "            print(\"Initializing sneaky prover fresh (round 1)...\")\n            sneaky_prover_state = load_prover(cfg, prover_base)  # " + MARK + " fresh s\n",
        "fresh s")

    # ---------------- new files ----------------
    P.write("training/sft_warmstart.py", SFT_WARMSTART, "warm-start module")
    P.write("data/build_model_written_rows.py", MODEL_WRITTEN_ROWS, "model-written rows builder")
    for name, rec in REAL_RECORDS.items():
        P.write(f"data/source_findings/{name}.json", json.dumps(rec, indent=2), f"real record {name}")
    return P


def run(cmd: List[str], root: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=root)
    if r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="project folder (contains config.py); found automatically")
    ap.add_argument("--no-data", action="store_true", help="patch only; skip rebuilding the dataset and model-written rows")
    ap.add_argument("--no-rows", action="store_true", help="skip generating model-written rows (needs a GPU)")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--no-pin", action="store_true", help="do not pin the project code to the commit the fixes target")
    args = ap.parse_args()

    root = find_root(args.root)
    print(f"Project: {root}")
    if not args.no_pin:
        pin_code(root)
    preflight(root)
    P = apply_all(root)
    print("\n".join(P.log))
    for rel in sorted(P.changed):
        if rel.endswith(".py"):
            py_compile.compile(str(root / rel), doraise=True)
    print("All patched files compile.\n")

    if not args.no_data:
        # Same command and seed as the notebook's data cell; deterministic, so re-running that cell is harmless.
        run([sys.executable, "data/build_findings_dataset.py", "--target-pairs", "120", "--seed", "42",
             "--out", "data/findings_pvg_dataset.jsonl", "--heldout-out", "data/findings_pvg_heldout.jsonl",
             "--manifest-out", "data/dataset_manifest.json"], root)
        if not args.no_rows:
            run([sys.executable, "data/build_model_written_rows.py", "--samples", str(args.samples), "--per-record", "2"], root)

    print("\nDone. In the round logs, check:")
    print("  - '[sft] after warm start: helpful fidelity' (aim for >= 0.5) before round 1")
    print("  - '[data] + N model-written train rows' and '[verifier labels] added SOUND=.. UNSOUND=..' (both > 0)")
    print("To reproduce the old behaviour for comparison, set in PVG_CONFIG_OVERRIDES:")
    print('  {"verifier_failed_helpful": "relabel", "verifier_round_balance": false, "model_written_train_path": "",')
    print('   "model_written_heldout_path": "", "prover_sft_epochs": 0, "prover_role_bonus": 0.0}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
