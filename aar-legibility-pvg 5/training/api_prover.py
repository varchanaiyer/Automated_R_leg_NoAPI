"""
Frozen API prover: Claude (via OpenRouter) plays BOTH prover roles from the
same helpful / sneaky prompts the local Qwen prover uses. No weights change;
only the verifier trains (LIMITATIONS.md #5). Used when
cfg.use_finetunable_prover is False.

Every completion is cached on disk (one JSON per prompt group) under the
checkpoint directory, keyed by seed / round / role / record / sample index,
so re-running a round after a crash or disconnect costs nothing.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from config import PVGConfig

_REFUSAL_MARKERS = ("i can't help", "i cannot help", "i can\u2019t help", "i won't", "i will not",
                    "i'm not able to", "i am not able to", "as an ai", "i can't write", "i cannot write")


def looks_like_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    return (not t) or any(m in t[:200] for m in _REFUSAL_MARKERS)


class APIProver:
    def __init__(self, cfg: PVGConfig, cache_dir):
        self.cfg = cfg
        self.model = cfg.prover_api_model
        self.url = cfg.prover_api_url
        self.key = os.environ.get(cfg.prover_api_key_env, "").strip()
        if not self.key:
            raise RuntimeError(f"{cfg.prover_api_key_env} is not set; the frozen API prover needs it")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "cost_usd": 0.0, "errors": 0, "refusals": 0}

    # -- one raw call ------------------------------------------------------
    def _call(self, prompt_text: str, temperature: float, max_tokens: int) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.cfg.prover_api_system},
                         {"role": "user", "content": prompt_text}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if self.cfg.prover_api_reasoning_effort:
            body["reasoning"] = {"effort": self.cfg.prover_api_reasoning_effort}
        data = json.dumps(body).encode()
        headers = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json",
                   "HTTP-Referer": "https://github.com/gunasti2002/Automated_R_leg_NoAPI",
                   "X-Title": "aar-legibility-pvg"}
        delay = 2.0
        last_err = None
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as r:
                    out = json.load(r)
                if "error" in out and not out.get("choices"):
                    raise RuntimeError(f"API error: {out['error']}")
                return out
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read().decode()[:300]}"
                if e.code in (400, 401, 402, 403, 404):
                    break
            except Exception as e:  # network / 5xx / parse
                last_err = repr(e)
            self.usage["errors"] += 1
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")

    # -- n completions for one prompt, cached ---------------------------------
    def complete(self, prompt_text: str, n: int, temperature: float, cache_key: str,
                 max_tokens: Optional[int] = None) -> List[str]:
        max_tokens = max_tokens or self.cfg.prover_api_max_tokens
        key = hashlib.sha1((f"{self.model}|{cache_key}|{n}|{temperature}|{max_tokens}|"
                            f"{self.cfg.prover_api_reasoning_effort}|{self.cfg.prover_api_system}|{prompt_text}").encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.usage["cache_hits"] += n
            return json.loads(path.read_text())["completions"]
        completions = []
        for i in range(n):
            text, finish = "", None
            for budget in (max_tokens, max_tokens * 2):   # one retry if reasoning ate the whole budget
                out = self._call(prompt_text, temperature, budget)
                choice = out["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                finish = choice.get("finish_reason")
                u = out.get("usage") or {}
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                self.usage["cost_usd"] += float(u.get("cost") or 0.0)
                served = out.get("model")
                if served:
                    self.usage.setdefault("served_models", {})
                    self.usage["served_models"][served] = self.usage["served_models"].get(served, 0) + 1
                if text or finish != "length":
                    break
            if finish in ("content_filter", "refusal") or looks_like_refusal(text):
                self.usage["refusals"] += 1
            completions.append(text)
        path.write_text(json.dumps({"model": self.model, "cache_key": cache_key, "prompt": prompt_text,
                                    "completions": completions}))
        return completions

    def save_usage(self, path) -> None:
        Path(path).write_text(json.dumps(self.usage, indent=2))
