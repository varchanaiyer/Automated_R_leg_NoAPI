"""
Hooks the TRAINED verifier from this PVG pipeline into the real AAR's
share_finding tool, in the automated-w2s-research repo.

This does NOT retrain the AAR itself — the AAR (Claude, called via API in
the real automated-w2s-research setup) is a frozen model, not fine-tuned
here (see LIMITATIONS.md #5). This is a submission GATE: it rejects
findings the trained verifier scores as unsound, forcing the AAR to
revise before a finding is accepted, the same way share_finding already
rejects "result" findings with fewer than 5 seeds.

Scoring is the same log-probability comparison training/pvg_loop.py uses
(P(sound) = sigmoid(logp("SOUND") - logp("UNSOUND")) under the chat
template). No free-text verdict is generated or parsed — the old substring
parser is what accepted everything during the degenerate-verifier incident.

Integration point (in the real repo):
    w2s_research/research_loop/tools/server_api_tools.py
    -> share_finding(), inside the `if finding_type == "result":` block,
       right after the existing num_seeds check.
"""
import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PVGConfig

_LOADED_DIR: Optional[str] = None


def render_verifier_prompt(summary: str, metrics: dict, config: dict) -> str:
    from training.pvg_loop import _render_verifier_prompt
    return _render_verifier_prompt(summary, metrics, config)


def _ensure_verifier(cfg: PVGConfig, verifier_dir: Optional[str]) -> None:
    """Loads the trained adapter once (untrained base if no checkpoint is given)."""
    global _LOADED_DIR
    from training import pvg_loop
    target = verifier_dir or cfg.trained_verifier_dir
    if _LOADED_DIR == target:
        return
    if target and Path(target).exists():
        pvg_loop.load_verifier_checkpoint(cfg, Path(target))
    else:
        print(f"[share_finding_gate] no trained verifier at {target!r}; using the untrained base model")
        pvg_loop._get_verifier(cfg.verifier_model, cfg)
    _LOADED_DIR = target


def score_finding_with_verifier_sync(summary: str, metrics: dict, config: dict, cfg: PVGConfig,
                                     verifier_dir: Optional[str] = None) -> Optional[float]:
    """P(sound) from the trained verifier, or None if the verdict is unparseable."""
    from training.pvg_loop import score_p_sound
    _ensure_verifier(cfg, verifier_dir)
    return score_p_sound(summary, metrics or {}, config or {}, cfg.verifier_model, cfg)


async def score_finding_with_verifier(summary: str, metrics: dict, config: dict, cfg: PVGConfig,
                                      verifier_dir: Optional[str] = None) -> Optional[float]:
    """Async wrapper matching share_finding's call site."""
    return score_finding_with_verifier_sync(summary, metrics, config, cfg, verifier_dir)


# ---------------------------------------------------------------------------
# The actual patch to apply inside share_finding() in server_api_tools.py
# ---------------------------------------------------------------------------
PATCH_SNIPPET = '''
# --- INSERT into share_finding(), right after the existing num_seeds check ---
if finding_type == "result":
    from aar_legibility_pvg.integration.share_finding_gate import score_finding_with_verifier
    from aar_legibility_pvg.config import PVGConfig

    verifier_score = await score_finding_with_verifier(
        summary=summary, metrics=metrics or {}, config=config or {}, cfg=PVGConfig(),
    )
    if verifier_score is not None and verifier_score < PVGConfig().gate_reject_threshold:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": False,
                    "error": (
                        f"Rejected: finding scored {verifier_score:.2f} on the trained "
                        f"legibility verifier (threshold {PVGConfig().gate_reject_threshold}). "
                        f"Check that every specific claim is directly supported by the "
                        f"metrics/config you cited before resubmitting."
                    ),
                }, indent=2)
            }]
        }
    # Surface the score on the leaderboard/forum too, not just as a gate.
    payload["legibility_verifier_score"] = verifier_score
# --- END INSERT ---
'''

if __name__ == "__main__":
    print("This module is meant to be imported from a patched copy of")
    print("server_api_tools.py in automated-w2s-research. See PATCH_SNIPPET")
    print("above for the exact code to insert, and the module docstring for")
    print("where it goes.")
    print()
    print(PATCH_SNIPPET)
