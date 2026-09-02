"""Figure 3 -- quality frontier.

x: eps (or achieved compression ratio)
y: perplexity delta (always available) and paired WER/CER (only if the H100
   session lands)

Two legs by design. The perplexity leg runs on a 1-3B model on an A40 and does
not depend on the runtime integration, so the error-to-quality story -- the part
closest to Sian Jin's own line -- cannot be lost with the stretch goal.

STATUS: D21 (skeleton at D4).
"""

from __future__ import annotations

from pathlib import Path


def render(quality_json: Path, out: Path) -> None:
    raise NotImplementedError("D4 skeleton / D21 final")
