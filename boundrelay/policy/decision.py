"""Per-chunk transport decision, plus the structured log that explains it.

Kept deliberately small. Bandwidth EWMA, hysteresis, and a regret-analysis
framework were cut (Runbook D21): with the atlas model-driven and the runtime
integration a stretch goal, they have nowhere to run, and the two or three days
they cost are scarcer than the dollars.

What survives is the part that carries the argument: a cost model, a rule, and a
log entry that says *why* — bytes saved, predicted saving, codec cost, margin,
and either the chosen tier or a ``BypassReason``. A decision with no recorded
reason is a bug, not a default.

STATUS: D21.
"""

from __future__ import annotations

from dataclasses import dataclass

from boundrelay.codec.contract import BypassReason, Mode
from boundrelay.policy.cost_model import CostInputs


@dataclass
class Decision:
    mode: Mode
    eps: float | None
    predicted_saving_s: float
    codec_cost_s: float
    bytes_before: int
    bytes_after: int
    bypass_reason: BypassReason | None = None


def decide(c: CostInputs, quality_tier_allowed: bool) -> Decision:
    """Choose a transport mode for one chunk."""
    raise NotImplementedError("D21")


def oracle(candidates: list[CostInputs]) -> Decision:
    """Best mode chosen with hindsight from measured times — the replay upper bound."""
    raise NotImplementedError("D21")
