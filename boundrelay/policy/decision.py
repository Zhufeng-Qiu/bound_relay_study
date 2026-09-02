"""Per-chunk transport decision, and the structured log that explains it.

Kept deliberately small. Bandwidth EWMA, hysteresis, and a regret-analysis
framework were cut: with the atlas model-driven and the runtime integration a
stretch goal, they have nowhere to run, and the days they cost are scarcer than
the dollars.

What survives is the part that carries the argument — a cost model, a rule, and a
log entry saying *why*: bytes saved, predicted saving, codec cost, margin, and
either the chosen tier or a :class:`BypassReason`. A decision with no recorded
reason is a bug, not a default.

On evaluating this honestly
---------------------------
A controller scored against an oracle that shares its cost model has zero regret
by construction, and reporting that would be theatre. The one knob actually kept
is the safety margin, so that is what gets measured: how much latency the margin
gives up, and how many chunks it turns from compress into bypass, against a
zero-margin oracle over the measured payload × bandwidth grid. That is a real
question with a real answer, and it is the question a reviewer would ask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from boundrelay.codec.contract import BypassReason, Mode
from boundrelay.policy.cost_model import CostInputs, net_benefit_s


@dataclass
class Candidate:
    """One transport option with its measured cost at this payload."""

    mode: Mode
    label: str
    compressed_bytes: int
    t_encode_s: float
    t_decode_s: float
    eps: float | None = None
    quality_ok: bool = True


@dataclass
class Decision:
    mode: Mode
    label: str
    eps: float | None
    predicted_saving_s: float
    codec_cost_s: float
    bytes_before: int
    bytes_after: int
    bw_gbps: float
    margin_s: float
    bypass_reason: BypassReason | None = None
    considered: list[dict] = field(default_factory=list)

    def as_log(self) -> dict:
        return asdict(self)


def decide(original_bytes: int, bw_gbps: float, candidates: list[Candidate],
           margin_s: float = 0.0) -> Decision:
    """Choose a transport mode for one chunk, and record why.

    Bypass is not a failure branch: it is the correct answer whenever no
    candidate's predicted saving clears its own cost plus the margin.
    """
    best: Candidate | None = None
    best_net = 0.0
    considered: list[dict] = []

    for c in candidates:
        if c.mode is Mode.BF16_PASSTHROUGH:
            continue
        if not c.quality_ok:
            considered.append({"label": c.label, "net_s": None,
                               "skipped": BypassReason.QUALITY_TIER_DISALLOWED.value})
            continue
        net = net_benefit_s(CostInputs(
            original_bytes, c.compressed_bytes, bw_gbps,
            c.t_encode_s, c.t_decode_s, 0.0, margin_s))
        considered.append({"label": c.label, "net_s": net})
        if net > best_net:
            best, best_net = c, net

    if best is None:
        return Decision(
            mode=Mode.BF16_PASSTHROUGH, label="bf16_passthrough", eps=None,
            predicted_saving_s=0.0, codec_cost_s=0.0,
            bytes_before=original_bytes, bytes_after=original_bytes,
            bw_gbps=bw_gbps, margin_s=margin_s,
            bypass_reason=BypassReason.NOT_PROFITABLE, considered=considered)

    return Decision(
        mode=best.mode, label=best.label, eps=best.eps,
        predicted_saving_s=best_net,
        codec_cost_s=best.t_encode_s + best.t_decode_s,
        bytes_before=original_bytes, bytes_after=best.compressed_bytes,
        bw_gbps=bw_gbps, margin_s=margin_s, considered=considered)


def oracle(original_bytes: int, bw_gbps: float,
           candidates: list[Candidate]) -> Decision:
    """Best achievable choice with no safety margin — the upper bound."""
    return decide(original_bytes, bw_gbps, candidates, margin_s=0.0)
