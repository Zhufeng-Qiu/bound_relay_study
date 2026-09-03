"""The break-even model behind Figure 1 — and the four assumptions this project
measured and could not sustain.

    benefit = (S_bf16 - S_compressed) / BW_effective
    cost    = T_encode + T_decode + T_sync + safety_margin
    compress iff  benefit > cost  and  quality tier allowed

    break_even_BW = (S_bf16 - S_compressed) / (T_encode + T_decode + T_sync + margin)

**Read the module's results before using this to decide anything.** The code below
is kept because it is what the project set out to build and what several
experiments were designed against; it is not kept because it works. Four
assumptions are load-bearing and all four failed, every one of them in the
direction that flatters compression:

1. ``T_encode`` is a property of the codec. It is not: measured on one 20 MB tensor
   the codec runs at 40 GB/s, over the real 56-tensor cache at 11.8 GB/s -- **3.4x**.
   Granularity is a hidden argument. (`c_transport_findings.md`)
2. Stage costs compose. A pipeline built from these same measured stages was
   predicted at 21.43 ms and measured at **43.65 ms**, with both threads saturated
   and under 2.2 ms of idling, so it is not a scheduling failure with headroom in
   it. A stage sum contains no per-item cost. (`pipeline_findings.md`)
3. ``BW_effective`` is exogenous. It is not: it is a function of ``S_compressed``,
   the quantity the decision sets. The same incompressible bytes achieve **0.55x**
   the throughput at 77.7 MB that they do at 234.9 MB on a network mount under
   ``fsync``. The denominator depends on the numerator, so ``break_even_BW`` is not
   a constant of the path and cannot be looked up.
   (`fsync_offload_findings.md`)
4. The saving is the bytes. Per-object cost is absent from the expression and
   dominated a real offload: fifty-six files cost **4.3x** one file on identical
   bytes. (`pipeline_findings.md`)

Assumptions 1 and 3 are the same error at opposite ends of the stack -- a rate
treated as a property of a component when it is a property of a component *in a
configuration*.

What survives is the **sign**, not the magnitude. Compression lost on the
host-staged GPU path (1.97x slower) and won on a filesystem offload write
(1.21-2.10x faster) in every configuration measured, which is the direction this
model gives. What it cannot do is tell you where the crossing is without measuring
the whole path at the granularity and durability you will actually run.

``BW_effective`` is swept below as a continuous free variable from 0.1 to
1000 GB/s. Given (3), **that sweep is not a line a reader can locate their hardware
on**, and ``FABRIC_GBPS`` invites exactly that mistake. Both are kept, annotated,
because the figure they produce is the claim this project set out to test.

``t_encode`` / ``t_decode`` come from the per-GPU lookup table built by
``codec.gpu.benchmark``, measured on the same generation as the link being
modelled. Pairing A40 codec timings with H100 link bandwidth would put the two
sides of the inequality on different machines -- a smaller version of the same
mistake as (1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Real fabric rates for the Figure 1 annotations, GB/s. Approximate and
#: deliberately so.
#:
#: **These are not work points.** They were written to mark "where a reader's
#: hardware sits on the sweep", and assumption (3) above says a fabric does not
#: have a single rate: the same mount delivered 0.255 GB/s to a 77.7 MB write and
#: 0.468 GB/s to a 234.9 MB one, and 3.27 GB/s without ``fsync`` against 1.81 with
#: it. A row here names a family of hardware, not a number you can divide bytes by.
FABRIC_GBPS: dict[str, float] = {
    "network storage": 0.5,
    "NVMe": 7.0,
    "100G Eth": 12.5,
    "400G IB": 50.0,
    "PCIe Gen5": 64.0,
    "NVLink": 450.0,
}

_GB = 1e9


@dataclass
class CostInputs:
    original_bytes: int
    compressed_bytes: int
    bw_gbps: float
    t_encode_s: float
    t_decode_s: float
    t_sync_s: float = 0.0
    safety_margin_s: float = 0.0

    @property
    def codec_cost_s(self) -> float:
        return self.t_encode_s + self.t_decode_s + self.t_sync_s + self.safety_margin_s


def transfer_s(nbytes: int, bw_gbps: float) -> float:
    return nbytes / (bw_gbps * _GB)


def net_benefit_s(c: CostInputs) -> float:
    """Seconds saved (positive) or lost (negative) by compressing this chunk."""
    saved = transfer_s(c.original_bytes - c.compressed_bytes, c.bw_gbps)
    return saved - c.codec_cost_s


def break_even_bw_gbps(c: CostInputs) -> float:
    """Bandwidth at which :func:`net_benefit_s` crosses zero.

    Below it compression pays; above it the codec is pure overhead.

    This was the project's headline. It is now one of its retractions: the quantity
    is well defined only if ``BW`` is independent of ``compressed_bytes``, and it is
    not. The number this returns is the crossing **for the granularity, payload size
    and durability at which its inputs were measured**, and it moves when any of
    those change -- 3.82 GB/s serial, 5.38 GB/s pipelined, on the same hardware and
    the same cache. Report it with those conditions attached or not at all.

    Returns ``inf`` when the codec is free and ``0.0`` when it can never pay.
    """
    delta = c.original_bytes - c.compressed_bytes
    if delta <= 0:
        return 0.0
    if c.codec_cost_s <= 0:
        return math.inf
    return delta / (c.codec_cost_s * _GB)


def sweep(c: CostInputs, lo_gbps: float = 0.1, hi_gbps: float = 1000.0, n: int = 256) -> dict:
    """Log-spaced sweep of :func:`net_benefit_s` over bandwidth. Feeds Figure 1."""
    step = (math.log10(hi_gbps) - math.log10(lo_gbps)) / (n - 1)
    bws = [10 ** (math.log10(lo_gbps) + i * step) for i in range(n)]
    ys = []
    for bw in bws:
        ys.append(net_benefit_s(CostInputs(
            c.original_bytes, c.compressed_bytes, bw,
            c.t_encode_s, c.t_decode_s, c.t_sync_s, c.safety_margin_s)))
    return {
        "bw_gbps": bws,
        "net_benefit_s": ys,
        "break_even_gbps": break_even_bw_gbps(c),
        "original_bytes": c.original_bytes,
        "compressed_bytes": c.compressed_bytes,
        "codec_cost_s": c.codec_cost_s,
    }
