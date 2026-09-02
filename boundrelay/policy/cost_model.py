"""The break-even model behind Figure 1.

    benefit = (S_bf16 - S_compressed) / BW_effective
    cost    = T_encode + T_decode + T_sync + safety_margin
    compress iff  benefit > cost  and  quality tier allowed

``BW_effective`` is swept as a **continuous free variable** from 0.1 to
1000 GB/s -- four orders of magnitude, from networked storage to NVLink. Only two
or three points on that axis are ever measured directly; the rest is model. So
Figure 1 draws measured points as markers with error bars and the model as a
line, distinguished in the legend. Blur that and the figure stops being analysis.

``t_encode`` / ``t_decode`` come from the per-GPU lookup table built by
``codec.gpu.benchmark``, measured on the same generation as the link being
modelled. Pairing A40 codec timings with H100 link bandwidth would put the two
sides of the inequality on different machines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Real fabric rates for the Figure 1 annotations, GB/s. Approximate and
#: deliberately so: they mark where a reader's hardware sits on the sweep. They
#: are not measurements -- confirm against the actual link before quoting one.
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

    Below it compression pays; above it the codec is pure overhead. This one
    number is the project's headline whether or not any speedup is observed.
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
