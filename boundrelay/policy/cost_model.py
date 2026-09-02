"""The break-even model behind Figure 1.

    benefit = (S_bf16 - S_compressed) / BW_effective
    cost    = T_encode + T_decode + T_sync + safety_margin
    compress iff  benefit > cost  and  quality tier allowed

``BW_effective`` is swept as a **continuous free variable** from 0.1 to
1000 GB/s — four orders of magnitude, covering everything from networked storage
to NVLink. Only two or three points on that axis are ever measured directly; the
rest is the model. Figure 1 must therefore draw measured points as markers with
error bars and the model as a line, distinguished in the legend. Blur that line
and the figure stops being analysis.

``T_encode`` / ``T_decode`` come from the per-GPU lookup table built by
``codec.gpu.benchmark`` — measured on the same generation as the link being
modelled, never mixed.

STATUS: D17-D19.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Real fabric rates for the Figure 1 annotations, GB/s. Approximate and
#: deliberately so: they mark where the reader's hardware sits on the sweep, they
#: are not measurements. Confirm against the actual link before quoting any.
FABRIC_GBPS: dict[str, float] = {
    "network_storage": 0.5,
    "nvme": 7.0,
    "eth_100g": 12.5,
    "ib_400g": 50.0,
    "pcie_gen5_x16": 64.0,
    "nvlink": 450.0,
}


@dataclass
class CostInputs:
    original_bytes: int
    compressed_bytes: int
    bw_gbps: float
    t_encode_s: float
    t_decode_s: float
    t_sync_s: float = 0.0
    safety_margin_s: float = 0.0


def net_benefit_s(c: CostInputs) -> float:
    """Seconds saved (positive) or lost (negative) by compressing this chunk."""
    raise NotImplementedError("D17")


def break_even_bw_gbps(c: CostInputs) -> float:
    """Bandwidth at which ``net_benefit_s`` crosses zero.

    Below it compression pays; above it the codec is pure overhead. This single
    number is the project's headline whether or not any speedup is observed.
    """
    raise NotImplementedError("D17")


def sweep(c: CostInputs, lo_gbps: float = 0.1, hi_gbps: float = 1000.0, n: int = 256) -> dict:
    """Log-spaced sweep of ``net_benefit_s`` over bandwidth. Feeds Figure 1."""
    raise NotImplementedError("D17")
