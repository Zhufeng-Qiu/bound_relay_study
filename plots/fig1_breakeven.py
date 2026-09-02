"""Figure 1 -- break-even map.

x: effective bandwidth, log, 0.1 -> 1000 GB/s (four orders of magnitude)
y: net latency benefit, zero line drawn

Two rules this figure lives or dies by:

1. Measured points are markers with error bars; the cost model is a line. The
   legend distinguishes them. Blur that and the figure stops being analysis.
2. Real fabric positions are annotated -- networked storage, NVMe, 100G, 400G IB,
   PCIe Gen5, NVLink. A few axvlines; it is what lets one figure answer three
   different audiences' question at once.

STATUS: D19 (runs on synthetic data at D4, before any GPU spend).
"""

from __future__ import annotations

from pathlib import Path


def render(sweep_json: Path, measured_json: Path, out: Path) -> None:
    raise NotImplementedError("D4 skeleton / D19 final")
