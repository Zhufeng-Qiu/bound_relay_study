"""Break-even atlas — Figure 1.

Continuous bandwidth sweep from the cost model, validated against two or three
measured link points on 2xA40 (P2P where the host allows it, host-staged, and a
bandwidth-capped path). If ``p2pBandwidthLatencyTest`` shows P2P disabled on the
rented host, the validation set becomes {host-staged, capped x2} -- discovered on
day 6 for $0.50, not on day 18 after the sweep is designed around it.

STATUS: D17-D19.
"""

from __future__ import annotations

from pathlib import Path


def run(cost_table: Path, measured: Path, out: Path) -> None:
    raise NotImplementedError("D18")
