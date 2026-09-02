"""Table A — native transport benchmark.

Everything stays in its native bf16 layout; this is the table about bytes that
actually move and GPU-resident throughput.

    bf16_passthrough      2 B/element baseline
    int8_per_tensor       the naive quantisation
    int8_per_channel      the MLSys default -- NOT optional
    error_bounded 4/6/8   this codec

``int8_per_channel`` earns its row because it is the first thing any inference-
systems reader will ask about. A table without it invites "why is this better
than per-channel int8?" as the opening question of every conversation.

STATUS: D13.
"""

from __future__ import annotations

from pathlib import Path


def run(corpus: Path, out: Path) -> None:
    raise NotImplementedError("D13")
