"""Figure 2 -- representation design.

Ratio at matched eps for {blockwise, per-channel, per-token} x
{hidden state, KV cache}, alongside the baselines from Tables A and B.

This is the figure that carries the research judgement: it says *why* generic
scientific compressors underperform on this data (their inductive bias is
spatial smoothness; these tensors are not smooth along the token axis) and what
design does fit (the exploitable structure is per-channel). Entirely offline, so
it cannot be lost to a failed GPU session.

STATUS: D16 (skeleton at D4).
"""

from __future__ import annotations

from pathlib import Path


def render(alloc_json: Path, table_a: Path, table_b: Path, out: Path) -> None:
    raise NotImplementedError("D4 skeleton / D16 final")
