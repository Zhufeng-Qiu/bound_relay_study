"""Error-budget allocation -- the Figure 2 independent variable.

Three ways to spend a global eps across a tensor:

``BLOCKWISE``
    Flat 1-D blocks over the raveled tensor. What a generic scientific
    compressor effectively degenerates to once its smoothness prediction fails.

``PER_CHANNEL``
    One budget per hidden dimension. The hypothesis: LLM intermediate states
    carry their exploitable structure on the channel axis (activation outliers),
    not the token axis.

``PER_TOKEN``
    One budget per token. The control that separates "channel structure helps"
    from "finer granularity helps".

A property worth knowing before it becomes a bug: **per-channel allocation is not
chunk-independent.** Its statistics are computed over whichever tokens are in
hand, so splitting a stream into chunks changes them and two different chunkings
reconstruct differently. Blockwise and per-token are chunk-independent when the
split falls on a block or token boundary. For a streaming transport this is a
real limitation of per-channel, not a detail -- it belongs in the note's
limitations section alongside whatever ratio advantage it turns out to buy.
"""

from __future__ import annotations

import torch

from boundrelay.codec.contract import Allocation


def partition(x: torch.Tensor, allocation: Allocation, block_size: int) -> torch.Tensor:
    """Map each element of ``x`` to the id of the budget group it draws from.

    Returns a flat int64 tensor of length ``x.numel()``.
    """
    n = x.numel()
    if allocation is Allocation.BLOCKWISE:
        return torch.arange(n, dtype=torch.int64) // block_size

    if x.dim() < 2:
        # No channel/token axes to speak of; degenerate to blockwise rather than
        # inventing a grouping the caller did not ask for.
        return torch.arange(n, dtype=torch.int64) // block_size

    channels = x.shape[-1]
    rows = n // channels
    if allocation is Allocation.PER_CHANNEL:
        return torch.arange(channels, dtype=torch.int64).repeat(rows)
    if allocation is Allocation.PER_TOKEN:
        return torch.arange(rows, dtype=torch.int64).repeat_interleave(channels)

    raise ValueError(f"unknown allocation {allocation}")


def group_bounds(values: torch.Tensor, groups: torch.Tensor, n_groups: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group ``(min, max)`` in float32. Non-finite inputs must be excluded first."""
    v = values.reshape(-1).float()
    g = groups.reshape(-1)
    gmin = torch.full((n_groups,), float("inf")).scatter_reduce(0, g, v, reduce="amin")
    gmax = torch.full((n_groups,), float("-inf")).scatter_reduce(0, g, v, reduce="amax")
    return gmin, gmax


def group_ranges(x: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Per-group ``max - min``. Feeds :func:`reference.choose_bit_width`."""
    n = int(groups.max()) + 1
    gmin, gmax = group_bounds(x, groups, n)
    return gmax - gmin
