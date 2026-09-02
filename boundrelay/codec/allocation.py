"""Error-budget allocation — the Figure 2 independent variable.

Three ways to spend a global eps across a tensor:

``BLOCKWISE``
    Flat 1-D blocks over the raveled tensor. What a generic scientific
    compressor effectively does once its smoothness prediction fails.

``PER_CHANNEL``
    One budget per hidden dimension. The hypothesis: LLM intermediate states
    carry their exploitable structure on the channel axis (activation outliers),
    not on the token axis.

``PER_TOKEN``
    One budget per token. The control that isolates whether the gain is really
    channel-specific rather than just "finer granularity helps".

Applied to both tensor families (hidden states, KV cache) at three eps values:
3 x 2 x 3 = 18 cells, all offline.

STATUS: D12.
"""

from __future__ import annotations

import torch

from boundrelay.codec.contract import Allocation


def partition(x: torch.Tensor, allocation: Allocation, block_size: int) -> torch.Tensor:
    """Map elements of ``x`` to budget-group ids.

    Returns an int tensor shaped like ``x`` whose values index the group each
    element's error budget is drawn from.
    """
    raise NotImplementedError("D12")


def group_ranges(x: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Per-group ``max - min``. Feeds :func:`reference.choose_bit_width`."""
    raise NotImplementedError("D12")
