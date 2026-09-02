"""Sub-byte bit packing.

Below 8 bits the codec packs for real. Storing 4-bit codes in int8 containers
would report a compression ratio the transport never sees — the whole project is
about bytes actually moved, so a fake ratio is worse than no ratio.

Widths: 4, 6, 8. Six is included because it is where the ratio/error trade tends
to sit for activations, and because a non-power-of-two width is the case a naive
packer gets wrong.

STATUS: D9-D11 (reference here; Triton in ``gpu.py``).
"""

from __future__ import annotations

import torch


def pack(codes: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Pack integer codes into a dense uint8 buffer. Widths 4, 6, 8."""
    raise NotImplementedError("D9")


def unpack(buf: torch.Tensor, bit_width: int, numel: int) -> torch.Tensor:
    """Inverse of :func:`pack`. ``unpack(pack(c)) == c`` exactly, for all widths."""
    raise NotImplementedError("D9")


def packed_bytes(numel: int, bit_width: int) -> int:
    """Payload size in bytes, excluding metadata. Pure arithmetic, safe to use now."""
    return (numel * bit_width + 7) // 8
