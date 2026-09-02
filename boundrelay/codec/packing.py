"""Sub-byte bit packing.

Below 8 bits the codec packs for real. Storing 4-bit codes in int8 containers
would report a compression ratio the transport never sees -- and the whole
project is about bytes that actually move, so a fake ratio is worse than none.

Widths: 4, 6, 8. Six is included deliberately. It is where the ratio/error trade
tends to land for activations, and it is the non-power-of-two case a naive packer
silently rounds up to a byte.

Implementation is bit-exact and vectorised via an explicit bit matrix: clarity
over speed, because this is the oracle the Triton kernel is checked against.
"""

from __future__ import annotations

import torch

SUPPORTED_WIDTHS: tuple[int, ...] = (4, 6, 8)


def packed_bytes(numel: int, bit_width: int) -> int:
    """Payload size in bytes, excluding metadata."""
    return (numel * bit_width + 7) // 8


def pack(codes: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Pack integer codes into a dense uint8 buffer, MSB-first within each code."""
    if bit_width not in SUPPORTED_WIDTHS:
        raise ValueError(f"unsupported bit_width {bit_width}")
    c = codes.reshape(-1).to(torch.int64)
    if c.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)

    shifts = torch.arange(bit_width - 1, -1, -1, dtype=torch.int64)
    bits = (c.unsqueeze(1) >> shifts) & 1          # (n, bit_width)

    flat = bits.reshape(-1)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])

    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.int64)
    return (flat.reshape(-1, 8) * weights).sum(1).to(torch.uint8)


def unpack(buf: torch.Tensor, bit_width: int, numel: int) -> torch.Tensor:
    """Inverse of :func:`pack`. Exact for every width and every ``numel``."""
    if bit_width not in SUPPORTED_WIDTHS:
        raise ValueError(f"unsupported bit_width {bit_width}")
    if numel == 0:
        return torch.empty(0, dtype=torch.int32)

    b = buf.reshape(-1).to(torch.int64)
    shifts = torch.arange(7, -1, -1, dtype=torch.int64)
    bits = ((b.unsqueeze(1) >> shifts) & 1).reshape(-1)[: numel * bit_width]

    weights = (2 ** torch.arange(bit_width - 1, -1, -1, dtype=torch.int64))
    return (bits.reshape(numel, bit_width) * weights).sum(1).to(torch.int32)
