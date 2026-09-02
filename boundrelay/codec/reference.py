"""PyTorch reference codec — the correctness oracle for the GPU kernel.

Runs on CPU/MPS, so the whole error-bound contract is settled on the Mac before
a single GPU-hour is spent (Runbook D2-D3, Gate A).

Method: blockwise uniform quantisation. For a block with range
``R = max(x) - min(x)`` and ``b`` bits, the step is ``s = R / (2**b - 1)``.
Nearest rounding then bounds the reconstruction error by ``s / 2``, so the
smallest width satisfying a target eps is::

    b = ceil(log2(R / (2 * eps) + 1))

Block metadata (min, scale) is stored in **fp32**. Storing it in fp16 would let
the metadata's own representation error break the bound the codec advertises.

STATUS: D2. Signatures and invariants are fixed; the arithmetic is the work.
"""

from __future__ import annotations

import torch

from boundrelay.codec.contract import CodecConfig, EncodeStats


def choose_bit_width(block_range: torch.Tensor, eps: float) -> torch.Tensor:
    """Smallest per-block bit width whose quantisation step satisfies ``eps``.

    Args:
        block_range: ``max - min`` per block, any float dtype.
        eps: target absolute error bound, > 0.

    Returns:
        Integer tensor of widths, same shape as ``block_range``.
    """
    raise NotImplementedError("D2")


def encode(x: torch.Tensor, cfg: CodecConfig) -> tuple[bytes, EncodeStats]:
    """Compress ``x`` under ``cfg``.

    Fails closed: any unsupported dtype, non-contiguous layout, or failed bound
    check must return a ``bf16_passthrough`` payload with the corresponding
    ``BypassReason`` set, never a silently lossy one.

    Blocks holding NaN/Inf are emitted bit-exact and counted in
    ``EncodeStats.nonfinite_blocks``; they sit outside the eps guarantee.
    """
    raise NotImplementedError("D2")


def decode(payload: bytes) -> torch.Tensor:
    """Reconstruct a tensor from ``encode``'s payload. Must be deterministic."""
    raise NotImplementedError("D2")


def max_abs_error(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """``max |x - x_hat|`` over **finite** positions only.

    Non-finite positions are excluded by construction: they travel bit-exact, and
    ``|nan - nan|`` is not a number the contract can bound.
    """
    finite = torch.isfinite(x)
    if not bool(finite.any()):
        return 0.0
    return float((x[finite] - x_hat[finite]).abs().max())
