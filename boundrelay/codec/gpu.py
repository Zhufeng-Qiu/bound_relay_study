"""Triton encode/decode kernels.

Import-guarded: Triton needs CUDA, so on the Mac this module imports but
:func:`available` returns False and the test suite skips the ``gpu`` marker.
All development happens against the reference codec first — the GPU path is
measured, never trusted on its own.

The full property suite is re-run against this path on the A40 (Runbook D9-D11).
It is not enough to have tested the reference.

STATUS: D9-D11.
"""

from __future__ import annotations

import torch

from boundrelay.codec.contract import CodecConfig, EncodeStats

try:  # pragma: no cover - environment dependent
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


def available() -> bool:
    """True only where a Triton kernel can actually run."""
    return _HAS_TRITON and torch.cuda.is_available()


def encode(x: torch.Tensor, cfg: CodecConfig) -> tuple[bytes, EncodeStats]:
    """GPU encode. Must be bit-identical to :func:`reference.encode`'s codes."""
    raise NotImplementedError("D9")


def decode(payload: bytes) -> torch.Tensor:
    """GPU decode."""
    raise NotImplementedError("D9")


def benchmark(shapes: list[tuple[int, ...]], cfg: CodecConfig) -> dict:
    """Encode/decode throughput by shape — the cost-model lookup table (D15).

    Run this on **each** GPU generation used in the study. Pairing A40 codec
    timings with H100 link bandwidth puts the two sides of the break-even
    inequality on different machines, which would invalidate the atlas.
    """
    raise NotImplementedError("D15")
