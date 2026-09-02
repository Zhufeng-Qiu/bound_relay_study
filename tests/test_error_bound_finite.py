"""Suite 1 of 2 -- the eps guarantee, on FINITE blocks only.

Deliberately separate from ``test_nonfinite_bypass.py``. NaN/Inf blocks travel
bit-exact and sit outside the bound; letting them into these cases would make
the assertion pass for the wrong reason and hollow out the contract.
"""

from __future__ import annotations

import pytest
import torch

from boundrelay.codec import FP_TOLERANCE, Allocation, CodecConfig
from boundrelay.codec import reference
from tests.conftest import HIDDEN, hidden_state, with_channel_outliers


def roundtrip(x: torch.Tensor, cfg: CodecConfig) -> torch.Tensor:
    payload, _ = reference.encode(x, cfg)
    return reference.decode(payload)


@pytest.mark.parametrize("allocation", list(Allocation))
def test_bound_holds_on_gaussian(eps, allocation, gen):
    x = hidden_state(gen=gen)
    cfg = CodecConfig(eps=eps, allocation=allocation)
    assert reference.max_abs_error(x, roundtrip(x, cfg)) <= eps + FP_TOLERANCE


@pytest.mark.parametrize("allocation", list(Allocation))
def test_bound_holds_with_channel_outliers(eps, allocation, gen):
    """The distribution the codec actually has to survive."""
    x = with_channel_outliers(hidden_state(gen=gen))
    cfg = CodecConfig(eps=eps, allocation=allocation)
    assert reference.max_abs_error(x, roundtrip(x, cfg)) <= eps + FP_TOLERANCE


def test_constant_block(eps):
    """Zero range. The classic divide-by-zero in ``choose_bit_width``."""
    x = torch.full((32, HIDDEN), 1.5, dtype=torch.bfloat16)
    cfg = CodecConfig(eps=eps)
    assert reference.max_abs_error(x, roundtrip(x, cfg)) <= eps + FP_TOLERANCE


def test_extreme_magnitudes(eps):
    """bf16 carries fp32's exponent range; the codec must not assume fp16 scale."""
    x = torch.tensor([[-3.0e30, 1.0e-30, 0.0, 6.5e4, -6.5e4]], dtype=torch.bfloat16)
    cfg = CodecConfig(eps=eps)
    assert reference.max_abs_error(x, roundtrip(x, cfg)) <= eps + FP_TOLERANCE


def test_tiny_tensor_falls_back_not_crashes(eps):
    """Below the block size, bypass -- never a partial-block miscompute."""
    x = torch.randn(3, dtype=torch.bfloat16)
    _, stats = reference.encode(x, CodecConfig(eps=eps))
    assert stats.bypass_reason is not None


def test_non_contiguous_falls_back(eps):
    x = hidden_state().T  # transposed view, non-contiguous
    _, stats = reference.encode(x, CodecConfig(eps=eps))
    assert stats.bypass_reason is not None


def test_chunk_boundary_independence(eps, gen):
    """Splitting a tensor must not change any element's reconstruction."""
    x = hidden_state(tokens=128, gen=gen)
    cfg = CodecConfig(eps=eps)
    whole = roundtrip(x, cfg)
    halves = torch.cat([roundtrip(x[:64], cfg), roundtrip(x[64:], cfg)])
    assert torch.equal(whole, halves)


def test_ratio_denominator_is_bf16_bytes(eps, gen):
    """Ratio is reported against 2 bytes/element, never an fp32 upcast.

    Guards the Table B trap: baselines that read fp32 must not collect a free 2x.
    """
    x = hidden_state(gen=gen)
    _, stats = reference.encode(x, CodecConfig(eps=eps))
    assert stats.original_bytes == x.numel() * 2
