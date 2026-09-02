"""Suite 2 of 2 -- NaN/Inf handling.

``|x - x_hat| <= eps`` is undefined when x is NaN or Inf, so these blocks are not
compressed at all: they travel bit-exact and are counted. The contract states the
exclusion explicitly rather than leaving it to be discovered.

A nonzero ``nonfinite_block_rate`` on a healthy model is a finding worth
reporting, not noise to smooth over.
"""

from __future__ import annotations

import pytest
import torch

from boundrelay.codec import CodecConfig
from boundrelay.codec import reference
from tests.conftest import HIDDEN, hidden_state


def roundtrip(x: torch.Tensor, cfg: CodecConfig) -> torch.Tensor:
    payload, _ = reference.encode(x, cfg)
    return reference.decode(payload)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_block_is_bit_exact(bad, gen):
    x = hidden_state(gen=gen)
    x[0, 0] = bad
    out = roundtrip(x, CodecConfig(eps=1e-2))
    head = out[0, :HIDDEN]
    assert torch.equal(head.view(torch.int16), x[0, :HIDDEN].view(torch.int16))


def test_nonfinite_blocks_are_counted(gen):
    x = hidden_state(gen=gen)
    x[0, 0] = float("nan")
    _, stats = reference.encode(x, CodecConfig(eps=1e-2))
    assert stats.nonfinite_blocks >= 1
    assert 0.0 < stats.nonfinite_block_rate <= 1.0


def test_all_finite_reports_zero_nonfinite(gen):
    _, stats = reference.encode(hidden_state(gen=gen), CodecConfig(eps=1e-2))
    assert stats.nonfinite_blocks == 0


def test_finite_blocks_still_compress_alongside_nonfinite(gen):
    """One bad value must not force the whole tensor onto the bypass path."""
    x = hidden_state(tokens=128, gen=gen)
    x[0, 0] = float("inf")
    _, stats = reference.encode(x, CodecConfig(eps=1e-2))
    assert stats.nonfinite_blocks < stats.total_blocks
