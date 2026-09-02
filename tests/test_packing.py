"""Sub-byte packing must be exact and must actually shrink the payload.

Storing 4-bit codes in int8 containers would report a ratio the transport never
sees. Since the entire project is about bytes that actually move, a fake ratio is
worse than no ratio at all.
"""

from __future__ import annotations

import pytest
import torch

from boundrelay.codec import packing
from tests.conftest import BIT_WIDTHS


@pytest.mark.parametrize("bit_width", BIT_WIDTHS)
@pytest.mark.parametrize("numel", [1, 7, 8, 255, 4096])
def test_pack_unpack_is_exact(bit_width, numel, gen):
    codes = torch.randint(0, 2**bit_width, (numel,), generator=gen, dtype=torch.int32)
    assert torch.equal(packing.unpack(packing.pack(codes, bit_width), bit_width, numel), codes)


@pytest.mark.parametrize("bit_width", BIT_WIDTHS)
def test_packed_size_matches_bit_width(bit_width):
    """No int8 container smuggling: 4-bit really is half of 8-bit."""
    assert packing.packed_bytes(4096, bit_width) == 4096 * bit_width // 8


def test_six_bit_is_not_rounded_up_to_a_byte():
    """The non-power-of-two width a naive packer gets wrong."""
    assert packing.packed_bytes(4, 6) == 3
