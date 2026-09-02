"""Verify the Triton kernels' bit addressing without a GPU.

The pack/unpack kernels were written against an API that could not be executed on
the development machine, and bit addressing is where such code goes wrong: an
off-by-one in "which bit of which code" produces a payload that decodes to
plausible-looking garbage rather than an error.

So the kernels' *algorithm* is restated here in torch, element for element, and
checked against :mod:`boundrelay.codec.packing`. This does not prove the Triton
code compiles or that ``rint`` matches ``torch.round`` — only a GPU can settle
those, and ``test_gpu_equivalence.py`` does. It does mean that when the first GPU
session fails, the failure is not this.
"""

from __future__ import annotations

import pytest
import torch

from boundrelay.codec import packing
from tests.conftest import BIT_WIDTHS


def _pack_like_kernel(codes: torch.Tensor, w: int) -> torch.Tensor:
    """Mirror of ``_pack_kernel``: one lane per output byte, eight bits each."""
    numel = codes.numel()
    n_out = packing.packed_bytes(numel, w)
    j = torch.arange(n_out, dtype=torch.int64)

    acc = torch.zeros(n_out, dtype=torch.int64)
    for b in range(8):
        gbit = j * 8 + b
        ci = gbit // w
        bi = gbit % w
        inb = ci < numel
        c = torch.where(inb, codes.to(torch.int64)[ci.clamp(max=numel - 1)],
                        torch.zeros_like(ci))
        acc |= ((c >> (w - 1 - bi)) & 1) << (7 - b)
    return acc.to(torch.uint8)


def _unpack_like_kernel(buf: torch.Tensor, w: int, numel: int) -> torch.Tensor:
    """Mirror of ``_unpack_dequant_kernel``: one lane per code, w bits each."""
    n_bytes = buf.numel()
    i = torch.arange(numel, dtype=torch.int64)

    acc = torch.zeros(numel, dtype=torch.int64)
    for b in range(w):
        gbit = i * w + b
        byi = gbit // 8
        bib = gbit % 8
        inb = byi < n_bytes
        byte = torch.where(inb, buf.to(torch.int64)[byi.clamp(max=n_bytes - 1)],
                           torch.zeros_like(byi))
        acc |= ((byte >> (7 - bib)) & 1) << (w - 1 - b)
    return acc.to(torch.int32)


@pytest.mark.parametrize("w", BIT_WIDTHS)
@pytest.mark.parametrize("numel", [1, 3, 7, 8, 17, 256, 4097])
def test_kernel_pack_layout_matches_reference(w, numel, gen):
    codes = torch.randint(0, 2**w, (numel,), generator=gen, dtype=torch.int32)
    assert torch.equal(_pack_like_kernel(codes, w), packing.pack(codes, w))


@pytest.mark.parametrize("w", BIT_WIDTHS)
@pytest.mark.parametrize("numel", [1, 3, 7, 8, 17, 256, 4097])
def test_kernel_unpack_layout_matches_reference(w, numel, gen):
    codes = torch.randint(0, 2**w, (numel,), generator=gen, dtype=torch.int32)
    buf = packing.pack(codes, w)
    assert torch.equal(_unpack_like_kernel(buf, w, numel), codes)


@pytest.mark.parametrize("w", BIT_WIDTHS)
def test_kernel_pack_unpack_roundtrip(w, gen):
    """The two kernel algorithms must invert each other, not just match a third party."""
    codes = torch.randint(0, 2**w, (1000,), generator=gen, dtype=torch.int32)
    assert torch.equal(_unpack_like_kernel(_pack_like_kernel(codes, w), w, 1000), codes)
