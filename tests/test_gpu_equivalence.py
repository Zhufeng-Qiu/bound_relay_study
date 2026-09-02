"""GPU path vs reference — the gate that must pass before any GPU number is quoted.

Skipped on the laptop. The first GPU session runs this *before* the throughput
benchmark, because a fast kernel that disagrees with the oracle is worth nothing.

Why this does not demand byte-identical payloads
------------------------------------------------
It did at first, and that test failed for a reason worth recording rather than
tuning away. With bit-identical float32 inputs, ``range / 255`` differs between
CPU and CUDA by one float32 ulp on roughly 60% of groups (measured: 302 of 512).
The Triton reductions themselves are exact — ``tl.min``/``tl.max`` matched
``scatter_reduce`` element for element — so the divergence is the division, not
the kernel. One ulp on the scale is enough to flip a value sitting exactly on a
rounding boundary, changing a code by ±1.

Forcing agreement would mean computing every scale on the host and shipping it to
the device: a synchronisation in the middle of encode, added to make a test pass
rather than to meet a requirement. The contract says ``max|x - x̂| <= eps``, and a
±1 code shift stays comfortably inside it.

So the strict assertions moved to the properties the system actually depends on:

* both encoders make the **same structural decisions** — identical widths,
  identical raw-group set, identical payload size;
* each decoder reads the **other's** payload, which is what a codec split across
  two devices requires;
* both outputs honour the error bound independently.

Codes may differ, by at most one step, on a minority of values. That the encoder
is not bit-reproducible across devices is a real property of this codec and
belongs in the note's reproducibility caveat — not something to hide behind a
loosened assertion.
"""

from __future__ import annotations

import pytest
import torch

from boundrelay.codec import Allocation, CodecConfig, FP_TOLERANCE, gpu, reference, serialize
from tests.conftest import BIT_WIDTHS, hidden_state, with_channel_outliers

pytestmark = pytest.mark.gpu

if not gpu.available():  # pragma: no cover - laptop
    pytest.skip("Triton/CUDA unavailable", allow_module_level=True)

EPS = 0.15


def _cfg(w: int) -> CodecConfig:
    return CodecConfig(eps=EPS, allocation=Allocation.BLOCKWISE, bit_width=w)


def _pair(x: torch.Tensor, cfg: CodecConfig):
    return reference.encode(x.cpu(), cfg), gpu.encode(x.cuda().contiguous(), cfg)


def _widths(buf: bytes) -> torch.Tensor:
    h, body = serialize.read(buf)
    return torch.frombuffer(bytearray(body[: h.n_groups]), dtype=torch.uint8).clone()


@pytest.mark.parametrize("w", BIT_WIDTHS)
def test_structural_decisions_agree(w, gen):
    """Same widths, same raw groups, same payload size — no ulp excuse applies here."""
    x = hidden_state(tokens=64, gen=gen)
    (cbuf, cst), (gbuf, gst) = _pair(x, _cfg(w))
    assert torch.equal(_widths(cbuf), _widths(gbuf))
    assert gst.compressed_bytes == cst.compressed_bytes
    assert gst.total_blocks == cst.total_blocks
    assert gst.nonfinite_blocks == cst.nonfinite_blocks


@pytest.mark.parametrize("w", BIT_WIDTHS)
def test_decoders_are_interchangeable(w, gen):
    """Either decoder reads either encoder's payload, and both land inside eps.

    This is the property a codec split across two devices actually needs.
    """
    x = hidden_state(tokens=64, gen=gen)
    (cbuf, _), (gbuf, _) = _pair(x, _cfg(w))
    for buf in (cbuf, gbuf):
        out = reference.decode(buf)
        assert out.shape == x.shape and out.dtype == x.dtype
        assert reference.max_abs_error(x, out) <= EPS + FP_TOLERANCE


@pytest.mark.parametrize("w", BIT_WIDTHS)
def test_codes_differ_by_at_most_one_step(w, gen):
    """Quantify the divergence instead of tolerating it silently.

    A regression that broke the kernel would move codes by far more than one
    step, so this stays a real test rather than a rubber stamp.
    """
    x = hidden_state(tokens=64, gen=gen)
    (cbuf, _), (gbuf, _) = _pair(x, _cfg(w))
    c, g = reference.decode(cbuf).float(), reference.decode(gbuf).float()
    step = 2 * EPS                      # one quantisation step is at most 2*eps
    assert float((c - g).abs().max()) <= step + FP_TOLERANCE
    assert float((c != g).float().mean()) < 0.05, "far more codes moved than rounding explains"


@pytest.mark.parametrize("w", BIT_WIDTHS)
def test_gpu_respects_the_error_bound(w, gen):
    """The contract holds on the GPU path in its own right, not by inheritance."""
    x = hidden_state(tokens=64, gen=gen)
    buf, _ = gpu.encode(x.cuda().contiguous(), _cfg(w))
    assert reference.max_abs_error(x, reference.decode(buf)) <= EPS + FP_TOLERANCE


def test_gpu_handles_channel_outliers(gen):
    """The distribution the codec actually meets, not a clean Gaussian."""
    x = with_channel_outliers(hidden_state(tokens=64, gen=gen))
    (cbuf, cst), (gbuf, gst) = _pair(x, _cfg(8))
    assert torch.equal(_widths(cbuf), _widths(gbuf))
    assert gst.compressed_bytes == cst.compressed_bytes
    assert reference.max_abs_error(x, reference.decode(gbuf)) <= EPS + FP_TOLERANCE


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_gpu_routes_nonfinite_groups_to_bit_exact(bad, gen):
    """A non-finite group travels bit-exact on the GPU path too, and is counted."""
    x = hidden_state(tokens=64, gen=gen)
    x[0, 0] = bad
    (cbuf, cst), (gbuf, gst) = _pair(x, _cfg(8))
    assert gst.nonfinite_blocks == cst.nonfinite_blocks >= 1
    out = reference.decode(gbuf)
    assert torch.equal(out[0, :256].view(torch.int16), x[0, :256].view(torch.int16))


def test_gpu_refuses_out_of_scope_configs():
    """The narrowed scope is enforced, not merely documented."""
    x = torch.randn(64, 2048, device="cuda").to(torch.bfloat16)
    with pytest.raises(NotImplementedError):
        gpu.encode(x, CodecConfig(eps=EPS, allocation=Allocation.PER_CHANNEL, bit_width=8))
    with pytest.raises(NotImplementedError):
        gpu.encode(x, CodecConfig(eps=EPS, allocation=Allocation.BLOCKWISE))
