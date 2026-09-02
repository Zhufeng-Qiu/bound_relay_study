"""PyTorch reference codec -- the correctness oracle for the GPU kernel.

Runs on CPU/MPS, so the entire error-bound contract is settled on the laptop
before a single GPU-hour is spent (Runbook D2-D3, Gate A).

Method
------
Uniform quantisation within each budget group. For a group with range
``R = max - min`` and ``b`` bits the step is ``s = R / (2**b - 1)``, and nearest
rounding bounds the error by ``s / 2``. The smallest width meeting a target eps
is therefore ``b = ceil(log2(R / (2 * eps) + 1))``.

Group metadata (``min``, ``scale``) is stored in **fp32**. In fp16 the metadata's
own representation error can exceed the bound the codec advertises.

The output dtype is part of the error budget
--------------------------------------------
Reconstruction is computed in fp32 and cast back to the input dtype. That cast
rounds, and for bf16 -- 8 mantissa bits -- the half-ulp at magnitude ``M`` is
about ``M * 2**-8``. Near ``|x| ~ 3`` that is ~0.008, which is a large share of a
tight eps and the whole of a very tight one. So the budget is split explicitly::

    effective_eps = eps - half_ulp(max |x| in group)

and a group whose effective budget is non-positive, or which still needs more
than 8 bits, is stored **raw** (bit-exact) rather than quietly violating the
bound. This is why very small eps values do not compress bf16 activations at all:
not a limitation to work around, but a real property of the representation, and
one worth stating in the note.

Every group is bound-checked after reconstruction regardless. A group that fails
falls back to raw -- fail closed, always.
"""

from __future__ import annotations

import math

import torch

from boundrelay.codec import packing, serialize
from boundrelay.codec.allocation import group_bounds, partition
from boundrelay.codec.contract import (
    FP_TOLERANCE,
    Allocation,
    BypassReason,
    CodecConfig,
    EncodeStats,
)
from boundrelay.codec.serialize import Header

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

#: width 0 in the wire format means "this group travels bit-exact"
RAW = 0


def _half_ulp(magnitude: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Half the spacing of ``dtype`` at ``magnitude``. Zero for fp32 inputs."""
    if dtype is torch.float32:
        return torch.zeros_like(magnitude)
    mantissa_bits = 7 if dtype is torch.bfloat16 else 10  # bf16 / fp16
    exp = torch.floor(torch.log2(magnitude.clamp(min=1e-30)))
    return torch.ldexp(torch.ones_like(magnitude), (exp - mantissa_bits - 1).to(torch.int32))


def choose_bit_width(block_range: torch.Tensor, eps: float | torch.Tensor) -> torch.Tensor:
    """Smallest supported width whose quantisation step satisfies ``eps``.

    Returns :data:`RAW` (0) where no supported width suffices, including where
    ``eps`` is non-positive after the output-dtype budget has been subtracted.
    """
    r = block_range.float().clamp(min=0.0)
    eps_t = torch.as_tensor(eps, dtype=torch.float32)
    if eps_t.dim() == 0:
        eps_t = eps_t.expand_as(r)

    need = torch.ceil(torch.log2(r / (2.0 * eps_t.clamp(min=1e-30)) + 1.0))
    need = torch.nan_to_num(need, nan=99.0, posinf=99.0, neginf=1.0).clamp(min=1.0)
    need = torch.where(eps_t > 0, need, torch.full_like(need, 99.0))

    out = torch.zeros_like(need, dtype=torch.int64)
    for w in reversed(packing.SUPPORTED_WIDTHS):  # 8, 6, 4 -- smallest wins
        out = torch.where(need <= w, torch.full_like(out, w), out)
    return out


def _bypass(x: torch.Tensor, cfg: CodecConfig, reason: BypassReason
            ) -> tuple[bytes, EncodeStats]:
    """Emit an uncompressed payload. The only correct answer when in doubt."""
    header = Header(
        codec_version=serialize.CODEC_VERSION, eps=cfg.eps,
        allocation=cfg.allocation.value, bit_width=-1,
        original_dtype=str(x.dtype), original_shape=tuple(x.shape),
        block_size=cfg.block_size, n_groups=0, compressed_bytes=0, crc32=0,
    )
    payload = x.contiguous().view(torch.uint8).numpy().tobytes()
    buf = serialize.write(header, payload)
    stats = EncodeStats(
        original_bytes=x.numel() * x.element_size(),
        compressed_bytes=len(payload),
        bit_width=-1, max_abs_error=0.0,
        nonfinite_blocks=0, total_blocks=0, bypass_reason=reason,
    )
    return buf, stats


def encode(x: torch.Tensor, cfg: CodecConfig) -> tuple[bytes, EncodeStats]:
    """Compress ``x`` under ``cfg``. Fails closed on anything unexpected."""
    if x.dtype not in _SUPPORTED_DTYPES:
        return _bypass(x, cfg, BypassReason.UNSUPPORTED_DTYPE)
    if not x.is_contiguous():
        return _bypass(x, cfg, BypassReason.NON_CONTIGUOUS)
    if x.numel() < cfg.block_size:
        return _bypass(x, cfg, BypassReason.TENSOR_TOO_SMALL)

    flat = x.reshape(-1)
    groups = partition(x, cfg.allocation, cfg.block_size)
    n_groups = int(groups.max()) + 1

    # Sorting by group id makes every group a contiguous run, which keeps one
    # code path for blockwise, per-channel and per-token alike.
    order = torch.argsort(groups, stable=True)
    vals = flat[order]
    sizes = torch.bincount(groups, minlength=n_groups)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), sizes.cumsum(0)[:-1]])

    finite = torch.isfinite(vals.float())
    g_sorted = groups[order]
    bad = torch.zeros(n_groups, dtype=torch.bool).scatter_reduce(
        0, g_sorted, ~finite, reduce="amax")

    safe = torch.where(finite, vals.float(), torch.zeros_like(vals.float()))
    gmin, gmax = group_bounds(safe, g_sorted, n_groups)
    ranges = gmax - gmin

    # The output dtype's own rounding eats part of the budget, so the width is
    # chosen against what is actually left, not against the nominal eps.
    budget = cfg.eps - _half_ulp(torch.maximum(gmax.abs(), gmin.abs()), x.dtype)
    if cfg.bit_width is None:
        widths = choose_bit_width(ranges, budget)
    else:
        # Ask the question directly: at this width, is half the quantisation step
        # inside the budget? The equivalent `ceil(log2(...))` form chains a log
        # and a ceil, and a one-ulp wobble at the input flips the ceil -- so two
        # implementations of the *same* predicate classify borderline groups
        # differently. One division and one comparison is both clearer and
        # stable enough that the CPU and GPU paths agree.
        w_ = cfg.bit_width
        half_step = ranges / float(2**w_ - 1) / 2.0
        widths = torch.where((budget > 0) & (half_step <= budget),
                             torch.full_like(ranges, w_, dtype=torch.int64),
                             torch.zeros_like(ranges, dtype=torch.int64))
    widths = torch.where(bad, torch.zeros_like(widths), widths)

    # Compute every scale in the dtype it is *stored* in.
    #
    # Doing `float(range) / (2**w - 1)` in Python performs the division in
    # float64 and only then rounds to float32; a GPU kernel divides in float32
    # throughout. The two differ by up to one float32 ulp, which is enough to
    # flip a value sitting exactly on a rounding boundary and change one code --
    # so the CPU oracle and the GPU kernel would disagree byte-for-byte while
    # both stayed inside the error bound. Vectorising here in float32 keeps them
    # identical.
    denom = torch.pow(2.0, widths.to(torch.float32)) - 1.0
    scales_all = torch.where(
        (widths > 0) & (ranges > 0) & (denom > 0),
        ranges / torch.where(denom > 0, denom, torch.ones_like(denom)),
        torch.zeros_like(ranges),
    ).to(torch.float32)

    chunks: list[bytes] = []
    recon = vals.clone()
    scales = torch.zeros(n_groups, dtype=torch.float32)
    elem = x.element_size()

    for g in range(n_groups):
        s, n = int(starts[g]), int(sizes[g])
        seg = vals[s : s + n]
        w = int(widths[g])
        if w == RAW:
            chunks.append(seg.contiguous().view(torch.uint8).numpy().tobytes())
            continue
        lo = float(gmin[g])
        scale = float(scales_all[g])
        scales[g] = scale
        if scale == 0.0:
            codes = torch.zeros(n, dtype=torch.int32)
        else:
            codes = torch.round((seg.float() - lo) / scale).clamp(0, 2**w - 1).to(torch.int32)
        chunks.append(packing.pack(codes, w).numpy().tobytes())
        recon[s : s + n] = (torch.as_tensor(lo) + codes.float() * scale).to(x.dtype)

    # Fail closed: any group that missed the bound is re-emitted bit-exact.
    err = (vals.float() - recon.float()).abs()
    err = torch.where(torch.isfinite(err), err, torch.zeros_like(err))
    g_err = torch.zeros(n_groups).scatter_reduce(0, g_sorted, err, reduce="amax")
    failed = (g_err > cfg.eps + FP_TOLERANCE) & (widths > 0)
    for g in failed.nonzero().flatten().tolist():
        s, n = int(starts[g]), int(sizes[g])
        chunks[g] = vals[s : s + n].contiguous().view(torch.uint8).numpy().tobytes()
        recon[s : s + n] = vals[s : s + n]
        widths[g] = RAW
        scales[g] = 0.0

    meta = (
        widths.to(torch.uint8).numpy().tobytes()
        + gmin.to(torch.float32).numpy().tobytes()
        + scales.numpy().tobytes()
    )
    payload = meta + b"".join(chunks)

    uniq = sorted({int(w) for w in widths.tolist() if w > 0})
    header = Header(
        codec_version=serialize.CODEC_VERSION, eps=cfg.eps,
        allocation=cfg.allocation.value,
        bit_width=uniq[0] if len(uniq) == 1 else -1,
        original_dtype=str(x.dtype), original_shape=tuple(x.shape),
        block_size=cfg.block_size, n_groups=n_groups,
        compressed_bytes=0, crc32=0,
    )
    buf = serialize.write(header, payload)

    out = torch.empty_like(flat)  # undo the sort for the error report
    out[order] = recon
    stats = EncodeStats(
        original_bytes=x.numel() * elem,
        compressed_bytes=len(payload),
        bit_width=uniq[0] if len(uniq) == 1 else -1,
        max_abs_error=max_abs_error(flat, out),
        nonfinite_blocks=int(bad.sum()),
        total_blocks=n_groups,
        bypass_reason=None,
    )
    return buf, stats


def decode(payload: bytes) -> torch.Tensor:
    """Reconstruct a tensor from :func:`encode`'s payload. Deterministic."""
    header, body = serialize.read(payload)
    dtype = getattr(torch, header.original_dtype.split(".")[-1])
    shape = tuple(header.original_shape)

    if header.n_groups == 0:  # bypass payload
        return torch.frombuffer(bytearray(body), dtype=dtype).reshape(shape).clone()

    n = header.n_groups
    off = 0
    widths = torch.frombuffer(bytearray(body[off : off + n]), dtype=torch.uint8).to(torch.int64)
    off += n
    gmin = torch.frombuffer(bytearray(body[off : off + 4 * n]), dtype=torch.float32)
    off += 4 * n
    scales = torch.frombuffer(bytearray(body[off : off + 4 * n]), dtype=torch.float32)
    off += 4 * n

    numel = 1
    for d in shape:
        numel *= d
    proto = torch.empty(0, dtype=dtype)
    groups = partition(torch.empty(shape, dtype=dtype), Allocation(header.allocation),
                       header.block_size)
    order = torch.argsort(groups, stable=True)
    sizes = torch.bincount(groups, minlength=n)
    elem = proto.element_size()

    out_sorted = torch.empty(numel, dtype=dtype)
    pos = 0
    for g in range(n):
        sz = int(sizes[g])
        w = int(widths[g])
        if w == RAW:
            nb = sz * elem
            seg = torch.frombuffer(bytearray(body[off : off + nb]), dtype=torch.uint8)
            out_sorted[pos : pos + sz] = seg.view(dtype)
            off += nb
        else:
            nb = packing.packed_bytes(sz, w)
            buf = torch.frombuffer(bytearray(body[off : off + nb]), dtype=torch.uint8)
            codes = packing.unpack(buf, w, sz)
            out_sorted[pos : pos + sz] = (gmin[g] + codes.float() * scales[g]).to(dtype)
            off += nb
        pos += sz

    out = torch.empty(numel, dtype=dtype)
    out[order] = out_sorted
    return out.reshape(shape)


def max_abs_error(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """``max |x - x_hat|`` over **finite** positions only.

    Non-finite positions are excluded by construction: they travel bit-exact, and
    ``|nan - nan|`` is not a quantity the contract can bound.
    """
    xf, hf = x.reshape(-1).float(), x_hat.reshape(-1).float()
    finite = torch.isfinite(xf)
    if not bool(finite.any()):
        return 0.0
    return float((xf[finite] - hf[finite]).abs().max())
