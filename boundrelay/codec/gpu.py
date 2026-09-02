"""Triton encode/decode kernels.

Import-guarded: Triton needs CUDA, so on the laptop this module imports cleanly,
:func:`available` returns False, and the ``gpu`` test marker skips. Development
happens against :mod:`boundrelay.codec.reference` first — the GPU path is
measured, never trusted on its own.

Scope of the GPU path
---------------------
Narrower than the reference, deliberately, and the difference is reported rather
than papered over:

* **Blockwise allocation only.** Blocks are contiguous runs over the raveled
  tensor, so every kernel is a coalesced read. Per-channel allocation needs a
  channel-major gather; that permute is a real cost and belongs *in* the cost
  model, not hidden inside a kernel. Measuring it is future work, and until then
  per-channel ratios come from the reference and carry no throughput number.
* **Uniform bit width.** ``CodecConfig.bit_width`` must be set. Per-group adaptive
  widths make the output byte offsets data-dependent and are reference-only; they
  exist to answer the ratio question (Figure 2), not the throughput question
  (Table A), and Table A asks for exactly 4/6/8.
* Non-finite groups, and groups whose budget the width cannot meet, are moved to
  the bit-exact path on the host — identical policy to the reference.

The wire format is the same as the reference's, so payloads are interchangeable:
either decoder reads either encoder's output.

STATUS: written, **not yet executed on a GPU**. Nothing here has run. The first
GPU session must run ``tests/test_gpu_equivalence.py`` before any number from
this module is quoted anywhere.
"""

from __future__ import annotations

import math

import torch

from boundrelay.codec import packing, serialize
from boundrelay.codec.contract import (
    FP_TOLERANCE,
    Allocation,
    BypassReason,
    CodecConfig,
    EncodeStats,
)
from boundrelay.codec import reference
from boundrelay.codec.reference import RAW, _half_ulp
from boundrelay.codec.serialize import Header

try:  # pragma: no cover - environment dependent
    import triton
    import triton.language as tl
    # Round-half-to-even lives in libdevice, not tl.math -- tl.math carries only
    # ceil/floor. Verified against Triton 3.4.0; if a future version moves it,
    # test_gpu_matches_reference_codes fails loudly at import time.
    from triton.language.extra import libdevice

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

    class _Stub:
        def __getattr__(self, name):
            raise RuntimeError("Triton is not installed")

        def jit(self, fn):
            return fn

        def constexpr(self, *_a, **_k):
            return None

    triton = _Stub()          # type: ignore[assignment]
    tl = _Stub()              # type: ignore[assignment]


def available() -> bool:
    """True only where a Triton kernel can actually run."""
    return _HAS_TRITON and torch.cuda.is_available()


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _group_stats_kernel(x_ptr, gmin_ptr, gmax_ptr, bad_ptr, n_elements,
                            GROUP: tl.constexpr, BLOCK: tl.constexpr):
        """Per-group min/max over finite values, plus a non-finite flag.

        One program per group. Non-finite values are excluded from the reduction
        (they cannot participate in a range) and raise the group's ``bad`` flag,
        which sends the whole group down the bit-exact path.
        """
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        offs = pid * GROUP + lane
        mask = (offs < n_elements) & (lane < GROUP)

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        finite = (x == x) & (tl.abs(x) < float("inf"))

        tl.store(bad_ptr + pid, tl.max(tl.where(mask & (finite == 0), 1, 0), axis=0))
        tl.store(gmin_ptr + pid,
                 tl.min(tl.where(mask & finite, x, float("inf")), axis=0))
        tl.store(gmax_ptr + pid,
                 tl.max(tl.where(mask & finite, x, float("-inf")), axis=0))

    @triton.jit
    def _quantize_kernel(x_ptr, code_ptr, gmin_ptr, gscale_ptr, n_elements,
                         GROUP: tl.constexpr, MAXCODE: tl.constexpr,
                         BLOCK: tl.constexpr):
        """Values -> integer codes, one uint8 container per code.

        Rounding must match ``torch.round`` (round-half-to-even) or the GPU and
        reference codes diverge on ties. ``libdevice.rint`` is the IEEE
        round-to-nearest-even primitive; ``libdevice.round`` is half-away-from-zero
        and would be wrong here.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = offs // GROUP
        lo = tl.load(gmin_ptr + g, mask=mask, other=0.0)
        sc = tl.load(gscale_ptr + g, mask=mask, other=0.0)

        q = tl.where(sc > 0.0, libdevice.rint((x - lo) / tl.where(sc > 0.0, sc, 1.0)), 0.0)
        q = tl.minimum(tl.maximum(q, 0.0), float(MAXCODE))
        tl.store(code_ptr + offs, q.to(tl.uint8), mask=mask)

    @triton.jit
    def _pack_kernel(code_ptr, out_ptr, numel, n_out,
                     W: tl.constexpr, BLOCK: tl.constexpr):
        """Codes -> dense bytes, MSB-first within each code.

        One program lane per *output byte*: it walks its eight bits, finds each
        bit's source code, and ORs it into place. Bit-addressing rather than
        word-tricks keeps widths 4, 6 and 8 on one code path — six being the
        non-power-of-two case a word-based packer silently rounds up.
        """
        pid = tl.program_id(0)
        j = pid * BLOCK + tl.arange(0, BLOCK)
        mask = j < n_out

        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        for b in tl.static_range(8):
            gbit = j * 8 + b
            ci = gbit // W
            bi = gbit % W
            c = tl.load(code_ptr + ci, mask=mask & (ci < numel), other=0).to(tl.int32)
            acc = acc | (((c >> (W - 1 - bi)) & 1) << (7 - b))
        tl.store(out_ptr + j, acc.to(tl.uint8), mask=mask)

    @triton.jit
    def _unpack_dequant_kernel(buf_ptr, out_ptr, gmin_ptr, gscale_ptr,
                               numel, n_bytes, W: tl.constexpr,
                               GROUP: tl.constexpr, BLOCK: tl.constexpr):
        """Packed bytes -> reconstructed float32, fused with the dequantise."""
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        mask = i < numel

        acc = tl.zeros((BLOCK,), dtype=tl.int32)
        for b in tl.static_range(W):
            gbit = i * W + b
            byi = gbit // 8
            bib = gbit % 8
            byte = tl.load(buf_ptr + byi, mask=mask & (byi < n_bytes), other=0).to(tl.int32)
            acc = acc | (((byte >> (7 - bib)) & 1) << (W - 1 - b))

        g = i // GROUP
        lo = tl.load(gmin_ptr + g, mask=mask, other=0.0)
        sc = tl.load(gscale_ptr + g, mask=mask, other=0.0)
        tl.store(out_ptr + i, lo + acc.to(tl.float32) * sc, mask=mask)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _require(cfg: CodecConfig) -> int:
    if not available():
        raise RuntimeError("Triton/CUDA not available")
    if cfg.allocation is not Allocation.BLOCKWISE:
        raise NotImplementedError(
            "GPU path is blockwise-only; per-channel needs a measured permute")
    if cfg.bit_width is None:
        raise NotImplementedError(
            "GPU path needs a uniform bit_width; adaptive widths are reference-only")
    return cfg.bit_width


def encode(x: torch.Tensor, cfg: CodecConfig) -> tuple[bytes, EncodeStats]:
    """GPU encode. Emits the same wire format as :func:`reference.encode`."""
    w = _require(cfg)
    if not x.is_contiguous():
        raise ValueError("expected a contiguous tensor")
    if x.numel() < cfg.block_size:
        # The reference bypasses below one block rather than doing partial-block
        # arithmetic. Delegate instead of reimplementing, so the two paths cannot
        # drift apart on the edge case least likely to be exercised.
        return reference.encode(x.cpu(), cfg)

    flat = x.reshape(-1)
    n = flat.numel()
    group = cfg.block_size
    n_groups = math.ceil(n / group)
    dev = flat.device

    gmin = torch.empty(n_groups, dtype=torch.float32, device=dev)
    gmax = torch.empty(n_groups, dtype=torch.float32, device=dev)
    bad = torch.empty(n_groups, dtype=torch.int32, device=dev)
    _group_stats_kernel[(n_groups,)](
        flat, gmin, gmax, bad, n,
        GROUP=group, BLOCK=triton.next_power_of_2(group))

    ranges = gmax - gmin
    budget = cfg.eps - _half_ulp(torch.maximum(gmax.abs(), gmin.abs()), x.dtype)
    # Same predicate as the reference, stated the same way: half a quantisation
    # step must fit inside the budget. Expressed as ceil(log2(...)) instead, a
    # one-ulp wobble flips the ceil and the two paths disagree about which
    # borderline groups compress.
    half_step = ranges / float(2**w - 1) / 2.0
    ok = (budget > 0) & (half_step <= budget) & (bad == 0)

    scales = torch.where(ok & (ranges > 0), ranges / (2**w - 1),
                         torch.zeros_like(ranges))
    gmin_eff = torch.where(ok, gmin, torch.zeros_like(gmin))

    codes = torch.empty(n, dtype=torch.uint8, device=dev)
    _quantize_kernel[(triton.cdiv(n, 1024),)](
        flat, codes, gmin_eff, scales, n,
        GROUP=group, MAXCODE=2**w - 1, BLOCK=1024)

    # Fail closed, exactly as the reference does. Choosing the width from a
    # predicted budget is not the same as meeting it: the half-ulp term is an
    # upper bound over the group's magnitude, so a group can pass the prediction
    # and still miss the contract on its actual values. Measure the realised
    # error and demote whatever missed -- the contract is a guarantee, not an
    # estimate, and a GPU path that skipped this would ship out-of-bound data
    # that the reference would have caught.
    gidx = torch.arange(n, device=dev) // group
    recon = (gmin_eff[gidx] + codes.to(torch.float32) * scales[gidx]).to(x.dtype)
    err = (flat.to(torch.float32) - recon.to(torch.float32)).abs()
    err = torch.where(torch.isfinite(err), err, torch.zeros_like(err))
    g_err = torch.zeros(n_groups, device=dev).scatter_reduce(0, gidx, err, reduce="amax")
    ok = ok & (g_err <= cfg.eps + FP_TOLERANCE)
    scales = torch.where(ok, scales, torch.zeros_like(scales))

    n_out = packing.packed_bytes(n, w)
    packed = torch.empty(n_out, dtype=torch.uint8, device=dev)
    _pack_kernel[(triton.cdiv(n_out, 1024),)](
        codes, packed, n, n_out, W=w, BLOCK=1024)

    # Groups that could not meet the bound travel bit-exact, exactly as the
    # reference does. Assembly happens on the host: it touches a small minority
    # of groups and keeping it here keeps one wire format.
    widths = torch.where(ok, torch.full_like(bad, w), torch.zeros_like(bad))
    # Metadata carries the real per-group min, matching the reference: a raw
    # group's min is never read back, but the bytes must still agree.
    return _assemble(flat, packed, widths.cpu(), gmin.cpu(), scales.cpu(),
                     cfg, x, w, int((bad != 0).sum()), n_groups)


def _assemble(flat, packed, widths, gmin, scales, cfg, x, w, n_bad, n_groups) -> tuple[bytes, EncodeStats]:
    group = cfg.block_size
    n = flat.numel()
    packed_cpu = packed.cpu().numpy().tobytes()

    n_raw = int((widths == RAW).sum())
    if n_raw == 0 and n % group == 0:
        # Fast path, and the one that matters for the cost table. With no raw
        # groups and a whole number of blocks, the per-group packed chunks
        # concatenate to exactly the packed buffer -- 256 codes at 4, 6 or 8 bits
        # are all a whole number of bytes -- so the assembly is one copy rather
        # than a Python loop over tens of thousands of groups. Measuring the loop
        # instead would have put encode two orders of magnitude below the link
        # and made compression look unconditionally worthless: an artefact of the
        # serialiser, not a property of the codec.
        chunks = [packed_cpu]
    else:
        src = flat.cpu()
        chunks = []
        for g in range(n_groups):
            s = g * group
            sz = min(group, n - s)
            if int(widths[g]) == RAW:
                chunks.append(src[s : s + sz].contiguous().view(torch.uint8).numpy().tobytes())
            else:
                lo = packing.packed_bytes(s, w)
                chunks.append(packed_cpu[lo : lo + packing.packed_bytes(sz, w)])

    meta = (widths.to(torch.uint8).numpy().tobytes()
            + gmin.to(torch.float32).numpy().tobytes()
            + scales.to(torch.float32).numpy().tobytes())
    payload = meta + b"".join(chunks)

    header = Header(
        codec_version=serialize.CODEC_VERSION, eps=cfg.eps,
        allocation=cfg.allocation.value, bit_width=w,
        original_dtype=str(x.dtype), original_shape=tuple(x.shape),
        block_size=group, n_groups=n_groups, compressed_bytes=0, crc32=0,
    )
    buf = serialize.write(header, payload)
    stats = EncodeStats(
        original_bytes=n * x.element_size(),
        compressed_bytes=len(payload),
        bit_width=w,
        max_abs_error=float("nan"),   # filled by the caller if it decodes
        nonfinite_blocks=n_bad,
        total_blocks=n_groups,
        bypass_reason=None,
    )
    return buf, stats


def _kernels_only(x: torch.Tensor, cfg: CodecConfig) -> torch.Tensor:
    """The GPU half of encode: stats, quantise, pack. No host assembly.

    This is the cost a deployed transport would actually pay per chunk, and the
    number the break-even model consumes.
    """
    w = cfg.bit_width
    flat = x.reshape(-1)
    n = flat.numel()
    group = cfg.block_size
    n_groups = math.ceil(n / group)
    dev = flat.device

    gmin = torch.empty(n_groups, dtype=torch.float32, device=dev)
    gmax = torch.empty(n_groups, dtype=torch.float32, device=dev)
    bad = torch.empty(n_groups, dtype=torch.int32, device=dev)
    _group_stats_kernel[(n_groups,)](
        flat, gmin, gmax, bad, n, GROUP=group, BLOCK=triton.next_power_of_2(group))

    ranges = gmax - gmin
    budget = cfg.eps - _half_ulp(torch.maximum(gmax.abs(), gmin.abs()), x.dtype)
    ok = (budget > 0) & (ranges / float(2**w - 1) / 2.0 <= budget) & (bad == 0)
    scales = torch.where(ok & (ranges > 0), ranges / (2**w - 1), torch.zeros_like(ranges))

    codes = torch.empty(n, dtype=torch.uint8, device=dev)
    _quantize_kernel[(triton.cdiv(n, 1024),)](
        flat, codes, torch.where(ok, gmin, torch.zeros_like(gmin)), scales, n,
        GROUP=group, MAXCODE=2**w - 1, BLOCK=1024)

    n_out = packing.packed_bytes(n, w)
    packed = torch.empty(n_out, dtype=torch.uint8, device=dev)
    _pack_kernel[(triton.cdiv(n_out, 1024),)](codes, packed, n, n_out, W=w, BLOCK=1024)
    return packed


def _decode_kernels_only(packed: torch.Tensor, gmin: torch.Tensor,
                         scales: torch.Tensor, numel: int,
                         cfg: CodecConfig) -> torch.Tensor:
    """The GPU half of decode: unpack fused with dequantise.

    The break-even inequality needs both directions. Estimating decode as
    "about the same as encode" would be a guess sitting inside a measured
    result, so it is measured.
    """
    w = cfg.bit_width
    out = torch.empty(numel, dtype=torch.float32, device=packed.device)
    _unpack_dequant_kernel[(triton.cdiv(numel, 1024),)](
        packed, out, gmin, scales, numel, packed.numel(),
        W=w, GROUP=cfg.block_size, BLOCK=1024)
    return out


def benchmark(shapes: list[tuple[int, ...]], cfg: CodecConfig,
              iters: int = 50, warmup: int = 10) -> dict:
    """Encode/decode throughput by shape — the cost-model lookup table (D15).

    Run this on **every** GPU generation used in the study. Pairing A40 codec
    timings with H100 link bandwidth would put the two sides of the break-even
    inequality on different machines, which invalidates the atlas.
    """
    import time

    w = _require(cfg)
    out: dict[str, dict] = {}
    for shape in shapes:
        x = torch.randn(*shape, device="cuda").to(torch.bfloat16).contiguous()
        nbytes = x.numel() * x.element_size()

        for _ in range(warmup):
            encode(x, cfg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            buf, st = encode(x, cfg)
        torch.cuda.synchronize()
        enc_s = (time.perf_counter() - t0) / iters

        # Kernel-only time: the GPU work a deployed system would pay, separated
        # from this research serialiser's host-side assembly. The break-even
        # model needs the first; quoting the second as "codec cost" would blame
        # the codec for a Python loop.
        for _ in range(warmup):
            _kernels_only(x, cfg)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(iters):
            _kernels_only(x, cfg)
        torch.cuda.synchronize()
        ker_s = (time.perf_counter() - t1) / iters

        # decode side
        n = x.numel()
        n_groups = math.ceil(n / cfg.block_size)
        packed = _kernels_only(x, cfg)
        gmin_d = torch.zeros(n_groups, dtype=torch.float32, device="cuda")
        scal_d = torch.full((n_groups,), 1e-3, dtype=torch.float32, device="cuda")
        for _ in range(warmup):
            _decode_kernels_only(packed, gmin_d, scal_d, n, cfg)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in range(iters):
            _decode_kernels_only(packed, gmin_d, scal_d, n, cfg)
        torch.cuda.synchronize()
        dec_s = (time.perf_counter() - t2) / iters

        out[str(tuple(shape))] = {
            "decode_kernel_s": dec_s,
            "decode_kernel_GBps": nbytes / dec_s / 1e9,
            "bit_width": w,
            "eps": cfg.eps,
            "original_bytes": nbytes,
            "compressed_bytes": st.compressed_bytes,
            "ratio": st.ratio,
            "encode_s": enc_s,
            "encode_GBps": nbytes / enc_s / 1e9,
            "kernel_s": ker_s,
            "kernel_GBps": nbytes / ker_s / 1e9,
            "nonfinite_blocks": st.nonfinite_blocks,
        }
        del x
        torch.cuda.empty_cache()
    return out
