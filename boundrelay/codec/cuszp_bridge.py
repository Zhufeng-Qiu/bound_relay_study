"""ctypes bridge to cuSZp, so ratios can be measured inline while a tensor is
still in hand rather than by persisting the whole corpus and reading it back.

Only the 1-D f32 entry points are bound. cuSZp reads fp32; KV is bf16, so every
call pays a lossless upcast, and that upcast is timed separately rather than
folded in — it is a property of the interface, not of the codec.
"""
from __future__ import annotations

import ctypes
import torch

_LIB = None
MODES = ("plain", "outlier", "fixed")

#: Absolute tolerance for every numerical acceptance check in this project.
#: Declared here, once, before any result was seen. It is not a knob to widen when
#: a check fails -- a failure at this tolerance is a finding, not a calibration.
TAU = 1e-6


#: cuSZp's headers declare no ``extern "C"``, so the shared object exports
#: C++-mangled names. Binding by the mangled symbol is more brittle than a C
#: shim would be, but it avoids maintaining a patched fork of the library, and a
#: rename shows up immediately as an AttributeError at import rather than as a
#: silently wrong number.
_MANGLED = {
    ("compress", "fixed"):     "_Z27cuSZp_compress_1D_fixed_f32PfPhmPmfP11CUstream_st",
    ("compress", "plain"):     "_Z27cuSZp_compress_1D_plain_f32PfPhmPmfP11CUstream_st",
    ("compress", "outlier"):   "_Z29cuSZp_compress_1D_outlier_f32PfPhmPmfP11CUstream_st",
    ("decompress", "fixed"):   "_Z29cuSZp_decompress_1D_fixed_f32PfPhmmfP11CUstream_st",
    ("decompress", "plain"):   "_Z29cuSZp_decompress_1D_plain_f32PfPhmmfP11CUstream_st",
    ("decompress", "outlier"): "_Z31cuSZp_decompress_1D_outlier_f32PfPhmmfP11CUstream_st",
}


def lib(path: str = "/workspace/cuSZp/build/libcuSZp.so"):
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(path)
        for m in MODES:
            for op in ("compress", "decompress"):
                fn = getattr(_LIB, _MANGLED[(op, m)])
                fn.restype = None
                fn.argtypes = ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_size_t), ctypes.c_float,
                                ctypes.c_void_p] if op == "compress" else
                               [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p])
    return _LIB


def roundtrip(x_bf16: torch.Tensor, eps: float, mode: str = "plain",
              want_decode: bool = True, probe_writes: bool = False) -> dict:
    """Compress and (optionally) reconstruct, returning bytes and achieved error.

    The error reported is against the **final bf16** value, not the fp32
    intermediate: that is what a receiver actually gets, and the downcast adds up
    to half a bf16 ulp on top of whatever the codec guaranteed in fp32.
    """
    L = lib()
    d32 = x_bf16.cuda().float().contiguous().reshape(-1)
    n = d32.numel()
    # zeroed, like the decode destination: cuSZp reads past cmpSize and leftovers
    # there make the decoder silently return an all-zero reconstruction
    cmp_buf = torch.zeros(n * 4 + 4096, dtype=torch.uint8, device="cuda")
    size = ctypes.c_size_t(0)

    getattr(L, _MANGLED[("compress", mode)])(
        ctypes.c_void_p(d32.data_ptr()), ctypes.c_void_p(cmp_buf.data_ptr()),
        ctypes.c_size_t(n), ctypes.byref(size), ctypes.c_float(eps), None)
    torch.cuda.synchronize()
    out = {"cmp_bytes": int(size.value), "n": int(n),
           "ratio_vs_bf16": (n * 2) / max(int(size.value), 1)}

    if want_decode:
        # Poison the destination before decoding. PyTorch's caching allocator hands
        # back the same block across repeated calls, so a decompress that silently
        # writes nothing leaves the *previous* call's reconstruction in place — and
        # the error then measured belongs to that earlier call, not this one. That
        # is exactly what happened in the first corpus run: three modes produced
        # different compressed bytes and byte-identical reconstruction errors, which
        # is impossible unless the errors were stale.
        # cuSZp's decompress does not write elements it expects to be zero: 128 to
        # 2912 per tensor here, rising with the error bound. The destination must
        # therefore arrive **zeroed** -- an API contract its headers do not state.
        # Filling with anything else leaves the allocator's previous contents in
        # those positions, which is what made the first corpus run appear to show
        # cuSZp missing its bound on a third of tensors. It was not; measured
        # against a zeroed buffer the error is 0.9984-0.9999 of eps throughout.
        dec = torch.zeros(n, dtype=torch.float32, device="cuda")
        getattr(L, _MANGLED[("decompress", mode)])(
            ctypes.c_void_p(dec.data_ptr()), ctypes.c_void_p(cmp_buf.data_ptr()),
            ctypes.c_size_t(n), ctypes.c_size_t(int(size.value)),
            ctypes.c_float(eps), None)
        torch.cuda.synchronize()

        if probe_writes:
            # Same decode into a poisoned buffer, purely to report how much the
            # decoder left untouched. Diagnostic, never the reconstruction.
            probe = torch.full((n,), float("nan"), dtype=torch.float32, device="cuda")
            getattr(L, _MANGLED[("decompress", mode)])(
                ctypes.c_void_p(probe.data_ptr()), ctypes.c_void_p(cmp_buf.data_ptr()),
                ctypes.c_size_t(n), ctypes.c_size_t(int(size.value)),
                ctypes.c_float(eps), None)
            torch.cuda.synchronize()
            out["unwritten_elements"] = int((~torch.isfinite(probe)).sum())

        # Error accounting, in float64 on the host. Computing a max-abs difference
        # in the precision being audited lets the precision hide its own error.
        #
        # Three quantities, deliberately separate:
        #   E32 = max|y - x|   the codec's own error, fp32 decode against fp32 source
        #   Ebf = max|z - x|   what a receiver actually gets, after the bf16 downcast
        #   R   = max|z - y|   the rounding the downcast added, measured not assumed
        #
        # `within_eps_fp32` is the codec's contract and is checked at eps + TAU.
        # The earlier `eps * 1.001` was a relative slack that grows with eps and has
        # no numerical justification; TAU is an absolute, pre-declared tolerance.
        #
        # There is deliberately no boolean claiming the *bf16* result satisfies the
        # original eps. It frequently does not -- the downcast adds up to half a bf16
        # ulp on top of whatever the codec guaranteed -- and an audit of 504 earlier
        # combinations found 489 of them outside eps by that margin while every one
        # of them satisfied the fp32 bound. `within_eps_plus_rounding` is a posterior
        # check against eps + TAU + this run's own measured R, not a guarantee the
        # codec offers for bf16.
        back = dec.to(torch.bfloat16)
        src = x_bf16.cuda().reshape(-1)
        x64 = d32.double().cpu()
        y64 = dec.double().cpu()
        z64 = back.double().cpu()
        e32 = float((y64 - x64).abs().max())
        ebf = float((z64 - x64).abs().max())
        rnd = float((z64 - y64).abs().max())
        out["max_error_fp32"] = e32
        out["max_error_bf16"] = ebf
        out["rounding_bf16"] = rnd
        out["eps"] = float(eps)
        out["e32_over_eps"] = e32 / eps if eps > 0 else float("inf")
        out["ebf_over_eps"] = ebf / eps if eps > 0 else float("inf")
        out["finite"] = bool(torch.isfinite(dec).all() and torch.isfinite(back).all())
        # finite first: a NaN compares false against every bound, so checking the
        # bound first lets a non-finite reconstruction pass as "not exceeding" it
        out["within_eps_fp32"] = out["finite"] and e32 <= eps + TAU
        out["within_eps_plus_rounding"] = out["finite"] and ebf <= eps + TAU + rnd
        out["within_eps_fp32_legacy_1p001"] = e32 <= eps * 1.001
    return out
