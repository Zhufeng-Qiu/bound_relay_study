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
              want_decode: bool = True) -> dict:
    """Compress and (optionally) reconstruct, returning bytes and achieved error.

    The error reported is against the **final bf16** value, not the fp32
    intermediate: that is what a receiver actually gets, and the downcast adds up
    to half a bf16 ulp on top of whatever the codec guaranteed in fp32.
    """
    L = lib()
    d32 = x_bf16.cuda().float().contiguous().reshape(-1)
    n = d32.numel()
    cmp_buf = torch.empty(n * 4 + 4096, dtype=torch.uint8, device="cuda")
    size = ctypes.c_size_t(0)

    getattr(L, _MANGLED[("compress", mode)])(
        ctypes.c_void_p(d32.data_ptr()), ctypes.c_void_p(cmp_buf.data_ptr()),
        ctypes.c_size_t(n), ctypes.byref(size), ctypes.c_float(eps), None)
    torch.cuda.synchronize()
    out = {"cmp_bytes": int(size.value), "n": int(n),
           "ratio_vs_bf16": (n * 2) / max(int(size.value), 1)}

    if want_decode:
        dec = torch.empty(n, dtype=torch.float32, device="cuda")
        getattr(L, _MANGLED[("decompress", mode)])(
            ctypes.c_void_p(dec.data_ptr()), ctypes.c_void_p(cmp_buf.data_ptr()),
            ctypes.c_size_t(n), ctypes.c_size_t(int(size.value)),
            ctypes.c_float(eps), None)
        torch.cuda.synchronize()
        back = dec.to(torch.bfloat16)
        src = x_bf16.cuda().reshape(-1)
        out["max_error_fp32"] = float((d32 - dec).abs().max())
        out["max_error_bf16"] = float((src.float() - back.float()).abs().max())
        out["within_eps_fp32"] = out["max_error_fp32"] <= eps * 1.001
    return out
