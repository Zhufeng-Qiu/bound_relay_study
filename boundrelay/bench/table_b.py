"""Table B — the scientific-compressor coordinate system.

Places this codec where the error-bounded compression literature already works,
on the same tensors and at the same absolute error bound.

    SZ3 (INTERP_LORENZO)  prediction-based, the reference point for the field.
                          CPU: contributes ratio and error, never throughput.
    zfp (fixed-accuracy)  transform-based, built for spatially-correlated 2D-4D
                          arrays and explicitly not recommended for 1D. Kept as a
                          labelled *structure-sensitive* baseline on the raw
                          [tokens, head_dim] layout, because whether that
                          structure exists here is the question being asked.
    cuSZp (fixed mode)    GPU, and the closest in spirit -- its own documentation
                          targets non-smooth data and ML weights/tokens. Needs a
                          CUDA build, so it is not in this CPU pass.

Two methodological traps, both fatal if missed
----------------------------------------------
SZ3 and zfp read fp32. bf16 -> fp32 is a *lossless* upcast (bf16 is a truncated
fp32), so the values are unchanged and an absolute eps transfers cleanly. But the
ratio denominator must stay at the **bf16 byte count**, or every fp32 baseline
collects a free 2x for reading a wider type it did not need.

Upcast cost -- the conversion itself, and twice the memory traffic -- is reported
in its own column rather than folded into anything.

Wording, everywhere: "benchmarked against ... under matched error settings".
Never "validated against".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from boundrelay.codec import Allocation, CodecConfig, reference

EPS_GRID = [0.05, 0.15, 0.5]


def _sz3(x32: np.ndarray, eps: float) -> dict:
    from pysz import sz, szAlgorithm, szConfig, szErrorBoundMode

    cfg = szConfig(x32.size)
    cfg.setDims(*x32.shape)
    cfg.errorBoundMode = szErrorBoundMode.ABS
    cfg.absErrorBound = eps
    cfg.cmprAlgo = szAlgorithm.INTERP_LORENZO

    t0 = time.perf_counter()
    comp, _ = sz.compress(x32, cfg)
    enc = time.perf_counter() - t0
    dec_arr, _ = sz.decompress(comp, x32.dtype, x32.shape)
    return {"bytes": int(comp.nbytes), "max_error": float(np.abs(x32 - dec_arr).max()),
            "encode_s": enc}


def _zfp(x32: np.ndarray, eps: float) -> dict:
    import zfpy

    t0 = time.perf_counter()
    comp = zfpy.compress_numpy(x32, tolerance=eps)
    enc = time.perf_counter() - t0
    dec = zfpy.decompress_numpy(comp)
    return {"bytes": len(comp), "max_error": float(np.abs(x32 - dec).max()),
            "encode_s": enc}


def _ours(x: torch.Tensor, eps: float, alloc: Allocation) -> dict:
    t0 = time.perf_counter()
    buf, st = reference.encode(x, CodecConfig(eps=eps, allocation=alloc))
    enc = time.perf_counter() - t0
    err = reference.max_abs_error(x, reference.decode(buf))
    return {"bytes": st.compressed_bytes, "max_error": err, "encode_s": enc}


def run(corpus: Path, out: Path) -> dict:
    from boundrelay.bench.table_a import load_corpus

    tensors = load_corpus(corpus)
    res: dict = {"eps_grid": EPS_GRID, "n_tensors": len(tensors),
                 "note": "all ratios are against bf16 bytes (2 B/element); "
                         "SZ3 and zfp consume a lossless bf16->fp32 upcast",
                 "rows": {}}
    agg: dict[str, list[float]] = {}

    for name, x in tensors.items():
        bf16_bytes = x.numel() * 2
        t0 = time.perf_counter()
        x32 = x.float().numpy()                      # lossless upcast
        upcast_s = time.perf_counter() - t0

        for eps in EPS_GRID:
            entries = {
                "sz3": _sz3(x32, eps),
                "zfp": _zfp(x32, eps),
                "ours_blockwise": _ours(x, eps, Allocation.BLOCKWISE),
                "ours_per_channel": _ours(x, eps, Allocation.PER_CHANNEL),
            }
            for k, e in entries.items():
                e["ratio"] = bf16_bytes / e["bytes"]
                e["within_bound"] = e["max_error"] <= eps + 1e-5
                agg.setdefault(f"{k}.eps{eps:g}.ratio", []).append(e["ratio"])
                agg.setdefault(f"{k}.eps{eps:g}.err", []).append(e["max_error"])
                agg.setdefault(f"{k}.eps{eps:g}.enc_s", []).append(e["encode_s"])
            entries["upcast_s"] = upcast_s
            entries["upcast_extra_bytes"] = bf16_bytes      # fp32 doubles traffic
            res["rows"].setdefault(name, {})[f"eps{eps:g}"] = entries

    res["mean"] = {k: sum(v) / len(v) for k, v in agg.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return res
