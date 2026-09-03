"""Phase C — real BF16 KV transport, end to end.

Everything a receiver actually waits for is inside one timing region:

    T = T_bf16_to_f32 + T_encode + T_header + T_transfer(actual bytes)
        + T_decode + T_f32_to_bf16

The dtype conversions are in there because cuSZp reads fp32 while KV is bf16, so
a real path pays an upcast and a downcast that earlier cost models omitted. The
transfer moves the **actual** compressed bytes, not a size implied by a ratio.

The payload is a real full cache: 56 K/V tensors from one request, each
compressed on its own and shipped as a framed collection with a size table.
Concatenating them into one array first would compress something no system ever
holds, and would hand the codec cross-tensor structure that is not there.

Two transport modes on the *same* hardware, so the comparison isolates the path
rather than the machine:

``p2p``      direct device-to-device copy
``staged``   through pinned host memory

Both arms — raw and compressed — use the same request, the same tensors and the
same copy primitive within a mode.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C_GRID = [0.01, 0.03, 0.10]
MODES = ["fixed", "plain"]


def load_cache(d: Path) -> list[torch.Tensor]:
    return [torch.load(p) for p in sorted(d.glob("l*.pt"))]


def _sync(*devs):
    for d in devs:
        torch.cuda.synchronize(d)


def transfer(src: torch.Tensor, dst_dev: int, how: str, host_buf=None) -> torch.Tensor:
    if how == "p2p":
        return src.to(f"cuda:{dst_dev}", non_blocking=False)
    host_buf[: src.numel()].copy_(src.reshape(-1))
    out = torch.empty_like(src, device=f"cuda:{dst_dev}")
    out.reshape(-1).copy_(host_buf[: src.numel()])
    return out


def run_raw(cache, how, host_buf, iters):
    ts = []
    for _ in range(iters):
        _sync(0, 1)
        t0 = time.perf_counter()
        for x in cache:
            transfer(x, 1, how, host_buf)
        _sync(0, 1)
        ts.append(time.perf_counter() - t0)
    return ts


def run_compressed(cache, c, mode, how, host_u8, iters):
    """Upcast, encode, ship actual bytes + size table, decode, downcast."""
    from boundrelay.codec.cuszp_bridge import _MANGLED, lib

    L = lib()
    eps = [c * float(x.float().std()) for x in cache]
    scratch = [torch.empty(x.numel() * 4 + 4096, dtype=torch.uint8, device="cuda:0")
               for x in cache]
    dst = [torch.empty(x.numel(), dtype=torch.float32, device="cuda:1") for x in cache]

    ts, total_bytes = [], 0
    for it in range(iters):
        _sync(0, 1)
        t0 = time.perf_counter()
        sizes = []
        torch.cuda.set_device(0)
        for i, x in enumerate(cache):
            f32 = x.to("cuda:0", non_blocking=False).float().contiguous().reshape(-1)
            n = ctypes.c_size_t(0)
            getattr(L, _MANGLED[("compress", mode)])(
                ctypes.c_void_p(f32.data_ptr()), ctypes.c_void_p(scratch[i].data_ptr()),
                ctypes.c_size_t(f32.numel()), ctypes.byref(n),
                ctypes.c_float(eps[i]), None)
            _sync(0)
            sizes.append(int(n.value))
        # size table is part of the payload: the receiver cannot frame without it
        hdr = torch.tensor(sizes, dtype=torch.int64)
        recv_hdr = hdr.to("cuda:1").cpu().tolist()
        for i, sz in enumerate(recv_hdr):
            payload = scratch[i][:sz]
            got = transfer(payload, 1, how, host_u8)
            torch.cuda.set_device(1)
            getattr(L, _MANGLED[("decompress", mode)])(
                ctypes.c_void_p(dst[i].data_ptr()), ctypes.c_void_p(got.data_ptr()),
                ctypes.c_size_t(dst[i].numel()), ctypes.c_size_t(sz),
                ctypes.c_float(eps[i]), None)
            _sync(1)
            dst[i].to(torch.bfloat16)                     # the downcast a receiver pays
            torch.cuda.set_device(0)
        _sync(0, 1)
        ts.append(time.perf_counter() - t0)
        if it == 0:
            total_bytes = sum(sizes)
    return ts, total_bytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("/workspace/corpus_v2/fullcache_d00_L2048"))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("/workspace/transport_e2e.json"))
    ap.add_argument("--modes", nargs="+", default=["p2p", "staged"])
    a = ap.parse_args()

    cache = [t.to("cuda:0") for t in load_cache(a.cache)]
    raw_bytes = sum(x.numel() * 2 for x in cache)
    print(f"{len(cache)} tensors, {raw_bytes/1e6:.1f} MB bf16, "
          f"p2p={torch.cuda.can_device_access_peer(0,1)}", flush=True)

    host = torch.empty(max(x.numel() for x in cache), dtype=torch.bfloat16,
                       device="cpu", pin_memory=True)
    host_u8 = torch.empty(raw_bytes * 2, dtype=torch.uint8, device="cpu", pin_memory=True)

    res = {"raw_bytes": raw_bytes, "n_tensors": len(cache), "iters": a.iters,
           "p2p_capable": bool(torch.cuda.can_device_access_peer(0, 1)), "cells": {}}

    for how in a.modes:
        run_raw(cache, how, host, 3)
        ts = run_raw(cache, how, host, a.iters)
        res["cells"][f"raw|{how}"] = {"median_s": st.median(ts), "p95_s": sorted(ts)[-2],
                                      "bytes": raw_bytes, "effective_GBps": raw_bytes/st.median(ts)/1e9}
        print(f"  raw        {how:>7}: {st.median(ts)*1e3:8.2f} ms  "
              f"{raw_bytes/st.median(ts)/1e9:6.2f} GB/s", flush=True)

        for c in C_GRID:
            for mode in MODES:
                run_compressed(cache, c, mode, how, host_u8, 2)
                ts, nb = run_compressed(cache, c, mode, how, host_u8, a.iters)
                key = f"cuszp|{how}|c{c:g}|{mode}"
                res["cells"][key] = {"median_s": st.median(ts), "p95_s": sorted(ts)[-2],
                                     "bytes": nb, "ratio": raw_bytes/nb}
                d = res["cells"][f"raw|{how}"]["median_s"] - st.median(ts)
                print(f"  cuszp c={c:<5g} {mode:>7} {how:>7}: {st.median(ts)*1e3:8.2f} ms  "
                      f"{raw_bytes/nb:5.2f}x  net {d*1e3:+8.2f} ms  "
                      f"{'COMPRESS' if d>0 else 'bypass'}", flush=True)

    a.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
