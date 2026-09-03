"""fsync-acknowledged offload write: raw against compressed, back to back.

The pipeline run put the cache on a MooseFS mount and compression *lost* — 1.17x
slower while moving a third of the bytes — because those writes were never made
durable and the cost fell on per-call overhead rather than on bytes. Swept
separately with `fsync`, the same mount writes at 0.43 GB/s, where a third of the
bytes should be worth a great deal. That was a bandwidth sweep, not an offload.

This is the offload. Two arms, identical in every respect but the payload:

* **raw** — GPU source, D2H, into a single laid-out cache file, one `fsync`.
* **compressed** — upcast, encode, D2H of the *actual compressed bytes*, into the
  same single-file layout by the same code path, one `fsync`.

Both preallocate the file to their own payload size before the clock starts, both
build an offset table the same way, both write tensor by tensor at those offsets,
and both call `fsync` exactly once for the whole cache. The clock runs from **GPU
source ready to `fsync` returning** — so the compressed arm is charged its upcast
and its encode, which is the point.

Before any timing, one pass is written and read back: raw must return the source
byte for byte, compressed must decode inside its error bound. An offload that does
not deliver what it was given is not an offload, and this project has already shipped
one transport benchmark whose receiver never received (`DEFECTS.md` D3).

**What this measures is a write, and the name says so.** `fsync` returning means the
filesystem has acknowledged the bytes; it is not a round trip. Cold restore — read
back after eviction, on a client whose cache holds nothing — is not measured here,
and the read side of this mount is known to be served warm at 3 GB/s, so nothing in
this file should be read as a restore cost.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/boundrelay")
from boundrelay.codec.cuszp_bridge import _MANGLED, lib  # noqa: E402

CACHE = Path("/workspace/corpus_v2/fullcache_d00_L2048")
MODE = "fixed"
C = 0.10


def offsets(sizes: list[int]) -> tuple[list[int], int]:
    """Contiguous layout. Built the same way for both arms so that whatever the
    layout costs, it costs both of them."""
    off, acc = [], 0
    for s in sizes:
        off.append(acc)
        acc += s
    return off, acc


def prepare(cache, L) -> dict:
    """Everything that is not the offload: upcast buffers, compressed bytes, and
    the host staging both arms write from.

    The compressed arm's encode is *not* done here -- it is inside the timed
    region. This only sizes the buffers, which needs one encode to learn the
    compressed length.
    """
    torch.cuda.set_device(0)
    eps = [C * float(x.float().std()) for x in cache]
    scratch = [torch.zeros(x.numel() * 4 + 4096, dtype=torch.uint8, device="cuda:0")
               for x in cache]
    sizes = []
    for i, x in enumerate(cache):
        f = x.float().contiguous().reshape(-1)
        m = ctypes.c_size_t(0)
        getattr(L, _MANGLED[("compress", MODE)])(
            ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(scratch[i].data_ptr()),
            ctypes.c_size_t(f.numel()), ctypes.byref(m), ctypes.c_float(eps[i]), None)
        sizes.append(int(m.value))
    torch.cuda.synchronize(0)
    raw_sizes = [x.numel() * 2 for x in cache]
    host_raw = torch.empty(sum(raw_sizes), dtype=torch.uint8, device="cpu", pin_memory=True)
    host_cmp = torch.empty(sum(sizes), dtype=torch.uint8, device="cpu", pin_memory=True)
    return {"eps": eps, "scratch": scratch, "cmp_sizes": sizes, "raw_sizes": raw_sizes,
            "host_raw": host_raw, "host_cmp": host_cmp}


def offload(cache, L, st_: dict, path: Path, compressed: bool,
            n_writes: int | None, do_fsync: bool = True) -> float:
    """One fsync-acknowledged offload write. Returns seconds.

    ``n_writes`` is how many `write` calls the payload is issued in: ``None`` means
    one per tensor at its offset, which is the layout the spec asks for; ``1`` means
    the whole contiguous buffer in a single call. The difference between those two
    is per-call cost, which the pipeline run found to dominate this mount and which
    a bandwidth-only model does not contain.

    ``do_fsync=False`` reproduces the regime the pipeline run used, where the bytes
    reach the page cache and nothing waits for the filesystem. Both regimes are
    measured here on **one mount in one process**, because the claim being made is
    that durability changes the answer, and a claim like that cannot rest on
    comparing two data centres.
    """
    sizes = st_["cmp_sizes"] if compressed else st_["raw_sizes"]
    host = st_["host_cmp"] if compressed else st_["host_raw"]
    off, total = offsets(sizes)

    # preallocate, outside the clock, identically for both arms
    with open(path, "wb") as f:
        f.truncate(total)

    torch.cuda.set_device(0)
    torch.cuda.synchronize(0)                       # GPU source ready
    t0 = time.perf_counter()

    if compressed:
        for i, x in enumerate(cache):
            f32 = x.float().contiguous().reshape(-1)
            m = ctypes.c_size_t(0)
            getattr(L, _MANGLED[("compress", MODE)])(
                ctypes.c_void_p(f32.data_ptr()), ctypes.c_void_p(st_["scratch"][i].data_ptr()),
                ctypes.c_size_t(f32.numel()), ctypes.byref(m),
                ctypes.c_float(st_["eps"][i]), None)
            assert int(m.value) == sizes[i], (i, int(m.value), sizes[i])
        torch.cuda.synchronize(0)
        for i, s in enumerate(sizes):
            host[off[i]:off[i] + s].copy_(st_["scratch"][i][:s])
    else:
        for i, x in enumerate(cache):
            s = sizes[i]
            host[off[i]:off[i] + s].copy_(x.reshape(-1).view(torch.uint8))
    torch.cuda.synchronize(0)                       # bytes are in host memory

    mv = memoryview(host.numpy())
    fd = os.open(path, os.O_WRONLY)
    try:
        if n_writes == 1:
            os.pwrite(fd, mv[:total], 0)
        else:
            for i, s in enumerate(sizes):
                os.pwrite(fd, mv[off[i]:off[i] + s], off[i])
        if do_fsync:
            os.fsync(fd)                            # the acknowledgement
    finally:
        os.close(fd)
    return time.perf_counter() - t0


def verify(cache, L, st_: dict, path: Path, compressed: bool) -> dict:
    """Read the file back and check it is what was sent."""
    sizes = st_["cmp_sizes"] if compressed else st_["raw_sizes"]
    off, total = offsets(sizes)
    buf = bytearray(total)
    with open(path, "rb", buffering=0) as f:
        n = f.readinto(memoryview(buf))
    assert n == total, (n, total)
    back = torch.frombuffer(buf, dtype=torch.uint8)

    if not compressed:
        worst = 0.0
        for i, x in enumerate(cache):
            # from the buffer at a byte offset rather than by viewing a slice of a
            # uint8 tensor: `view(dtype)` on a slice carries a storage offset and
            # its alignment rules are not something a correctness check should be
            # relying on
            got = torch.frombuffer(buf, dtype=torch.bfloat16,
                                   count=x.numel(), offset=off[i]).reshape(x.shape)
            worst = max(worst, float((x.cpu().float() - got.float()).abs().max()))
        return {"exact": worst == 0.0, "max_abs_diff": worst}

    torch.cuda.set_device(0)
    worst_ratio, worst_abs = 0.0, 0.0
    for i, x in enumerate(cache):
        s = sizes[i]
        # zeroed on both sides: cuSZp reads past cmpSize and skips elements it
        # expects to be zero -- the contract `remeasure_findings.md` documents
        dev = torch.zeros(x.numel() * 4 + 4096, dtype=torch.uint8, device="cuda:0")
        dev[:s].copy_(back[off[i]:off[i] + s].to("cuda:0"))
        dec = torch.zeros(x.numel(), dtype=torch.float32, device="cuda:0")
        getattr(L, _MANGLED[("decompress", MODE)])(
            ctypes.c_void_p(dec.data_ptr()), ctypes.c_void_p(dev.data_ptr()),
            ctypes.c_size_t(x.numel()), ctypes.c_size_t(s),
            ctypes.c_float(st_["eps"][i]), None)
        torch.cuda.synchronize(0)
        err = float((x.float().reshape(-1).cuda() - dec).abs().max())
        worst_abs = max(worst_abs, err)
        worst_ratio = max(worst_ratio, err / st_["eps"][i])
    return {"within_bound": worst_ratio <= 1.001,
            "max_abs_error": worst_abs, "max_error_over_eps": worst_ratio}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+",
                    default=["/workspace/offload", "/root/offload"])
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="/workspace/out/fsync_offload.json")
    a = ap.parse_args()

    L = lib()
    cache = [torch.load(p).to("cuda:0") for p in sorted(CACHE.glob("l*.pt"))]
    stt = prepare(cache, L)
    raw_b, cmp_b = sum(stt["raw_sizes"]), sum(stt["cmp_sizes"])
    print(f"{len(cache)} tensors   raw {raw_b/1e6:.1f} MB   compressed {cmp_b/1e6:.1f} MB   "
          f"ratio {raw_b/cmp_b:.2f}x", flush=True)

    res: dict = {"n_tensors": len(cache), "raw_bytes": raw_b, "comp_bytes": cmp_b,
                 "c": C, "mode": MODE, "reps": a.reps,
                 "gpu": torch.cuda.get_device_name(0),
                 "measurement": "fsync-acknowledged offload write; NOT a round trip. "
                                "Cold restore is not measured.",
                 "paths": {}}

    for root in a.paths:
        d = Path(root)
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"skipping {root}: {e}", flush=True)
            continue
        print(f"\n=== {root} ===", flush=True)
        entry: dict = {}
        configs = [(None, True,  "per-tensor writes, fsync"),
                   (1,    True,  "one write, fsync"),
                   (None, False, "per-tensor writes, no fsync"),
                   (1,    False, "one write, no fsync")]
        for n_writes, do_fsync, label in configs:
            row = {}
            for compressed in (False, True):
                arm = "compressed" if compressed else "raw"
                p = d / f"cache_{arm}.bin"
                offload(cache, L, stt, p, compressed, n_writes, do_fsync)   # verified pass
                v = verify(cache, L, stt, p, compressed)
                ok = v.get("exact", v.get("within_bound"))
                if not ok:
                    raise SystemExit(f"{arm} on {root} did not read back correctly: {v}")
                offload(cache, L, stt, p, compressed, n_writes, do_fsync)   # warm
                s = [offload(cache, L, stt, p, compressed, n_writes, do_fsync)
                     for _ in range(a.reps)]
                nb = cmp_b if compressed else raw_b
                row[arm] = {"median_s": st.median(s), "min_s": min(s), "max_s": max(s),
                            "samples_s": s, "bytes": nb,
                            "gbps_of_payload": nb / 1e9 / st.median(s),
                            "gbps_of_cache": raw_b / 1e9 / st.median(s), "verify": v}
                p.unlink(missing_ok=True)
                print(f"  {label:<28} {arm:<11} {st.median(s)*1e3:8.1f} ms  "
                      f"[{min(s)*1e3:7.1f}, {max(s)*1e3:7.1f}]  "
                      f"{nb/1e9/st.median(s):5.3f} GB/s of payload   "
                      f"{'exact' if v.get('exact') else 'err/eps ' + format(v.get('max_error_over_eps', 0), '.4f')}",
                      flush=True)
            sp = row["raw"]["median_s"] / row["compressed"]["median_s"]
            row["speedup_compressed_over_raw"] = sp
            row["fsync"] = do_fsync
            row["writes_per_cache"] = len(stt["raw_sizes"]) if n_writes is None else 1
            print(f"  {label:<28} {'':11} -> compression {sp:.2f}x "
                  f"{'FASTER' if sp > 1 else 'slower'}", flush=True)
            key = ("per_tensor_writes" if n_writes is None else "single_write") + \
                  ("" if do_fsync else "_nofsync")
            entry[key] = row
        res["paths"][root] = entry

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
