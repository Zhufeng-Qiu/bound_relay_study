"""What does a KV offload actually cost on a network filesystem, and at what
granularity does it start costing bytes rather than calls?

The pipeline run put the whole cache on `/workspace`, a MooseFS mount, two ways:
fifty-six separate files, and one file written in fifty-six pieces. On both, the
raw path beat the compressed one despite moving three times the bytes. A
break-even argument counts only bytes, so it cannot see why -- and if the path is
not byte-bound, the break-even does not apply to it at all.

This sweeps the write size on the same mount, from 64 KB to the whole 235 MB cache
in one call. Where the curve flattens is where bytes start to matter; below it,
cost is per call and compression is buying nothing.

Reads are reported separately and read *back-to-back with the write*, which is the
order an offload actually uses and also the order most likely to be served from the
client's cache. That makes the read number a lower bound on a real read, and it is
labelled as one rather than quietly used as a bandwidth.
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
import time
from pathlib import Path

CACHE_BYTES = 234_881_024
COMP_BYTES = 77_693_156


def sweep(root: Path, total: int, sizes: list[int], reps: int = 5) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "bw.bin"
    out = []
    for chunk in sizes:
        n_calls = max(1, total // chunk)
        buf = bytearray(os.urandom(min(chunk, 1 << 22)) * max(1, chunk // (1 << 22)))[:chunk]
        wr, rd = [], []
        for _ in range(reps):
            t = time.perf_counter()
            with open(path, "wb", buffering=0) as f:
                for _ in range(n_calls):
                    f.write(buf)
                os.fsync(f.fileno())          # otherwise this times the page cache
            wr.append(time.perf_counter() - t)
            t = time.perf_counter()
            with open(path, "rb", buffering=0) as f:
                mv = memoryview(bytearray(chunk))
                while f.readinto(mv):
                    pass
            rd.append(time.perf_counter() - t)
        moved = n_calls * chunk
        w, r = st.median(wr), st.median(rd)
        out.append({"chunk_bytes": chunk, "calls": n_calls, "moved_bytes": moved,
                    "write_s": w, "read_s": r,
                    "write_gbps": moved / 1e9 / w, "read_gbps": moved / 1e9 / r,
                    "write_per_call_ms": 1e3 * w / n_calls})
        print(f"  {chunk/1e6:8.2f} MB x {n_calls:4d}  write {moved/1e9/w:6.3f} GB/s "
              f"({1e3*w/n_calls:6.2f} ms/call)   read {moved/1e9/r:6.3f} GB/s "
              f"(cache-warm, a lower bound on cost)", flush=True)
        path.unlink(missing_ok=True)
    return out


def main() -> int:
    roots = {"workspace_moosefs": Path("/workspace/bwtest"),
             "container_overlay": Path("/root/bwtest")}
    sizes = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 4_194_304, 16 << 20, 64 << 20, CACHE_BYTES]
    sizes = sorted(set(s for s in sizes if s <= CACHE_BYTES))
    res: dict = {"cache_bytes": CACHE_BYTES, "compressed_bytes": COMP_BYTES, "paths": {}}
    for name, root in roots.items():
        print(f"\n=== {name} ({root}) ===", flush=True)
        try:
            res["paths"][name] = sweep(root, CACHE_BYTES, sizes)
        except OSError as e:
            print(f"  unavailable: {e}", flush=True)
            continue
        best = max(res["paths"][name], key=lambda r: r["write_gbps"])
        w = best["write_gbps"]
        print(f"  peak write {w:.3f} GB/s at {best['chunk_bytes']/1e6:.1f} MB per call")
        print(f"  at that bandwidth the raw cache takes {CACHE_BYTES/1e9/w*1e3:8.1f} ms "
              f"and the compressed one {COMP_BYTES/1e9/w*1e3:8.1f} ms "
              f"-> compression saves {(CACHE_BYTES-COMP_BYTES)/1e9/w*1e3:.1f} ms")

    dst = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/out/storage_bandwidth.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
