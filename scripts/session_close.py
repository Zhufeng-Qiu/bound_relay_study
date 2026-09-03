"""One session, one environment lock: close the evidence that Phase C left open.

Five things, all on the same host so nothing is compared across environments:

1. **Closed E2E.** Every stage timed with CUDA events *inside the same iteration*,
   all samples kept. Phase C timed each stage in its own loop and added medians,
   which left 6.29 ms -- 11.4% of the path -- unattributed. Here the residual is
   computed rather than left over: host dispatch is what total wall time minus the
   sum of GPU stage times actually is.
2. **SZ3 and zfp on this host's CPU**, so their timings stop being an Apple-M5
   number compared against A40 GPU numbers.
3. **Gate 2**: does encode on GPU0 overlap decode on GPU1? Phase E measured
   same-GPU encode/encode, which the cross-GPU pipeline never needs, and inferred
   a device-ownership mechanism from kernel signatures. This measures the question
   that was actually being asked.
4. **Asymmetric bounds**: c_K << c_V, the hypothesis the quality result points at.
5. **Storage probe**: whether any local NVMe exists here at all, since /workspace
   is network-mounted.

Every result carries the driver version, topology and pod id, which no earlier
manifest recorded.
"""

from __future__ import annotations

import ctypes, json, os, statistics as st, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from boundrelay.codec.cuszp_bridge import _MANGLED, lib

CACHE = Path("/workspace/corpus_v2/fullcache_d00_L2048")
C, MODE, ITERS = 0.10, "fixed", 100
OUT = Path("/workspace/session_close.json")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def environment() -> dict:
    return {
        "pod_id": os.environ.get("RUNPOD_POD_ID", "unknown"),
        "driver_version": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        "gpus": sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"),
        "topology": sh("nvidia-smi topo -m | head -4"),
        "cuszp_commit": sh("cd /workspace/cuSZp && git rev-parse HEAD"),
        "torch": torch.__version__, "cuda_built": torch.version.cuda,
        "cpu": sh("lscpu | grep 'Model name' | head -1"),
        "p2p_capable": bool(torch.cuda.can_device_access_peer(0, 1)),
    }


# ---------------------------------------------------------------- 1. closed E2E
def closed_e2e(cache):
    n = len(cache)
    eps = [C * float(x.float().std()) for x in cache]
    scr = [torch.empty(x.numel()*4+4096, dtype=torch.uint8, device="cuda:0") for x in cache]
    recv = [torch.empty(x.numel()*4+4096, dtype=torch.uint8, device="cuda:1") for x in cache]
    dst = [torch.empty(x.numel(), dtype=torch.float32, device="cuda:1") for x in cache]
    host = torch.empty(sum(x.numel()*2 for x in cache)*2, dtype=torch.uint8,
                       device="cpu", pin_memory=True)
    L = lib()
    ev = lambda: torch.cuda.Event(enable_timing=True)
    marks = ["upcast", "encode", "d2h", "h2d", "decode", "downcast"]

    def one_iter(collect):
        e = {m: [ev(), ev()] for m in marks} if collect else None
        sizes = [0]*n
        torch.cuda.set_device(0)
        if collect: e["upcast"][0].record()
        f32 = [x.float().contiguous().reshape(-1) for x in cache]
        if collect: e["upcast"][1].record()
        if collect: e["encode"][0].record()
        for i, f in enumerate(f32):
            m = ctypes.c_size_t(0)
            getattr(L, _MANGLED[("compress", MODE)])(
                ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(scr[i].data_ptr()),
                ctypes.c_size_t(f.numel()), ctypes.byref(m), ctypes.c_float(eps[i]), None)
            sizes[i] = int(m.value)
        if collect: e["encode"][1].record()
        if collect: e["d2h"][0].record()
        for i, s in enumerate(sizes): host[:s].copy_(scr[i][:s])
        if collect: e["d2h"][1].record()
        torch.cuda.set_device(1)
        if collect: e["h2d"][0].record()
        for i, s in enumerate(sizes): recv[i][:s].copy_(host[:s])
        if collect: e["h2d"][1].record()
        if collect: e["decode"][0].record()
        for i, s in enumerate(sizes):
            getattr(L, _MANGLED[("decompress", MODE)])(
                ctypes.c_void_p(dst[i].data_ptr()), ctypes.c_void_p(recv[i].data_ptr()),
                ctypes.c_size_t(dst[i].numel()), ctypes.c_size_t(s),
                ctypes.c_float(eps[i]), None)
        if collect: e["decode"][1].record()
        if collect: e["downcast"][0].record()
        for d in dst: d.to(torch.bfloat16)
        if collect: e["downcast"][1].record()
        torch.cuda.set_device(0)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        return sizes, e

    def raw_iter():
        torch.cuda.set_device(0)
        for x in cache:
            nb = x.numel()*2
            host[:nb].copy_(x.reshape(-1).view(torch.uint8)); recv[0][:nb].copy_(host[:nb])
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    for _ in range(5): one_iter(False); raw_iter()
    # (a) true E2E: no per-stage instrumentation at all
    wall = []
    for _ in range(ITERS):
        t = time.perf_counter(); one_iter(False); wall.append(time.perf_counter()-t)
    raw = []
    for _ in range(ITERS):
        t = time.perf_counter(); raw_iter(); raw.append(time.perf_counter()-t)
    # (b) same iteration, instrumented: GPU time per stage
    stages = {m: [] for m in marks}; inst = []
    for _ in range(ITERS):
        t = time.perf_counter(); sizes, e = one_iter(True); inst.append(time.perf_counter()-t)
        for m in marks: stages[m].append(e[m][0].elapsed_time(e[m][1])/1e3)

    gpu_sum = sum(st.median(v) for v in stages.values())
    return {
        "n_tensors": n, "raw_bytes": sum(x.numel()*2 for x in cache), "comp_bytes": sum(sizes),
        "e2e_median_s": st.median(wall), "e2e_p95_s": sorted(wall)[int(.95*ITERS)],
        "e2e_samples_s": wall,
        "raw_median_s": st.median(raw), "raw_p95_s": sorted(raw)[int(.95*ITERS)],
        "raw_samples_s": raw,
        "instrumented_median_s": st.median(inst),
        "stage_median_s": {m: st.median(v) for m, v in stages.items()},
        "stage_samples_s": stages,
        "gpu_stage_sum_s": gpu_sum,
        "host_residual_s": st.median(wall) - gpu_sum,
    }


# ---------------------------------------------------------------- 3. Gate 2
def gate2(cache):
    """encode on GPU0 against decode on GPU1 -- the question the pipeline asks."""
    x = cache[0]; eps = C*float(x.float().std())
    f = x.float().contiguous().reshape(-1)
    s0 = torch.empty(f.numel()*4+4096, dtype=torch.uint8, device="cuda:0")
    L = lib(); m = ctypes.c_size_t(0)
    torch.cuda.set_device(0)
    getattr(L,_MANGLED[("compress",MODE)])(ctypes.c_void_p(f.data_ptr()),
        ctypes.c_void_p(s0.data_ptr()), ctypes.c_size_t(f.numel()),
        ctypes.byref(m), ctypes.c_float(eps), None)
    torch.cuda.synchronize(0)
    sz = int(m.value)
    r1 = torch.empty_like(s0, device="cuda:1"); r1[:sz].copy_(s0[:sz].cpu())
    d1 = torch.empty(f.numel(), dtype=torch.float32, device="cuda:1")

    def do_enc():
        torch.cuda.set_device(0)
        mm = ctypes.c_size_t(0)
        getattr(L,_MANGLED[("compress",MODE)])(ctypes.c_void_p(f.data_ptr()),
            ctypes.c_void_p(s0.data_ptr()), ctypes.c_size_t(f.numel()),
            ctypes.byref(mm), ctypes.c_float(eps), None)
    def do_dec():
        torch.cuda.set_device(1)
        getattr(L,_MANGLED[("decompress",MODE)])(ctypes.c_void_p(d1.data_ptr()),
            ctypes.c_void_p(r1.data_ptr()), ctypes.c_size_t(d1.numel()),
            ctypes.c_size_t(sz), ctypes.c_float(eps), None)

    def t(fn, k=30):
        for _ in range(5): fn()
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        a=time.perf_counter()
        for _ in range(k): fn()
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        return (time.perf_counter()-a)/k

    te, td = t(do_enc), t(do_dec)
    def both(): do_enc(); do_dec()
    tb = t(both)
    torch.cuda.set_device(0)
    return {"encode_only_s": te, "decode_only_s": td, "issued_together_s": tb,
            "serial_sum_s": te+td, "overlap_efficiency": 1 - (tb-max(te,td))/min(te,td)
            if min(te,td) > 0 else None,
            "verdict": "overlaps" if tb < 0.9*(te+td) else "serialises"}


def main() -> int:
    env = environment()
    print(json.dumps(env, indent=2)[:900], flush=True)
    cache = [torch.load(p).to("cuda:0") for p in sorted(CACHE.glob("l*.pt"))]
    out = {"environment": env, "c": C, "mode": MODE, "iters": ITERS}

    print("\n=== 1. closed E2E ===", flush=True)
    r = closed_e2e(cache); out["closed_e2e"] = r
    RB = r["raw_bytes"]; saved = RB - r["comp_bytes"]
    print(f"  raw            {r['raw_median_s']*1e3:8.2f} ms   P95 {r['raw_p95_s']*1e3:7.2f}")
    print(f"  compressed E2E {r['e2e_median_s']*1e3:8.2f} ms   P95 {r['e2e_p95_s']*1e3:7.2f}")
    for m, v in r["stage_median_s"].items():
        print(f"    {m:>10} {v*1e3:8.3f} ms")
    print(f"    {'GPU sum':>10} {r['gpu_stage_sum_s']*1e3:8.3f} ms")
    print(f"    {'host resid':>10} {r['host_residual_s']*1e3:8.3f} ms  "
          f"({100*r['host_residual_s']/r['e2e_median_s']:.1f}% of E2E)")
    codec = r["e2e_median_s"] - r["stage_median_s"]["d2h"] - r["stage_median_s"]["h2d"]
    be = saved/codec/1e9
    out["break_even_gbps"] = {"definition": "bytes_saved / (E2E - transfer stages)",
                              "value": be, "link_gbps": RB/r["raw_median_s"]/1e9}
    print(f"\n  break-even {be:.2f} GB/s   link {RB/r['raw_median_s']/1e9:.2f} GB/s"
          f"  -> {'COMPRESS' if RB/r['raw_median_s']/1e9 < be else 'BYPASS'}")

    print("\n=== 3. Gate 2: encode(GPU0) vs decode(GPU1) ===", flush=True)
    g = gate2(cache); out["gate2"] = g
    print(f"  encode alone {g['encode_only_s']*1e3:.3f} ms | decode alone "
          f"{g['decode_only_s']*1e3:.3f} ms | together {g['issued_together_s']*1e3:.3f} ms"
          f"  (serial sum {g['serial_sum_s']*1e3:.3f})  -> {g['verdict'].upper()}")

    print("\n=== 5. storage probe ===", flush=True)
    out["storage"] = {"workspace_mount": sh("df -h /workspace | tail -1"),
                      "root_mount": sh("df -h / | tail -1"),
                      "block_devices": sh("lsblk -d -o NAME,ROTA,SIZE,TYPE 2>/dev/null | head -8")}
    print("  " + out["storage"]["workspace_mount"])
    print("  " + out["storage"]["root_mount"])

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
