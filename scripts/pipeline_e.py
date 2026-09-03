"""Phase E — does overlapping the stages recover what per-tensor granularity costs?

The serial measurement put cuSZp at 76% of the compressed path (encode 19.9 ms +
decode 17.1 ms against 9.0 ms of transfer), and the whole path 2.1x slower than
moving the cache raw. If those stages can run concurrently the steady state is
max(encode, transfer, decode) rather than their sum, which would be 19.9 ms
against a 23.5 ms raw baseline -- enough to flip the decision.

The chunks here are the cache's own 56 tensors, not a re-slicing of one array.
Each is compressed whole, so nothing is given up in ratio, and the streaming unit
is the one a real KV transfer already has.

E0 is a feasibility gate before any of that: cuSZp takes a stream argument, but
whether two encodes actually overlap on separate streams -- rather than serialising
on an internal device-wide sync -- is an empirical question, and the answer
decides whether the rest of the phase is worth implementing.
"""
import ctypes, statistics as st, sys, time, json
sys.path.insert(0, "/workspace/boundrelay")
import torch
from pathlib import Path
from boundrelay.codec.cuszp_bridge import _MANGLED, lib

cache = [torch.load(p).to("cuda:0") for p in
         sorted(Path("/workspace/corpus_v2/fullcache_d00_L2048").glob("l*.pt"))]
N = len(cache); raw_bytes = sum(x.numel()*2 for x in cache)
L = lib(); C, MODE, IT = 0.10, "fixed", 10
eps = [C*float(x.float().std()) for x in cache]
f32  = [x.float().contiguous().reshape(-1) for x in cache]
scr  = [torch.empty(x.numel()*4+4096, dtype=torch.uint8, device="cuda:0") for x in cache]
recv = [torch.empty(x.numel()*4+4096, dtype=torch.uint8, device="cuda:1") for x in cache]
dst  = [torch.empty(x.numel(), dtype=torch.float32, device="cuda:1") for x in cache]
host = torch.empty(raw_bytes*2, dtype=torch.uint8, device="cpu", pin_memory=True)
sizes = [0]*N

def enc(i, stream=None):
    n = ctypes.c_size_t(0)
    getattr(L,_MANGLED[("compress",MODE)])(
        ctypes.c_void_p(f32[i].data_ptr()), ctypes.c_void_p(scr[i].data_ptr()),
        ctypes.c_size_t(f32[i].numel()), ctypes.byref(n), ctypes.c_float(eps[i]),
        ctypes.c_void_p(stream.cuda_stream if stream else 0))
    sizes[i] = int(n.value)

def dec(i, stream=None):
    getattr(L,_MANGLED[("decompress",MODE)])(
        ctypes.c_void_p(dst[i].data_ptr()), ctypes.c_void_p(recv[i].data_ptr()),
        ctypes.c_size_t(dst[i].numel()), ctypes.c_size_t(sizes[i]),
        ctypes.c_float(eps[i]), ctypes.c_void_p(stream.cuda_stream if stream else 0))

# ---------------- E0 feasibility gate ----------------
torch.cuda.set_device(0)
s1, s2 = torch.cuda.Stream(device=0), torch.cuda.Stream(device=0)
for i in (0,1): enc(i)
torch.cuda.synchronize(0)
t=time.perf_counter()
for _ in range(IT):
    with torch.cuda.stream(s1): enc(0, s1)
    with torch.cuda.stream(s2): enc(1, s2)
    torch.cuda.synchronize(0)
concurrent = (time.perf_counter()-t)/IT
t=time.perf_counter()
for _ in range(IT):
    enc(0); enc(1); torch.cuda.synchronize(0)
serial2 = (time.perf_counter()-t)/IT
overlap = 1 - concurrent/serial2
print(f"E0 gate: two encodes serial {serial2*1e3:.2f} ms, on separate streams "
      f"{concurrent*1e3:.2f} ms -> overlap efficiency {overlap:+.1%}")
if overlap < 0.10:
    print("E0 FAILED: encodes do not overlap; recording as a streamability limitation")
    json.dump({"gate":"failed","overlap":overlap,"serial2_s":serial2,
               "concurrent_s":concurrent}, open("/workspace/pipeline.json","w"), indent=2)
    raise SystemExit(0)
print("E0 passed\n")

# ---------------- four arms ----------------
def arm_raw_bulk():
    for x in cache:
        nb = x.numel()*2
        host[:nb].copy_(x.reshape(-1).view(torch.uint8)); recv[0][:nb].copy_(host[:nb])

def arm_serial():
    torch.cuda.set_device(0)
    for i in range(N): enc(i)
    torch.cuda.synchronize(0)
    for i in range(N):
        host[:sizes[i]].copy_(scr[i][:sizes[i]]); recv[i][:sizes[i]].copy_(host[:sizes[i]])
    torch.cuda.set_device(1)
    for i in range(N): dec(i)
    torch.cuda.synchronize(1); torch.cuda.set_device(0)

def arm_pipeline():
    """encode(i) || transfer(i-1) || decode(i-2) across the cache's own tensors."""
    torch.cuda.set_device(0)
    se = torch.cuda.Stream(device=0)
    torch.cuda.set_device(1)
    sd = torch.cuda.Stream(device=1)
    torch.cuda.set_device(0)
    for i in range(N+2):
        if i < N:
            with torch.cuda.stream(se): enc(i, se)
        if 1 <= i <= N:
            j = i-1
            torch.cuda.current_stream(0).wait_stream(se) if j == i-1 else None
            se.synchronize()
            host[:sizes[j]].copy_(scr[j][:sizes[j]]); recv[j][:sizes[j]].copy_(host[:sizes[j]])
        if i >= 2:
            k = i-2
            torch.cuda.set_device(1)
            with torch.cuda.stream(sd): dec(k, sd)
            torch.cuda.set_device(0)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)

def timeit(fn, n=IT):
    for _ in range(3): fn()
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn()
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        ts.append(time.perf_counter()-t)
    return st.median(ts), sorted(ts)[-2]

res={}
for name, fn in (("raw bulk", arm_raw_bulk), ("serial compressed", arm_serial),
                 ("pipelined compressed", arm_pipeline)):
    m,p95 = timeit(fn)
    res[name] = {"median_s": m, "p95_s": p95}
    print(f"  {name:>22}: {m*1e3:8.2f} ms   P95 {p95*1e3:8.2f} ms")

raw = res["raw bulk"]["median_s"]; ser = res["serial compressed"]["median_s"]
pip = res["pipelined compressed"]["median_s"]
print(f"\nserial   net {(raw-ser)*1e3:+8.2f} ms  -> {'COMPRESS' if raw>ser else 'BYPASS'}")
print(f"pipeline net {(raw-pip)*1e3:+8.2f} ms  -> {'COMPRESS' if raw>pip else 'BYPASS'}")
print(f"pipeline recovers {100*(ser-pip)/ser:.1f}% of the serial compressed time")
print(f"break-even link speed: serial {raw_bytes/ser/1e9:.2f} GB/s, "
      f"pipelined {raw_bytes/pip/1e9:.2f} GB/s")
res.update({"gate":"passed","overlap_efficiency":overlap,"raw_bytes":raw_bytes,
            "comp_bytes":sum(sizes),"n_tensors":N,"c":C,"mode":MODE})
json.dump(res, open("/workspace/pipeline.json","w"), indent=2)
