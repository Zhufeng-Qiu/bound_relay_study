"""C13/A4 — attribute the end-to-end compressed cost to its components.

The composed model predicted a benefit; the measurement says compression loses by
2.3x. This decomposes where the time actually goes, on the same full cache, so
the gap is explained rather than just reported.
"""
import ctypes, statistics as st, sys, time, json
sys.path.insert(0, "/workspace/boundrelay")
import torch
from pathlib import Path
from boundrelay.codec.cuszp_bridge import _MANGLED, lib

cache = [torch.load(p).to("cuda:0") for p in
         sorted(Path("/workspace/corpus_v2/fullcache_d00_L2048").glob("l*.pt"))]
raw_bytes = sum(x.numel()*2 for x in cache)
L = lib(); C, MODE, IT = 0.10, "fixed", 12
eps = [C*float(x.float().std()) for x in cache]
scratch = [torch.empty(x.numel()*4+4096, dtype=torch.uint8, device="cuda:0") for x in cache]
dst = [torch.empty(x.numel(), dtype=torch.float32, device="cuda:1") for x in cache]
# the receiver decodes from *its own* memory: with P2P pathological here, a decode
# kernel on device 1 cannot read device-0 scratch, and pretending otherwise would
# time a path no receiver can take
recv = [torch.empty(x.numel()*4+4096, dtype=torch.uint8, device="cuda:1") for x in cache]
host = torch.empty(raw_bytes*2, dtype=torch.uint8, device="cpu", pin_memory=True)

def timeit(fn, n=IT):
    for _ in range(3): fn()
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        ts.append(time.perf_counter()-t)
    return st.median(ts)

f32s=[None]*len(cache); sizes=[0]*len(cache)
def upcast():
    for i,x in enumerate(cache): f32s[i]=x.float().contiguous().reshape(-1)
def encode():
    torch.cuda.set_device(0)
    for i,f in enumerate(f32s):
        n=ctypes.c_size_t(0)
        getattr(L,_MANGLED[("compress",MODE)])(ctypes.c_void_p(f.data_ptr()),
            ctypes.c_void_p(scratch[i].data_ptr()), ctypes.c_size_t(f.numel()),
            ctypes.byref(n), ctypes.c_float(eps[i]), None)
        sizes[i]=int(n.value)
def xfer_c():
    for i,s in enumerate(sizes):
        host[:s].copy_(scratch[i][:s]); recv[i][:s].copy_(host[:s])
def decode():
    torch.cuda.set_device(1)
    for i,s in enumerate(sizes):
        getattr(L,_MANGLED[("decompress",MODE)])(ctypes.c_void_p(dst[i].data_ptr()),
            ctypes.c_void_p(recv[i].data_ptr()), ctypes.c_size_t(dst[i].numel()),
            ctypes.c_size_t(s), ctypes.c_float(eps[i]), None)
    torch.cuda.set_device(0)
def downcast():
    for d in dst: d.to(torch.bfloat16)
def xfer_raw():
    for x in cache:
        host[:x.numel()*2].copy_(x.reshape(-1).view(torch.uint8)); x.to("cuda:1")

upcast(); encode(); xfer_c()
comp_bytes = sum(sizes)
parts = {"bf16->fp32 upcast": timeit(upcast), "cuSZp encode": timeit(encode),
         "transfer (compressed)": timeit(xfer_c), "cuSZp decode": timeit(decode),
         "fp32->bf16 downcast": timeit(downcast)}
raw_t = timeit(xfer_raw)
tot = sum(parts.values())
print(f"full cache: {len(cache)} tensors, {raw_bytes/1e6:.1f} MB bf16 -> {comp_bytes/1e6:.1f} MB "
      f"({raw_bytes/comp_bytes:.2f}x) at c={C}, mode={MODE}\n")
print(f"{'component':>24} {'median ms':>10} {'share':>8} {'GB/s':>9}")
print("-"*55)
for k,v in parts.items():
    b = comp_bytes if "compressed" in k else raw_bytes
    print(f"{k:>24} {v*1e3:9.2f} {100*v/tot:7.1f}% {b/v/1e9:8.2f}")
print("-"*55)
print(f"{'compressed path total':>24} {tot*1e3:9.2f}")
print(f"{'raw transfer (baseline)':>24} {raw_t*1e3:9.2f}          {raw_bytes/raw_t/1e9:8.2f}")
print(f"\nnet: {(raw_t-tot)*1e3:+.2f} ms  -> {'COMPRESS' if raw_t>tot else 'BYPASS'}")
conv = parts["bf16->fp32 upcast"]+parts["fp32->bf16 downcast"]
codec = parts["cuSZp encode"]+parts["cuSZp decode"]
print(f"dtype conversion: {conv*1e3:.2f} ms = {100*conv/(conv+codec):.1f}% of codec-side, "
      f"{100*conv/codec:.1f}% surcharge on encode+decode")
print(f"per-tensor encode: {parts['cuSZp encode']/len(cache)*1e3:.3f} ms "
      f"({raw_bytes/len(cache)/1e6:.1f} MB each)")
json.dump({"parts_s":parts,"raw_transfer_s":raw_t,"raw_bytes":raw_bytes,
           "comp_bytes":comp_bytes,"n_tensors":len(cache),"c":C,"mode":MODE},
          open("/workspace/breakdown.json","w"), indent=2)
