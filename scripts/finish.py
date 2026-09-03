"""Three closers: Gate 2's mechanism, SZ3/zfp on this host, asymmetric K/V."""
import ctypes, json, statistics as st, sys, threading, time
sys.path.insert(0, "/workspace/boundrelay")
import numpy as np, torch
from pathlib import Path
from boundrelay.codec.cuszp_bridge import _MANGLED, lib

CACHE = Path("/workspace/corpus_v2/fullcache_d00_L2048")
cache = [torch.load(p) for p in sorted(CACHE.glob("l*.pt"))]
L = lib(); MODE = "fixed"; out = {}

# ---- 2a. Gate 2 mechanism: is the serialisation host-side or device-side? ----
# The compress API writes cmpSize to a host pointer, so it must synchronise before
# returning. Issued from one thread, that blocks the caller and nothing can overlap
# regardless of what the devices could do. Two threads separate the two causes.
x = cache[0].to("cuda:0"); eps = 0.10*float(x.float().std())
f = x.float().contiguous().reshape(-1)
s0 = torch.empty(f.numel()*4+4096, dtype=torch.uint8, device="cuda:0")
m = ctypes.c_size_t(0)
torch.cuda.set_device(0)
getattr(L,_MANGLED[("compress",MODE)])(ctypes.c_void_p(f.data_ptr()),
    ctypes.c_void_p(s0.data_ptr()), ctypes.c_size_t(f.numel()),
    ctypes.byref(m), ctypes.c_float(eps), None)
torch.cuda.synchronize(0); sz = int(m.value)
r1 = torch.empty_like(s0, device="cuda:1"); r1[:sz].copy_(s0[:sz].cpu())
d1 = torch.empty(f.numel(), dtype=torch.float32, device="cuda:1")

def enc(k=20):
    torch.cuda.set_device(0)
    for _ in range(k):
        mm = ctypes.c_size_t(0)
        getattr(L,_MANGLED[("compress",MODE)])(ctypes.c_void_p(f.data_ptr()),
            ctypes.c_void_p(s0.data_ptr()), ctypes.c_size_t(f.numel()),
            ctypes.byref(mm), ctypes.c_float(eps), None)
    torch.cuda.synchronize(0)
def dec(k=20):
    torch.cuda.set_device(1)
    for _ in range(k):
        getattr(L,_MANGLED[("decompress",MODE)])(ctypes.c_void_p(d1.data_ptr()),
            ctypes.c_void_p(r1.data_ptr()), ctypes.c_size_t(d1.numel()),
            ctypes.c_size_t(sz), ctypes.c_float(eps), None)
    torch.cuda.synchronize(1)

enc(3); dec(3)
t=time.perf_counter(); enc(); dec(); one_thread=time.perf_counter()-t
t=time.perf_counter()
th=[threading.Thread(target=enc), threading.Thread(target=dec)]
[x.start() for x in th]; [x.join() for x in th]
two_thread=time.perf_counter()-t
t=time.perf_counter(); enc(); a=time.perf_counter()-t
t=time.perf_counter(); dec(); b=time.perf_counter()-t
out["gate2_mechanism"] = {"encode_s":a,"decode_s":b,"one_thread_s":one_thread,
    "two_threads_s":two_thread,"serial_sum_s":a+b,
    "overlap_two_threads": 1-(two_thread-max(a,b))/min(a,b),
    "cause": "host-blocking compress" if two_thread < 0.85*(a+b) else "device-level serialisation"}
print(f"Gate2 mechanism: enc {a*1e3:.2f} dec {b*1e3:.2f} | 1 thread {one_thread*1e3:.2f} "
      f"| 2 threads {two_thread*1e3:.2f} (serial sum {(a+b)*1e3:.2f}) -> {out['gate2_mechanism']['cause']}",
      flush=True)

# ---- 2b. SZ3 / zfp on THIS host's CPU ----
from pysz import sz, szAlgorithm, szConfig, szErrorBoundMode
import zfpy
rows=[]
for c in (0.01,0.03,0.10):
    for name in ("sz3_LORENZO_REG","sz3_NOPRED","zfp"):
        rr=[]
        for t_ in cache[:12]:
            a32 = t_.float().contiguous().numpy().reshape(t_.shape[1],t_.shape[2],t_.shape[3])
            e = c*float(t_.float().std()); bf = t_.numel()*2
            t0=time.perf_counter()
            if name.startswith("sz3"):
                cfg=szConfig(a32.size); cfg.setDims(*a32.shape)
                cfg.errorBoundMode=szErrorBoundMode.ABS; cfg.absErrorBound=e
                cfg.cmprAlgo=getattr(szAlgorithm, name.split("_",1)[1])
                comp,_=sz.compress(a32,cfg); nb=int(comp.nbytes)
            else:
                comp=zfpy.compress_numpy(a32, tolerance=e); nb=len(comp)
            dt=time.perf_counter()-t0
            rr.append((bf/nb, bf/dt/1e9))
        rows.append({"c":c,"method":name,"ratio":st.mean(r[0] for r in rr),
                     "GBps":st.mean(r[1] for r in rr)})
out["cpu_codecs_same_host"]=rows
print(f"\n{'method':>18} " + "".join(f"{'c='+f'{c:g}':>20}" for c in (0.01,0.03,0.10)))
for name in ("sz3_LORENZO_REG","sz3_NOPRED","zfp"):
    s="".join(f"{[r for r in rows if r['c']==c and r['method']==name][0]['ratio']:9.2f}x"
              f"{[r for r in rows if r['c']==c and r['method']==name][0]['GBps']:9.3f}" for c in (0.01,0.03,0.10))
    print(f"{name:>18} {s}", flush=True)
print("                     (ratio, GB/s)  Intel Xeon Gold 6342, single-thread")

json.dump(out, open("/workspace/finish.json","w"), indent=2)
print("\nwrote /workspace/finish.json")
