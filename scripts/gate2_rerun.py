"""Gate 2, re-measured — and measured against the version that produced the
number now flagged as suspect.

The original Gate 2 ran encode on GPU0 against decode on GPU1 and reported 30%
overlap efficiency. Every buffer it used was uninitialised: the compressed
staging buffer on GPU0, the receive buffer on GPU1 (only ``[:sz]`` of it was
filled), and the decode destination. cuSZp turns out to require zeroed buffers on
both sides — it reads past ``cmpSize`` and it does not write elements it expects
to be zero — so that run was timing a decode of partly-garbage input into a
destination whose result nothing checked.

A plain re-run on new hardware would confound the fix with the hardware. So both
variants run here, back to back, in one process on one host:

* ``as_measured`` reproduces the original exactly, uninitialised buffers and all.
* ``corrected`` zeroes the staging buffer, zeroes the receive buffer past the
  compressed size, zeroes the destination before every decode, and **checks the
  reconstruction**, which the original never did.

The difference between the two columns is the fix. The difference between this
host and the A40 the original ran on is then the only thing left to attribute to
hardware, and it applies to both columns equally.

Zeroing the destination is not free and is not optional, so decode is reported
twice: the codec call alone, and the memset plus the call, which is what a
receiver actually pays. Overlap efficiency is computed from both.
"""

import ctypes
import json
import statistics as st
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/boundrelay")
from boundrelay.codec.cuszp_bridge import _MANGLED, lib  # noqa: E402

CACHE = Path("/workspace/corpus_v2/fullcache_d00_L2048")
MODE = "fixed"
C = 0.10
ITERS = 30
WARMUP = 5
REPS = 5


def environment() -> dict:
    return {"torch": torch.__version__,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cuda": torch.version.cuda,
            "capability": [torch.cuda.get_device_capability(i)
                           for i in range(torch.cuda.device_count())]}


def measure(f, s0, r1, d1, sz, eps, L, zero_dst: bool) -> dict:
    """Encode on GPU0, decode on GPU1, alone and together.

    ``zero_dst`` is the contract the original run did not know about. It is a
    parameter rather than a constant so the two variants differ in exactly the
    thing under test.
    """
    n = f.numel()

    def do_enc():
        torch.cuda.set_device(0)
        mm = ctypes.c_size_t(0)
        getattr(L, _MANGLED[("compress", MODE)])(
            ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(s0.data_ptr()),
            ctypes.c_size_t(n), ctypes.byref(mm), ctypes.c_float(eps), None)

    def do_dec():
        torch.cuda.set_device(1)
        getattr(L, _MANGLED[("decompress", MODE)])(
            ctypes.c_void_p(d1.data_ptr()), ctypes.c_void_p(r1.data_ptr()),
            ctypes.c_size_t(n), ctypes.c_size_t(sz), ctypes.c_float(eps), None)

    def do_dec_zeroed():
        torch.cuda.set_device(1)
        d1.zero_()
        do_dec()

    dec_call = do_dec_zeroed if zero_dst else do_dec

    def loop(fn, k):
        for _ in range(k):
            fn()

    def timed(fn, k=ITERS) -> float:
        loop(fn, WARMUP)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        t = time.perf_counter()
        loop(fn, k)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        return (time.perf_counter() - t) / k

    def two_threads() -> float:
        loop(do_enc, WARMUP); loop(dec_call, WARMUP)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        t = time.perf_counter()
        th = [threading.Thread(target=loop, args=(do_enc, ITERS)),
              threading.Thread(target=loop, args=(dec_call, ITERS))]
        for x in th: x.start()
        for x in th: x.join()
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        return (time.perf_counter() - t) / ITERS

    reps: dict[str, list[float]] = {k: [] for k in
                                    ("encode", "decode", "decode_raw", "one_thread", "two_threads")}
    for _ in range(REPS):
        reps["encode"].append(timed(do_enc))
        reps["decode"].append(timed(dec_call))
        reps["decode_raw"].append(timed(do_dec))
        reps["one_thread"].append(timed(lambda: (do_enc(), dec_call())))
        reps["two_threads"].append(two_threads())

    m = {k: st.median(v) for k, v in reps.items()}
    e, d = m["encode"], m["decode"]
    return {**{f"{k}_s": v for k, v in m.items()},
            "serial_sum_s": e + d,
            "overlap_one_thread": 1 - (m["one_thread"] - max(e, d)) / min(e, d),
            "overlap_two_threads": 1 - (m["two_threads"] - max(e, d)) / min(e, d),
            "cause": "host-blocking compress" if m["two_threads"] < 0.85 * (e + d)
                     else "device-level serialisation",
            "reps_s": reps}


def variant(x, name: str, zeroed: bool, L) -> dict:
    """One end of the comparison. ``zeroed`` switches every buffer at once,
    because that is how the original was wrong: not in one buffer, in all three."""
    eps = C * float(x.float().std())
    f = x.to("cuda:0").float().contiguous().reshape(-1)
    n = f.numel()
    alloc = torch.zeros if zeroed else torch.empty

    s0 = alloc(n * 4 + 4096, dtype=torch.uint8, device="cuda:0")
    m = ctypes.c_size_t(0)
    torch.cuda.set_device(0)
    getattr(L, _MANGLED[("compress", MODE)])(
        ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(s0.data_ptr()),
        ctypes.c_size_t(n), ctypes.byref(m), ctypes.c_float(eps), None)
    torch.cuda.synchronize(0)
    sz = int(m.value)

    # the original filled only [:sz] of an uninitialised receive buffer; cuSZp
    # reads past cmpSize, so the tail is part of the input whether or not it was
    # meant to be
    r1 = alloc(s0.numel(), dtype=torch.uint8, device="cuda:1")
    r1[:sz].copy_(s0[:sz].cpu())
    d1 = alloc(n, dtype=torch.float32, device="cuda:1")

    res = measure(f, s0, r1, d1, sz, eps, L, zero_dst=zeroed)

    # what did the decode actually produce? the original checked nothing
    torch.cuda.set_device(1)
    if zeroed:
        d1.zero_()
    getattr(L, _MANGLED[("decompress", MODE)])(
        ctypes.c_void_p(d1.data_ptr()), ctypes.c_void_p(r1.data_ptr()),
        ctypes.c_size_t(n), ctypes.c_size_t(sz), ctypes.c_float(eps), None)
    torch.cuda.synchronize(1)
    # via host, deliberately. On this host a direct device-0 -> device-1 copy
    # reports success and delivers zeros (see `peer_copy_audit.py`), so a
    # reference fetched across the peer path would fail every tensor and blame
    # the codec for it. This is the second rented multi-GPU host on which that
    # has happened, and it is why the transport in this project is host-staged.
    err = float((f.cpu() - d1.cpu()).abs().max())

    probe = torch.full((n,), float("nan"), dtype=torch.float32, device="cuda:1")
    getattr(L, _MANGLED[("decompress", MODE)])(
        ctypes.c_void_p(probe.data_ptr()), ctypes.c_void_p(r1.data_ptr()),
        ctypes.c_size_t(n), ctypes.c_size_t(sz), ctypes.c_float(eps), None)
    torch.cuda.synchronize(1)
    unwritten = int((~torch.isfinite(probe)).sum())
    torch.cuda.set_device(0)
    del probe

    return {"variant": name, "buffers": "zeroed" if zeroed else "uninitialised",
            "eps": eps, "cmp_bytes": sz, "n": n,
            "max_error_fp32": err, "within_eps": err <= eps * 1.001,
            "unwritten_elements": unwritten, **res}


def main() -> int:
    env = environment()
    print(json.dumps(env), flush=True)
    if torch.cuda.device_count() < 2:
        raise SystemExit("Gate 2 is a two-device question; this host has "
                         f"{torch.cuda.device_count()}")

    L = lib()
    ks = sorted(CACHE.glob("l*_k.pt"))
    vs = sorted(CACHE.glob("l*_v.pt"))
    # layer 0 K is the tensor the original Gate 2 used; the rest check that
    # whatever the answer is, it is not a property of one tensor or one kind
    picks = [ks[0], ks[len(ks) // 2], ks[-1], vs[0], vs[len(vs) // 2], vs[-1]]
    out = {"environment": env, "c": C, "mode": MODE, "iters": ITERS,
           "reps": REPS, "tensors": {}}

    for p in picks:
        x = torch.load(p)
        print(f"\n=== {p.stem}  {tuple(x.shape)} ===", flush=True)
        rows = {}
        for name, zeroed in (("as_measured", False), ("corrected", True)):
            r = variant(x, name, zeroed, L)
            rows[name] = r
            print(f"  {name:<12} buffers={r['buffers']:<14} "
                  f"enc {r['encode_s']*1e3:6.2f}  dec {r['decode_s']*1e3:6.2f} "
                  f"(codec alone {r['decode_raw_s']*1e3:6.2f})  "
                  f"1thr {r['one_thread_s']*1e3:6.2f}  2thr {r['two_threads_s']*1e3:6.2f}  "
                  f"serial {r['serial_sum_s']*1e3:6.2f}", flush=True)
            print(f"  {'':12} overlap 1thr {100*r['overlap_one_thread']:5.1f}%  "
                  f"2thr {100*r['overlap_two_threads']:5.1f}%   "
                  f"max|err| {r['max_error_fp32']:.4g} vs eps {r['eps']:.4g} "
                  f"-> {'within' if r['within_eps'] else 'VIOLATED'}   "
                  f"unwritten {r['unwritten_elements']}", flush=True)
        out["tensors"][p.stem] = rows

    # Six tensors were enough to show the two buffer regimes differ; they are not
    # enough to quote an overlap number, because the between-tensor spread turned
    # out to be wider than the effect. The sweep runs the corrected regime over
    # every tensor in the cache and reports the distribution instead of a point.
    print("\n=== sweep: corrected regime over the whole cache ===", flush=True)
    sweep = {}
    for i, p in enumerate(sorted(CACHE.glob("l*.pt")), 1):
        x = torch.load(p)
        r = variant(x, "corrected", True, L)
        sweep[p.stem] = r
        if i % 8 == 0 or i == 1:
            print(f"  {i}/56 {p.stem}  overlap 1thr {100*r['overlap_one_thread']:6.1f}%  "
                  f"2thr {100*r['overlap_two_threads']:6.1f}%  "
                  f"{'within' if r['within_eps'] else 'VIOLATED'}", flush=True)
    out["sweep"] = sweep

    import statistics as _st
    for key, label in (("overlap_one_thread", "one host thread"),
                       ("overlap_two_threads", "two host threads")):
        v = sorted(100 * r[key] for r in sweep.values())
        n = len(v)
        out.setdefault("sweep_summary", {})[key] = {
            "n": n, "median": _st.median(v), "q1": v[n // 4], "q3": v[3 * n // 4],
            "min": v[0], "max": v[-1],
            "n_above_90pct": sum(1 for z in v if z >= 90)}
        print(f"  {label:<16} overlap median {_st.median(v):6.1f}%  "
              f"IQR [{v[n//4]:.1f}, {v[3*n//4]:.1f}]  range [{v[0]:.1f}, {v[-1]:.1f}]  "
              f"reaching 90%: {sum(1 for z in v if z >= 90)}/{n}", flush=True)
    bad = [k for k, r in sweep.items() if not r["within_eps"]]
    print(f"  bound violations across the sweep: {len(bad)}/{len(sweep)}", flush=True)
    out["sweep_violations"] = bad

    dst = Path("/workspace/out/gate2_rerun.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
