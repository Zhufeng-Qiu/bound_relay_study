"""B2 -- raw against compressed on three full paths, paired and interleaved.

Every earlier performance number here compared a raw loop against a compressed loop
run at a different moment on a shared machine, and reported the ratio of their
medians. This pairs them: each measurement is a raw run and a compressed run taken
back to back, half the pairs raw-first and half compressed-first, spread over three
segments of the session. The statistic is the ratio *within* a pair, so drift in the
machine moves both arms together instead of becoming a result.

**The endpoint is delivery, not the last kernel launch.**

* move  -- GPU0 holds the raw cache  ->  all 56 independent bf16 outputs live on
  GPU1. Upcast, encode, the zeroing a correct implementation cannot skip, D2H, host
  scheduling, H2D, decode, downcast and the final synchronisation are all inside.
* write -- GPU0 holds the raw cache  ->  payload written, `fsync` returned, file
  closed. No decode: this is a write, and calling it a restore would be wrong.

Layout, offsets, epsilon and buffer allocation happen outside the clock, so these
are **warm replays of a prepared cache**, not the latency of a cache being seen for
the first time. The statistics that preparation would cost are not measured here.

No required speedup, and slow samples stay in.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import random
import statistics as st
import threading
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "/workspace/boundrelay")
from boundrelay.codec.cuszp_bridge import _MANGLED, TAU, lib  # noqa: E402

MODE = "fixed"
C = 0.10
DEPTH = 8


def write_all(fd: int, buf: memoryview, offset: int) -> None:
    view, pos = buf, offset
    while view:
        w = os.pwrite(fd, view, pos)
        if w <= 0:
            raise OSError(f"pwrite returned {w}, {len(view)} bytes left at {pos}")
        view, pos = view[w:], pos + w


class Bench:
    """Buffers and constants for one cache. Everything here is outside the clock."""

    def __init__(self, cache: list[torch.Tensor], L, spool: Path | None):
        self.cache = cache
        self.L = L
        self.n = len(cache)
        self.eps = [float(torch.tensor(C * float(x.float().std(correction=1)),
                                       dtype=torch.float32)) for x in cache]
        self.raw_sizes = [x.numel() * 2 for x in cache]
        self.max_numel = max(x.numel() for x in cache)
        self.slot_bytes = self.max_numel * 4 + 4096

        torch.cuda.set_device(0)
        self.scr = [torch.zeros(self.slot_bytes, dtype=torch.uint8, device="cuda:0")
                    for _ in range(DEPTH)]
        self.host = torch.empty(self.slot_bytes * DEPTH, dtype=torch.uint8,
                                device="cpu", pin_memory=True)
        self.host_flat = torch.empty(sum(self.raw_sizes), dtype=torch.uint8,
                                     device="cpu", pin_memory=True)
        torch.cuda.set_device(1)
        self.recv = [torch.zeros(self.slot_bytes, dtype=torch.uint8, device="cuda:1")
                     for _ in range(DEPTH)]
        self.dec = [torch.zeros(self.max_numel, dtype=torch.float32, device="cuda:1")
                    for _ in range(DEPTH)]
        # the delivery target: per tensor, allocated once, live together
        self.out = [torch.zeros(x.numel(), dtype=torch.bfloat16, device="cuda:1")
                    for x in cache]
        torch.cuda.set_device(0)
        self.spool = spool
        self.cmp_sizes = self._probe()

    def _probe(self) -> list[int]:
        """Compressed lengths, learned once so the file layout is known in advance."""
        sizes = []
        torch.cuda.set_device(0)
        for i, x in enumerate(self.cache):
            f = x.float().contiguous().reshape(-1)
            self.scr[0].zero_()
            m = ctypes.c_size_t(0)
            getattr(self.L, _MANGLED[("compress", MODE)])(
                ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(self.scr[0].data_ptr()),
                ctypes.c_size_t(f.numel()), ctypes.byref(m),
                ctypes.c_float(self.eps[i]), None)
            sizes.append(int(m.value))
        torch.cuda.synchronize(0)
        return sizes

    def arm_zero(self) -> None:
        torch.cuda.set_device(1)
        for o in self.out:
            o.zero_()
        torch.cuda.synchronize(1)
        torch.cuda.set_device(0)

    # ---------------------------------------------------------------- serial
    def serial(self, compressed: bool) -> float:
        self.arm_zero()
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        for i, x in enumerate(self.cache):
            torch.cuda.set_device(0)
            if compressed:
                self.scr[0].zero_()
                f = x.float().contiguous().reshape(-1)
                m = ctypes.c_size_t(0)
                getattr(self.L, _MANGLED[("compress", MODE)])(
                    ctypes.c_void_p(f.data_ptr()),
                    ctypes.c_void_p(self.scr[0].data_ptr()),
                    ctypes.c_size_t(f.numel()), ctypes.byref(m),
                    ctypes.c_float(self.eps[i]), None)
                nb = int(m.value)
                self.host[:nb].copy_(self.scr[0][:nb])
            else:
                nb = self.raw_sizes[i]
                self.host[:nb].copy_(x.reshape(-1).view(torch.uint8))
            torch.cuda.synchronize(0)
            torch.cuda.set_device(1)
            self.recv[0].zero_()
            self.recv[0][:nb].copy_(self.host[:nb])
            if compressed:
                self.dec[0].zero_()
                getattr(self.L, _MANGLED[("decompress", MODE)])(
                    ctypes.c_void_p(self.dec[0].data_ptr()),
                    ctypes.c_void_p(self.recv[0].data_ptr()),
                    ctypes.c_size_t(x.numel()), ctypes.c_size_t(nb),
                    ctypes.c_float(self.eps[i]), None)
                self.out[i].copy_(self.dec[0][:x.numel()])
            else:
                self.out[i].copy_(self.recv[0][:nb].view(torch.bfloat16))
            torch.cuda.synchronize(1)
        torch.cuda.set_device(0)
        return time.perf_counter() - t0

    # -------------------------------------------------------------- pipeline
    def pipeline(self, compressed: bool) -> float:
        self.arm_zero()
        free: queue.Queue = queue.Queue()
        for s in range(DEPTH):
            free.put(s)
        ready: queue.Queue = queue.Queue()
        err: list[BaseException] = []
        hoff = [s * self.slot_bytes for s in range(DEPTH)]

        def producer():
            try:
                torch.cuda.set_device(0)
                s_copy = torch.cuda.Stream(device=0)
                for i, x in enumerate(self.cache):
                    slot = free.get()
                    if slot < 0:
                        return
                    if compressed:
                        self.scr[slot].zero_()
                        f = x.float().contiguous().reshape(-1)
                        m = ctypes.c_size_t(0)
                        getattr(self.L, _MANGLED[("compress", MODE)])(
                            ctypes.c_void_p(f.data_ptr()),
                            ctypes.c_void_p(self.scr[slot].data_ptr()),
                            ctypes.c_size_t(f.numel()), ctypes.byref(m),
                            ctypes.c_float(self.eps[i]), None)
                        nb = int(m.value)
                        src = self.scr[slot][:nb]
                    else:
                        nb = self.raw_sizes[i]
                        src = x.reshape(-1).view(torch.uint8)
                    with torch.cuda.stream(s_copy):
                        self.host[hoff[slot]:hoff[slot] + nb].copy_(src,
                                                                    non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(s_copy)
                    ready.put((i, slot, nb, ev))
                ready.put(None)
            except BaseException as e:      # noqa: BLE001
                err.append(e)
                ready.put(None)

        def consumer():
            try:
                torch.cuda.set_device(1)
                s_copy = torch.cuda.Stream(device=1)
                while True:
                    item = ready.get()
                    if item is None:
                        return
                    i, slot, nb, ev = item
                    ev.synchronize()
                    self.recv[slot].zero_()
                    with torch.cuda.stream(s_copy):
                        self.recv[slot][:nb].copy_(
                            self.host[hoff[slot]:hoff[slot] + nb], non_blocking=True)
                    s_copy.synchronize()
                    x = self.cache[i]
                    if compressed:
                        self.dec[slot].zero_()
                        getattr(self.L, _MANGLED[("decompress", MODE)])(
                            ctypes.c_void_p(self.dec[slot].data_ptr()),
                            ctypes.c_void_p(self.recv[slot].data_ptr()),
                            ctypes.c_size_t(x.numel()), ctypes.c_size_t(nb),
                            ctypes.c_float(self.eps[i]), None)
                        self.out[i].copy_(self.dec[slot][:x.numel()])
                    else:
                        self.out[i].copy_(self.recv[slot][:nb].view(torch.bfloat16))
                    # the write into out reads the slot asynchronously; the slot is
                    # not free until that read has actually happened
                    done = torch.cuda.Event()
                    done.record()
                    done.synchronize()
                    free.put(slot)
            except BaseException as e:      # noqa: BLE001
                err.append(e)
                for _ in range(DEPTH + self.n):
                    free.put(-1)

        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        th = [threading.Thread(target=producer), threading.Thread(target=consumer)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        el = time.perf_counter() - t0
        if err:
            raise err[0]
        torch.cuda.set_device(0)
        return el

    # ----------------------------------------------------------------- fsync
    def fsync_write(self, compressed: bool, path: Path) -> float:
        sizes = self.cmp_sizes if compressed else self.raw_sizes
        off, total = [], 0
        for s in sizes:
            off.append(total); total += s
        with open(path, "wb") as fh:
            fh.truncate(total)
        host = self.host_flat[:total]
        torch.cuda.set_device(0)
        torch.cuda.synchronize(0)
        t0 = time.perf_counter()
        for i, x in enumerate(self.cache):
            if compressed:
                self.scr[0].zero_()
                f = x.float().contiguous().reshape(-1)
                m = ctypes.c_size_t(0)
                getattr(self.L, _MANGLED[("compress", MODE)])(
                    ctypes.c_void_p(f.data_ptr()),
                    ctypes.c_void_p(self.scr[0].data_ptr()),
                    ctypes.c_size_t(f.numel()), ctypes.byref(m),
                    ctypes.c_float(self.eps[i]), None)
                nb = int(m.value)
                assert nb == sizes[i], (i, nb, sizes[i])
                host[off[i]:off[i] + nb].copy_(self.scr[0][:nb])
            else:
                host[off[i]:off[i] + sizes[i]].copy_(x.reshape(-1).view(torch.uint8))
        torch.cuda.synchronize(0)
        fd = os.open(path, os.O_WRONLY)
        try:
            write_all(fd, memoryview(host.numpy()), 0)
            os.fsync(fd)
        finally:
            os.close(fd)
        return time.perf_counter() - t0

    # ------------------------------------------------------------ verifiers
    def check_moved(self, compressed: bool) -> dict:
        """Every delivered output, fetched after the pass, compared on the host."""
        bad_bytes, worst = 0, 0.0
        for i, x in enumerate(self.cache):
            got = self.out[i].cpu()
            if compressed:
                d = (x.float().reshape(-1).double().cpu() - got.double()).abs().max()
                worst = max(worst, float(d))
                if not bool(torch.isfinite(got).all()):
                    bad_bytes += 1
            elif not torch.equal(got, x.reshape(-1).cpu()):
                bad_bytes += 1
        eps_max = max(self.eps)
        return {"bytes_mismatched": bad_bytes, "max_abs_error": worst,
                "within_eps_plus_round": worst <= eps_max + TAU + 2 ** -8 * worst
                if compressed else True}

    def check_written(self, compressed: bool, path: Path) -> dict:
        sizes = self.cmp_sizes if compressed else self.raw_sizes
        off, total = [], 0
        for s in sizes:
            off.append(total); total += s
        buf = bytearray(total)
        with open(path, "rb", buffering=0) as fh:
            n = fh.readinto(memoryview(buf))
        if n != total:
            return {"short_read": True, "read": n, "expected": total}
        back = torch.frombuffer(buf, dtype=torch.uint8)
        if not compressed:
            bad = sum(1 for i, x in enumerate(self.cache)
                      if not torch.equal(back[off[i]:off[i] + sizes[i]],
                                         x.reshape(-1).view(torch.uint8).cpu()))
            return {"short_read": False, "bytes_mismatched_tensors": bad}
        worst = 0.0
        torch.cuda.set_device(0)
        for i, x in enumerate(self.cache):
            s = sizes[i]
            self.scr[0].zero_()
            self.scr[0][:s].copy_(back[off[i]:off[i] + s].to("cuda:0"))
            d = torch.zeros(x.numel(), dtype=torch.float32, device="cuda:0")
            getattr(self.L, _MANGLED[("decompress", MODE)])(
                ctypes.c_void_p(d.data_ptr()), ctypes.c_void_p(self.scr[0].data_ptr()),
                ctypes.c_size_t(x.numel()), ctypes.c_size_t(s),
                ctypes.c_float(self.eps[i]), None)
            torch.cuda.synchronize(0)
            e = float((x.float().reshape(-1).double().cpu() - d.double().cpu())
                      .abs().max())
            worst = max(worst, e / self.eps[i])
        return {"short_read": False, "max_E32_over_eps": worst,
                "fp32_ok": worst <= 1.0 + TAU}


def paired_ratio(pairs: list[tuple[float, float]], seed: int, n: int = 5000) -> dict:
    """R = exp(mean(log(Tc/Tr))). R < 1 means the compressed path is faster."""
    r = np.array([c / raw for raw, c in pairs])
    lg = np.log(r)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(r))
    boot = np.array([np.exp(lg[rng.choice(idx, len(idx), replace=True)].mean())
                     for _ in range(n)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"R": float(np.exp(lg.mean())), "ci95": [float(lo), float(hi)],
            "latency_change_pct": 100 * (float(np.exp(lg.mean())) - 1),
            "n_pairs": len(r), "compressed_faster": bool(hi < 1.0),
            "undetermined": bool(lo < 1.0 < hi)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--mount", required=True, help="directory for the fsync path")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="/workspace/out/b2")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--boot-seed", type=int, default=20260918)
    a = ap.parse_args()
    if torch.cuda.device_count() < 2:
        raise SystemExit("B2 needs two devices")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mount = Path(a.mount); mount.mkdir(parents=True, exist_ok=True)
    trials = (out / "path_trials.jsonl").open("w")
    checks = (out / "path_checks.jsonl").open("w")

    L = lib()
    per_seg = a.pairs // a.segments
    rng = random.Random(a.seed)
    results: dict = {}

    benches = {}
    for c in a.caches:
        d = Path(c)
        cache = [torch.load(p).to("cuda:0") for p in sorted(d.glob("l*.pt"))]
        benches[d.name] = Bench(cache, L, mount)
        print(f"{d.name}: {len(cache)} tensors, "
              f"{sum(x.numel()*2 for x in cache)/1e6:.1f} MB raw, "
              f"{sum(benches[d.name].cmp_sizes)/1e6:.1f} MB compressed", flush=True)

    paths = ["serial", "pipeline", "fsync"]
    for seg in range(a.segments):
        print(f"\n=== segment {seg + 1}/{a.segments} ===", flush=True)
        for cname, b in benches.items():
            for path in paths:
                key = f"{cname}|{path}"
                results.setdefault(key, [])
                fp = mount / f"b2_{cname}_{path}.bin"

                def run(compressed: bool) -> float:
                    if path == "serial":
                        return b.serial(compressed)
                    if path == "pipeline":
                        return b.pipeline(compressed)
                    return b.fsync_write(compressed, fp)

                for _ in range(a.warmup):
                    run(False); run(True)

                pre = (b.check_written(True, fp) if path == "fsync"
                       else b.check_moved(True))
                checks.write(json.dumps({"segment": seg, "cache": cname, "path": path,
                                         "when": "pre", **pre}) + "\n")

                orders = [True] * (per_seg // 2) + [False] * (per_seg - per_seg // 2)
                rng.shuffle(orders)
                for pi, raw_first in enumerate(orders):
                    if raw_first:
                        tr = run(False); tc = run(True)
                    else:
                        tc = run(True); tr = run(False)
                    results[key].append((tr, tc))
                    trials.write(json.dumps({
                        "segment": seg, "cache": cname, "path": path, "pair": pi,
                        "raw_first": raw_first, "raw_s": tr, "compressed_s": tc,
                        "ratio": tc / tr, "t_wall": time.time()}) + "\n")
                    trials.flush()

                post = (b.check_written(True, fp) if path == "fsync"
                        else b.check_moved(True))
                checks.write(json.dumps({"segment": seg, "cache": cname, "path": path,
                                         "when": "post", **post}) + "\n")
                checks.flush()
                rs = [p[0] for p in results[key]]
                cs = [p[1] for p in results[key]]
                print(f"  {cname:<26} {path:<9} raw {st.median(rs)*1e3:8.2f} ms  "
                      f"comp {st.median(cs)*1e3:8.2f} ms  "
                      f"R {st.median([c/r for r, c in results[key]]):.3f}", flush=True)
                if path == "fsync":
                    fp.unlink(missing_ok=True)

    summary = {}
    for key, pairs in results.items():
        cname, path = key.split("|")
        rs = sorted(p[0] for p in pairs); cs = sorted(p[1] for p in pairs)
        q = lambda v, f: v[int(f * (len(v) - 1))]                      # noqa: E731
        b = benches[cname]
        summary[key] = {
            "cache": cname, "path": path,
            "raw_median_ms": st.median(rs) * 1e3,
            "raw_iqr_ms": [q(rs, .25) * 1e3, q(rs, .75) * 1e3],
            "compressed_median_ms": st.median(cs) * 1e3,
            "compressed_iqr_ms": [q(cs, .25) * 1e3, q(cs, .75) * 1e3],
            "payload_over_raw": sum(b.cmp_sizes) / sum(b.raw_sizes),
            **paired_ratio(pairs, a.boot_seed)}
    (out / "path_summary.json").write_text(json.dumps(summary, indent=2))
    trials.close(); checks.close()

    print(f"\n{'configuration':<38} {'raw ms':>9} {'comp ms':>9} {'R':>7} "
          f"{'95% CI':>18}")
    for k, s in summary.items():
        verdict = ("compressed faster" if s["compressed_faster"]
                   else "undetermined" if s["undetermined"] else "compressed slower")
        print(f"{k:<38} {s['raw_median_ms']:9.2f} {s['compressed_median_ms']:9.2f} "
              f"{s['R']:7.3f} [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]  {verdict}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
