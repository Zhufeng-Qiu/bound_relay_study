"""The seven-stage pipeline Gate 2 stood in for, and the raw path through the same
machinery.

Gate 2 overlapped encode on GPU0 against decode on GPU1. That is two stages of
seven, and the conclusion drawn from it -- that pipelining could not flip the
decision -- did not follow: a pipeline's steady-state cost is its busiest resource,
not the total minus the smaller of a chosen pair. Grouped by resource the measured
stages give GPU0 21.43 ms, GPU1 17.28 ms, link 5.01 ms, so a free-overlap pipeline
would land at 21.43 ms against a serial raw path of 23.64 ms -- a win.

Whether it actually does is what this measures. Predictions and falsification
conditions are fixed in `results/public/pipeline_prereg.md`, written first.

Three things this harness does that the inference could not:

* **It pipelines the raw path too.** ``raw_iter`` in the closing session copies each
  tensor down and then up on the default stream, so quoting a pipelined compressed
  number against it would compare a pipeline to a serial loop. Identical ring,
  identical threads, codec removed.
* **It sweeps depth, and depth 1 is the control.** A depth-1 pipeline is a serial
  loop wearing extra bookkeeping. If it does not reproduce 50.48 / 23.64 ms, the
  harness is measuring something other than what it claims and nothing else it
  prints may be quoted.
* **It asks first whether the overlap is even available.** cuSZp's entry points
  call ``cudaMalloc`` and ``cudaFree`` around every kernel, and ``cudaFree``
  synchronises the device. If that is what happens here, no transfer on GPU0 can
  overlap the encode before it, and the pipeline is bounded 4.3 ms higher. The
  contention probe settles that without needing the pipeline at all.

The transport is an interface rather than a memcpy, because the break-even the
pipeline implies -- 10.96 GB/s -- sits far below the 19.87 GB/s host link, so the
interesting measurement is the same pipeline over a slower path. ``/workspace`` on
these pods is a MooseFS mount, and offloading a cache to storage is one of the
three scenarios this project is about, not a stand-in for one.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import queue
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


# --------------------------------------------------------------------------- #
# transports -- the only thing that differs between the host link and storage
# --------------------------------------------------------------------------- #

class HostRing:
    """Pinned host slots. The transport the rest of this project measured."""

    name = "host_pinned"

    def __init__(self, depth: int, slot_bytes: int, _spool: Path | None = None):
        self.slots = [torch.empty(slot_bytes, dtype=torch.uint8,
                                  device="cpu", pin_memory=True) for _ in range(depth)]

    def put(self, slot: int, src: torch.Tensor, nbytes: int) -> None:
        self.slots[slot][:nbytes].copy_(src[:nbytes], non_blocking=True)

    def get(self, slot: int, dst: torch.Tensor, nbytes: int) -> None:
        dst[:nbytes].copy_(self.slots[slot][:nbytes], non_blocking=True)

    def close(self) -> None:
        pass


class FileSpool:
    """Same ring, with a file system between the two halves.

    The staging tensor is still pinned -- the device cannot DMA to a page cache --
    so this measures the *added* cost of the storage hop honestly rather than
    charging the transport for an unpinned copy it would not make.
    """

    def __init__(self, depth: int, slot_bytes: int, spool: Path):
        self.slots = [torch.empty(slot_bytes, dtype=torch.uint8,
                                  device="cpu", pin_memory=True) for _ in range(depth)]
        self.dir = spool
        self.dir.mkdir(parents=True, exist_ok=True)
        self.paths = [self.dir / f"slot{i:02d}.bin" for i in range(depth)]
        self.name = f"file:{spool}"

    def put(self, slot: int, src: torch.Tensor, nbytes: int) -> None:
        self.slots[slot][:nbytes].copy_(src[:nbytes], non_blocking=True)
        torch.cuda.current_stream().synchronize()      # bytes must exist to be written
        with open(self.paths[slot], "wb", buffering=0) as f:
            f.write(memoryview(self.slots[slot][:nbytes].numpy()))

    def get(self, slot: int, dst: torch.Tensor, nbytes: int) -> None:
        with open(self.paths[slot], "rb", buffering=0) as f:
            n = f.readinto(memoryview(self.slots[slot][:nbytes].numpy()))
        assert n == nbytes, (n, nbytes)
        dst[:nbytes].copy_(self.slots[slot][:nbytes], non_blocking=True)

    def close(self) -> None:
        for p in self.paths:
            p.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# the contention probe -- does a transfer overlap the codec on one device?
# --------------------------------------------------------------------------- #

def contention_probe(x, L, iters: int = 20) -> dict:
    """Encode alone, D2H alone, and both issued together on one device.

    Under free overlap the third is the max of the first two. If cuSZp's internal
    ``cudaFree`` synchronises the device, it is their sum. This decides between the
    two registered hypotheses without running the pipeline.
    """
    torch.cuda.set_device(0)
    f = x.to("cuda:0").float().contiguous().reshape(-1)
    n = f.numel()
    eps = C * float(f.std())
    scr = torch.zeros(n * 4 + 4096, dtype=torch.uint8, device="cuda:0")
    host = torch.empty(n * 4 + 4096, dtype=torch.uint8, device="cpu", pin_memory=True)
    other = torch.empty(n * 2, dtype=torch.uint8, device="cuda:0")
    s_copy = torch.cuda.Stream(device=0)

    def enc():
        m = ctypes.c_size_t(0)
        getattr(L, _MANGLED[("compress", MODE)])(
            ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(scr.data_ptr()),
            ctypes.c_size_t(n), ctypes.byref(m), ctypes.c_float(eps), None)

    def d2h():
        with torch.cuda.stream(s_copy):
            host[: other.numel()].copy_(other, non_blocking=True)

    def timed(fn, k=iters):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(0)
        t = time.perf_counter()
        for _ in range(k):
            fn()
        torch.cuda.synchronize(0)
        return (time.perf_counter() - t) / k

    def both():
        d2h()
        enc()

    a, b, c = timed(enc), timed(d2h), timed(both)
    return {"encode_s": a, "d2h_s": b, "together_s": c,
            "sum_s": a + b, "max_s": max(a, b),
            "overlap": (a + b - c) / min(a, b) if min(a, b) > 0 else None,
            "verdict": "transfer overlaps codec" if c < 0.85 * (a + b)
                       else "device serialised by the codec's own allocations"}


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

def run_pipeline(cache, L, depth: int, compressed: bool, transport_cls,
                 spool: Path | None, iters: int = 15) -> dict:
    """Producer on GPU0, consumer on GPU1, `depth` slots between them.

    Two host threads rather than streams alone: cuSZp's compress writes the
    compressed size back to a host pointer, so it blocks its caller no matter which
    stream it was issued on. Gate 2 measured that directly -- 1.4% overlap from one
    thread against 53.7% from two.

    Two and not three, which has a consequence that depends on which hypothesis
    holds, so it is worth writing down before the run rather than after. One consumer
    thread runs H2D and decode in sequence, so those cannot overlap each other even
    though they use different resources, and the consumer side is charged
    5.01 + 0.93 + 15.01 + 1.34 = 22.29 ms rather than max(5.01, 17.28) = 17.28.

    * If the codec serialises its own device (H2), the producer is
      1.44 + 19.99 + 4.32 = 25.75 ms and binds. Two threads are enough and the
      measurement is tight.
    * If transfers overlap the codec freely (H1), the producer is max(21.43, 4.32)
      = 21.43 ms and the **consumer** binds at 22.29 ms. Two threads then overstate
      the pipeline's cost by about 4%, and the true bound is the 21.43 the
      resource grouping predicts.

    So the per-thread busy times are not decoration. If the consumer comes out
    ahead, this harness has understated the pipeline, the reported number is an
    upper bound, and a third thread is needed before it can be quoted as the
    pipeline's cost rather than as this harness's cost.

    Slot ownership runs the whole chain: scratch on GPU0, the host ring entry, the
    receive buffer and the decode destination on GPU1 all carry the same index, and
    a slot returns to the free pool only after its downcast. Nothing is reused while
    a stage still holds it, which is the failure that produced `DEFECTS.md` entry D2.
    """
    n = len(cache)
    eps = [C * float(x.float().std()) for x in cache]
    slot_bytes = max(x.numel() * 4 + 4096 for x in cache)

    torch.cuda.set_device(0)
    ring = transport_cls(depth, slot_bytes, spool)
    scr = [torch.zeros(slot_bytes, dtype=torch.uint8, device="cuda:0") for _ in range(depth)]
    torch.cuda.set_device(1)
    recv = [torch.zeros(slot_bytes, dtype=torch.uint8, device="cuda:1") for _ in range(depth)]
    dst = [torch.zeros(max(x.numel() for x in cache), dtype=torch.float32,
                       device="cuda:1") for _ in range(depth)]
    out: list[torch.Tensor | None] = [None] * n

    def one_pass(keep: bool) -> tuple[float, dict]:
        free: queue.Queue = queue.Queue()
        for s in range(depth):
            free.put(s)
        ready: queue.Queue = queue.Queue()
        err: list[BaseException] = []

        busy = {"producer_s": 0.0, "consumer_s": 0.0}

        def producer():
            try:
                torch.cuda.set_device(0)
                t_busy = time.perf_counter()
                s_copy = torch.cuda.Stream(device=0)
                for i, x in enumerate(cache):
                    slot = free.get()
                    if slot < 0:            # consumer died and unblocked us
                        return
                    if compressed:
                        f = x.float().contiguous().reshape(-1)
                        m = ctypes.c_size_t(0)
                        getattr(L, _MANGLED[("compress", MODE)])(
                            ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(scr[slot].data_ptr()),
                            ctypes.c_size_t(f.numel()), ctypes.byref(m),
                            ctypes.c_float(eps[i]), None)
                        nb = int(m.value)
                        src = scr[slot]
                    else:
                        nb = x.numel() * 2
                        src = x.reshape(-1).view(torch.uint8)
                    with torch.cuda.stream(s_copy):
                        ring.put(slot, src, nb)
                        ev = torch.cuda.Event()
                        ev.record(s_copy)
                    ready.put((i, slot, nb, ev))
                busy["producer_s"] = time.perf_counter() - t_busy
                ready.put(None)
            except BaseException as e:      # noqa: BLE001 - re-raised on the main thread
                err.append(e)
                ready.put(None)

        def consumer():
            try:
                torch.cuda.set_device(1)
                s_copy = torch.cuda.Stream(device=1)
                t_busy = time.perf_counter()
                while True:
                    item = ready.get()
                    if item is None:
                        busy["consumer_s"] = time.perf_counter() - t_busy
                        return
                    i, slot, nb, ev = item
                    ev.synchronize()                    # the bytes are in the ring
                    with torch.cuda.stream(s_copy):
                        ring.get(slot, recv[slot], nb)
                    s_copy.synchronize()
                    if compressed:
                        dst[slot].zero_()
                        getattr(L, _MANGLED[("decompress", MODE)])(
                            ctypes.c_void_p(dst[slot].data_ptr()),
                            ctypes.c_void_p(recv[slot].data_ptr()),
                            ctypes.c_size_t(cache[i].numel()), ctypes.c_size_t(nb),
                            ctypes.c_float(eps[i]), None)
                        fin = dst[slot][: cache[i].numel()].to(torch.bfloat16)
                    else:
                        fin = recv[slot][:nb].view(torch.bfloat16).clone()
                    if keep:
                        out[i] = fin.cpu()
                    free.put(slot)
            except BaseException as e:      # noqa: BLE001
                err.append(e)
                # the producer blocks on free.get(); without this it hangs on a
                # billing GPU until something kills it
                for _ in range(depth + n):
                    free.put(-1)

        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        th = [threading.Thread(target=producer), threading.Thread(target=consumer)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        el = time.perf_counter() - t0
        if err:
            raise err[0]
        return el, dict(busy)

    one_pass(False)                               # warm
    runs = [one_pass(False) for _ in range(iters)]
    samples = [r[0] for r in runs]
    busy = {k: st.median([r[1][k] for r in runs]) for k in ("producer_s", "consumer_s")}
    one_pass(True)                                # the pass that is checked
    ring.close()

    # compared on the host. The source lives on GPU0 and the reconstruction came off
    # GPU1, and a direct device-to-device fetch on this infrastructure returns zeros
    # while reporting success -- see `peer_copy_findings.md`. That is how the closing
    # session's gate came to report `inf` and blame the codec.
    worst = 0.0
    for i, x in enumerate(cache):
        assert out[i] is not None
        worst = max(worst, float((x.float().cpu().reshape(-1)
                                  - out[i].float().reshape(-1)).abs().max()))
    return {"depth": depth, "compressed": compressed, "transport": ring.name,
            "median_s": st.median(samples), "min_s": min(samples), "max_s": max(samples),
            "samples_s": samples, "reconstruction_max_error": worst, **busy,
            "bound_by": "producer (GPU0)" if busy["producer_s"] > busy["consumer_s"]
                        else "consumer (GPU1)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--spool", type=str, default=None,
                    help="run the file transport with slots in this directory")
    ap.add_argument("--out", type=str, default="/workspace/out/pipeline_multistage.json")
    a = ap.parse_args()

    if torch.cuda.device_count() < 2:
        raise SystemExit("a two-device pipeline needs two devices")
    L = lib()
    # resident on GPU0, exactly as the serial baseline has it. Loading from the
    # host inside the producer would charge the pipeline a CPU->GPU0 transfer that
    # `raw_iter` never made, and the whole point is a like-for-like comparison.
    cache = [torch.load(p).to("cuda:0") for p in sorted(CACHE.glob("l*.pt"))]
    raw_bytes = sum(x.numel() * 2 for x in cache)
    print(f"{len(cache)} tensors, {raw_bytes/1e6:.1f} MB bf16", flush=True)
    print("depth 1 is the control. Note it is *tensor-major* serial -- encode, ship, "
          "decode, next -- while the reference is *stage-major*: all 56 encodes, then "
          "all 56 transfers. Same total work, different per-call amortisation, so they "
          "should agree closely without that being a tautology.", flush=True)

    res: dict = {"n_tensors": len(cache), "raw_bytes": raw_bytes, "c": C, "mode": MODE,
                 "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
                 "serial_reference": {"compressed_s": 0.05048, "raw_s": 0.023641},
                 "runs": []}

    print("\n=== contention probe: does a D2H overlap an encode on one device? ===", flush=True)
    cp = contention_probe(cache[0], L)
    res["contention"] = cp
    print(f"  encode {cp['encode_s']*1e3:6.3f} ms   d2h {cp['d2h_s']*1e3:6.3f} ms   "
          f"together {cp['together_s']*1e3:6.3f} ms   "
          f"(sum {cp['sum_s']*1e3:.3f}, max {cp['max_s']*1e3:.3f})", flush=True)
    print(f"  -> {cp['verdict']}", flush=True)

    transports: list[tuple[type, Path | None]] = [(HostRing, None)]
    if a.spool:
        transports.append((FileSpool, Path(a.spool)))

    for cls, spool in transports:
        for compressed in (True, False):
            print(f"\n=== {'compressed' if compressed else 'raw':<10} "
                  f"{cls.__name__}{'' if spool is None else ' ' + str(spool)} ===", flush=True)
            for d in a.depths:
                r = run_pipeline(cache, L, d, compressed, cls, spool)
                res["runs"].append(r)
                ref = res["serial_reference"]["compressed_s" if compressed else "raw_s"]
                print(f"  depth {d:<2} {r['median_s']*1e3:8.2f} ms  "
                      f"[{r['min_s']*1e3:.2f}, {r['max_s']*1e3:.2f}]  "
                      f"{raw_bytes/1e6/(r['median_s']*1e3):6.2f} GB/s of cache  "
                      f"vs serial {ref*1e3:.2f} ms ({r['median_s']/ref:.2f}x)  "
                      f"max|err| {r['reconstruction_max_error']:.4g}", flush=True)
                print(f"          producer {r['producer_s']*1e3:7.2f} ms  "
                      f"consumer {r['consumer_s']*1e3:7.2f} ms  -> {r['bound_by']}", flush=True)

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
