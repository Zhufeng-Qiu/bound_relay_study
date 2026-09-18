"""B0 -- does a reused buffer still deliver the right bytes, and what is the error?

Two questions, and the second one has been answered loosely before.

**Reuse.** Every earlier transport result allocated its buffers once and then ran
one cache through them. A real pipeline runs slot after slot of different tensors,
different lengths and different content through the same memory, and cuSZp reads
past `cmpSize` -- so what a slot held last time is part of this time's input unless
someone clears it. This runs four caches through one pool in an order that forces
large->small->large and same-length-different-content, and requires the result to
be **bit-identical** to the same tensor decoded into freshly zeroed buffers.

**Error.** The codec's contract is on fp32. What a receiver holds is bf16, and the
downcast adds rounding on top. Those are different numbers and this records both,
in float64 on the host, because a max-abs difference taken in the precision being
audited lets that precision hide its own error. An earlier audit found 489 of 504
combinations outside the original epsilon *after* the downcast while every one
satisfied the fp32 bound. Neither fact is a licence to state the other.

Nothing here is timed. Correctness first, and conservatively: zero everything,
synchronise explicitly, and only then ask what it costs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/boundrelay")
from boundrelay.codec.cuszp_bridge import _MANGLED, TAU, lib  # noqa: E402

MODE = "fixed"
C_GRID = [0.01, 0.10]


# --------------------------------------------------------------------------- #
# error accounting
# --------------------------------------------------------------------------- #

def errors(x_f32: torch.Tensor, y_f32: torch.Tensor, z_bf16: torch.Tensor,
           eps: float) -> dict:
    """E32, Ebf and R, computed on the host in float64.

    x is the source upcast losslessly to fp32, y the codec's fp32 reconstruction,
    z that reconstruction downcast to bf16 -- what a receiver actually holds.
    """
    x = x_f32.double().cpu()
    y = y_f32.double().cpu()
    z = z_bf16.double().cpu()
    finite = bool(torch.isfinite(y).all() and torch.isfinite(z).all())
    e32 = float((y - x).abs().max())
    ebf = float((z - x).abs().max())
    rnd = float((z - y).abs().max())
    return {
        "eps": eps, "E32": e32, "Ebf": ebf, "R": rnd,
        "E32_over_eps": e32 / eps if eps > 0 else float("inf"),
        "Ebf_over_eps": ebf / eps if eps > 0 else float("inf"),
        "finite": finite,
        # finite first, always: NaN compares false against every bound and would
        # otherwise pass as "not exceeding" it
        "fp32_ok": finite and e32 <= eps + TAU,
        "bf16_within_eps_plus_rounding": finite and ebf <= eps + TAU + rnd,
        "bf16_within_eps_alone": finite and ebf <= eps + TAU,
    }


def eps_for(x: torch.Tensor) -> float:
    """float32(c * std) -- the value actually handed to the C interface."""
    return float(x.float().std(correction=1))


# --------------------------------------------------------------------------- #
# one round trip
# --------------------------------------------------------------------------- #

class Pool:
    """A reusable set of buffers, sized for the largest tensor it will ever see."""

    def __init__(self, max_numel: int, device: str = "cuda:0"):
        self.max_numel = max_numel
        self.cmp = torch.zeros(max_numel * 4 + 4096, dtype=torch.uint8, device=device)
        self.dec = torch.zeros(max_numel, dtype=torch.float32, device=device)
        self.addr = (self.cmp.data_ptr(), self.dec.data_ptr())
        self.uses = 0

    def soil(self, rng: random.Random) -> None:
        """Leave something in the buffers that a correct run must overwrite."""
        self.cmp.fill_(rng.randint(1, 255))
        self.dec.fill_(float("nan"))


def roundtrip(L, x: torch.Tensor, eps: float, pool: Pool) -> tuple[dict, int, torch.Tensor]:
    """Compress and decompress `x` through `pool`, clearing it first.

    The clearing is the production path, not a test artefact: cuSZp reads past
    `cmpSize` and does not write elements it expects to be zero, so a receiver that
    does not arrive zeroed is reading whatever was there before.
    """
    n = x.numel()
    f = x.float().contiguous().reshape(-1)
    pool.cmp.zero_()
    pool.dec.zero_()
    torch.cuda.synchronize()

    m = ctypes.c_size_t(0)
    getattr(L, _MANGLED[("compress", MODE)])(
        ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(pool.cmp.data_ptr()),
        ctypes.c_size_t(n), ctypes.byref(m), ctypes.c_float(eps), None)
    torch.cuda.synchronize()
    nb = int(m.value)

    getattr(L, _MANGLED[("decompress", MODE)])(
        ctypes.c_void_p(pool.dec.data_ptr()), ctypes.c_void_p(pool.cmp.data_ptr()),
        ctypes.c_size_t(n), ctypes.c_size_t(nb), ctypes.c_float(eps), None)
    torch.cuda.synchronize()
    pool.uses += 1

    y = pool.dec[:n]
    z = y.to(torch.bfloat16)
    return errors(f, y, z, eps), nb, z.clone()


# --------------------------------------------------------------------------- #
# B0.1  fresh baselines
# --------------------------------------------------------------------------- #

def fresh_baselines(L, caches: dict, rec) -> dict:
    """Every tensor, every c, through buffers allocated and zeroed for it alone."""
    ref: dict = {}
    n_fail = 0
    for cname, cache in caches.items():
        for ti, (tname, x) in enumerate(cache):
            sd = eps_for(x)
            for c in C_GRID:
                eps = float(torch.tensor(c * sd, dtype=torch.float32))
                pool = Pool(x.numel())              # fresh every time, by design
                e, nb, z = roundtrip(L, x, eps, pool)
                ref[(cname, tname, c)] = z
                row = {"kind": "fresh", "cache": cname, "tensor": tname, "c": c,
                       "std": sd, "cmp_bytes": nb, "raw_bytes": x.numel() * 2, **e}
                rec(row)
                if not e["fp32_ok"]:
                    n_fail += 1
                del pool
    return {"n_combinations": len(ref), "fp32_failures": n_fail, "_ref": ref}


# --------------------------------------------------------------------------- #
# B0.2  the same pool, over and over
# --------------------------------------------------------------------------- #

def reuse_passes(L, caches: dict, ref: dict, rec, passes_per_c: int) -> dict:
    """Run the four caches through one pool in an order that forces the hard cases.

    The order alternates lengths so that a large payload is followed by a small one
    and then large again -- the case where a slot's tail still holds the previous
    tensor -- and puts two same-length different-content caches next to each other.
    """
    names = list(caches)
    big = [n for n in names if n.endswith("2048")]
    small = [n for n in names if n.endswith("1024")]
    order = []
    while len(order) < passes_per_c:
        order += [big[0], small[0], big[1], small[1], big[0], big[1],
                  small[0], small[1]]
    order = order[:passes_per_c]

    rng = random.Random(20260917)
    max_numel = max(x.numel() for c in caches.values() for _, x in c)
    pool = Pool(max_numel)
    summary = {"pool_addresses": list(pool.addr), "order": {}, "mismatches": 0,
               "fp32_failures": 0, "nonfinite": 0, "reuse_roundtrips": 0}

    for c in C_GRID:
        seen_counts: dict[str, int] = {}
        for p, cname in enumerate(order):
            seen_counts[cname] = seen_counts.get(cname, 0) + 1
            if p % 3 == 2:
                pool.soil(rng)          # production zeroing must survive this
            for tname, x in caches[cname]:
                eps = float(torch.tensor(c * eps_for(x), dtype=torch.float32))
                e, nb, z = roundtrip(L, x, eps, pool)
                summary["reuse_roundtrips"] += 1
                same = bool(torch.equal(z, ref[(cname, tname, c)]))
                if not same:
                    summary["mismatches"] += 1
                if not e["fp32_ok"]:
                    summary["fp32_failures"] += 1
                if not e["finite"]:
                    summary["nonfinite"] += 1
                if not same or not e["fp32_ok"]:
                    rec({"kind": "reuse_failure", "cache": cname, "tensor": tname,
                         "c": c, "pass": p, "bitwise_equal_to_fresh": same,
                         "cmp_bytes": nb, **e})
            if p == len(order) - 1:
                summary["order"][f"c{c:g}"] = dict(seen_counts)
    summary["pool_uses"] = pool.uses
    summary["addresses_stable"] = list(pool.addr) == summary["pool_addresses"]
    return summary


# --------------------------------------------------------------------------- #
# B0.3  does the checker actually reject a bad answer?
# --------------------------------------------------------------------------- #

def negative_tests(L, caches: dict, rec) -> dict:
    """Corrupt a good reconstruction six ways and require every one to be caught.

    This tests the *validator*, not the codec. It does not ask cuSZp to survive a
    damaged payload -- it asks whether this harness would notice if something else
    had. The old aggregate took `max(worst, err)` without checking finiteness first,
    which a NaN slips straight through, so these are not hypothetical.
    """
    cname, cache = next(iter(caches.items()))
    tname, x = cache[0]
    eps = float(torch.tensor(0.10 * eps_for(x), dtype=torch.float32))
    pool = Pool(x.numel())
    good, _, z_good = roundtrip(L, x, eps, pool)
    assert good["fp32_ok"], "baseline for the negative tests is itself failing"
    f = x.float().contiguous().reshape(-1)

    cases = []

    def check(label: str, y: torch.Tensor):
        z = y.to(torch.bfloat16)
        e = errors(f, y, z, eps)
        caught = (not e["finite"]) or (not e["fp32_ok"]) or \
                 (not bool(torch.equal(z, z_good)))
        cases.append({"case": label, "detected": caught,
                      "finite": e["finite"], "fp32_ok": e["fp32_ok"],
                      "E32_over_eps": e["E32_over_eps"]})
        rec({"kind": "negative", "cache": cname, "tensor": tname, **cases[-1]})

    y = pool.dec[:x.numel()]
    for label, mut in (
        ("nan", lambda t: t.index_fill_(0, torch.tensor([7], device=t.device),
                                        float("nan"))),
        ("pos_inf", lambda t: t.index_fill_(0, torch.tensor([9], device=t.device),
                                            float("inf"))),
        ("neg_inf", lambda t: t.index_fill_(0, torch.tensor([11], device=t.device),
                                            float("-inf"))),
        ("beyond_eps", lambda t: t.index_copy_(
            0, torch.tensor([13], device=t.device),
            (t[13] + 4 * eps + 1e-3).reshape(1))),
    ):
        t = y.clone()
        mut(t)
        check(label, t)

    # a single flipped bit in the delivered bf16, which no error bound will notice
    z = z_good.clone()
    bits = z.view(torch.int16)
    bits[17] ^= 1
    e = errors(f, y, z, eps)
    caught = not bool(torch.equal(z, z_good))
    cases.append({"case": "single_bit_flip_in_output", "detected": caught,
                  "finite": e["finite"], "fp32_ok": e["fp32_ok"],
                  "note": "bound checks cannot see this; the bitwise comparison must"})
    rec({"kind": "negative", "cache": cname, "tensor": tname, **cases[-1]})

    # a truncated payload: decode fewer bytes than were written
    pool.dec.zero_()
    m = ctypes.c_size_t(0)
    pool.cmp.zero_()
    getattr(L, _MANGLED[("compress", MODE)])(
        ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(pool.cmp.data_ptr()),
        ctypes.c_size_t(x.numel()), ctypes.byref(m), ctypes.c_float(eps), None)
    torch.cuda.synchronize()
    trunc = max(int(m.value) // 2, 64)
    getattr(L, _MANGLED[("decompress", MODE)])(
        ctypes.c_void_p(pool.dec.data_ptr()), ctypes.c_void_p(pool.cmp.data_ptr()),
        ctypes.c_size_t(x.numel()), ctypes.c_size_t(trunc), ctypes.c_float(eps), None)
    torch.cuda.synchronize()
    check("truncated_payload", pool.dec[:x.numel()].clone())

    return {"cases": cases, "all_detected": all(c["detected"] for c in cases)}


# --------------------------------------------------------------------------- #

def load_cache(d: Path) -> list[tuple[str, torch.Tensor]]:
    return [(p.stem, torch.load(p).to("cuda:0")) for p in sorted(d.glob("l*.pt"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True,
                    help="directories, each a full 28-layer cache")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--out", default="/workspace/out/b0")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = (out / "b0_tensor_checks.jsonl").open("w")

    def rec(row: dict) -> None:
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()

    L = lib()
    caches = {}
    for c in a.caches:
        d = Path(c)
        caches[d.name] = load_cache(d)
        print(f"{d.name}: {len(caches[d.name])} tensors, "
              f"{sum(x.numel()*2 for _, x in caches[d.name])/1e6:.1f} MB", flush=True)

    t0 = time.perf_counter()
    print("\n[1/3] fresh baselines ...", flush=True)
    fresh = fresh_baselines(L, caches, rec)
    ref = fresh.pop("_ref")
    print(f"  {fresh['n_combinations']} combinations, "
          f"{fresh['fp32_failures']} fp32 failures  ({time.perf_counter()-t0:.0f}s)",
          flush=True)

    print(f"\n[2/3] reuse, {a.passes} passes per c through one pool ...", flush=True)
    reuse = reuse_passes(L, caches, ref, rec, a.passes)
    print(f"  {reuse['reuse_roundtrips']} round trips, "
          f"{reuse['mismatches']} bitwise mismatches, "
          f"{reuse['fp32_failures']} fp32 failures, "
          f"{reuse['nonfinite']} non-finite  ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("\n[3/3] negative tests ...", flush=True)
    neg = negative_tests(L, caches, rec)
    for c in neg["cases"]:
        print(f"  {c['case']:<28} {'detected' if c['detected'] else 'MISSED'}",
              flush=True)

    passed = (fresh["fp32_failures"] == 0 and reuse["mismatches"] == 0
              and reuse["fp32_failures"] == 0 and reuse["nonfinite"] == 0
              and neg["all_detected"])
    summary = {"fresh": fresh, "reuse": reuse, "negative": neg,
               "caches": {k: len(v) for k, v in caches.items()},
               "passes_per_c": a.passes, "c_grid": C_GRID, "tau": TAU,
               "elapsed_s": time.perf_counter() - t0,
               "gpu": torch.cuda.get_device_name(0), "B0_PASSED": passed}
    (out / "b0_summary.json").write_text(json.dumps(summary, indent=2))
    jsonl.close()
    print(f"\n{'B0 PASSED' if passed else 'B0 FAILED'}  -> {out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
