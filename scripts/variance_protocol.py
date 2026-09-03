"""B2 — separate cold-start, within-process and between-process variance.

cuSZp's `fixed` mode showed 3-6x spread in encode time across separate processes
at tight error bounds. Before that can be called "process-to-process variability"
it has to be shown not to be an artefact of initialisation, asynchronous timing,
or thermal state -- three explanations that would each produce the same-looking
number.

Protocol: 6 stratified tensors, N fresh processes each running 20 warmup + 100
measured iterations with pre-allocated buffers, mode and eps order randomised
within a process, and the first iteration after context creation recorded
separately as cold start.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C_GRID = [0.01, 0.10]
MODES = ["fixed", "plain"]


def worker(cache: Path, seed: int) -> dict:
    import torch
    from boundrelay.codec.cuszp_bridge import roundtrip

    random.seed(seed)
    picks = [("l00_k", 0), ("l14_k", 14), ("l27_k", 27),
             ("l00_v", 0), ("l14_v", 14), ("l27_v", 27)]
    out: dict = {"pid": os.getpid(), "cells": {}}

    tensors = {n: torch.load(cache / f"{n}.pt") for n, _ in picks}
    combos = [(n, c, m) for n, _ in picks for c in C_GRID for m in MODES]
    random.shuffle(combos)

    first = True
    for name, c, mode in combos:
        x = tensors[name]
        eps = c * float(x.float().std())
        t0 = time.perf_counter()
        roundtrip(x, eps, mode, want_decode=False)
        cold = time.perf_counter() - t0                      # first call in this process
        for _ in range(20):
            roundtrip(x, eps, mode, want_decode=False)
        ts = []
        for _ in range(100):
            t = time.perf_counter()
            roundtrip(x, eps, mode, want_decode=False)
            ts.append(time.perf_counter() - t)
        key = f"{name}|c{c:g}|{mode}"
        out["cells"][key] = {
            "cold_s": cold if first else None,
            "median_s": st.median(ts), "p95_s": sorted(ts)[94],
            "within_cv": st.pstdev(ts) / st.mean(ts),
            "n": len(ts),
        }
        first = False
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("/workspace/corpus_v2/fullcache_d00_L2048"))
    ap.add_argument("--processes", type=int, default=10)
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--out", type=Path, default=Path("/workspace/variance.json"))
    a = ap.parse_args()

    if a.worker >= 0:
        print(json.dumps(worker(a.cache, a.worker)))
        return 0

    runs = []
    for i in range(a.processes):
        r = subprocess.run([sys.executable, __file__, "--worker", str(i),
                            "--cache", str(a.cache)],
                           capture_output=True, text=True,
                           cwd=str(Path(__file__).resolve().parents[1]))
        if r.returncode:
            print(r.stderr[-600:], file=sys.stderr); return 1
        runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
        print(f"  process {i+1}/{a.processes} done", flush=True)

    keys = sorted(runs[0]["cells"])
    summary = {}
    for k in keys:
        meds = [r["cells"][k]["median_s"] for r in runs]
        summary[k] = {
            "between_process_median_s": st.median(meds),
            "between_process_cv": st.pstdev(meds) / st.mean(meds),
            "between_process_spread": max(meds) / min(meds),
            "within_process_cv_median": st.median(r["cells"][k]["within_cv"] for r in runs),
            "p95_over_median": st.median(r["cells"][k]["p95_s"] / r["cells"][k]["median_s"]
                                         for r in runs),
        }
    colds = [c["cold_s"] for r in runs for c in r["cells"].values() if c["cold_s"]]
    warm = st.median(v["between_process_median_s"] for v in summary.values())
    a.out.write_text(json.dumps({"runs": runs, "summary": summary,
                                 "cold_start_median_s": st.median(colds),
                                 "cold_over_warm": st.median(colds) / warm}, indent=2))

    print(f"\ncold start median {st.median(colds)*1e3:.2f} ms vs warm median "
          f"{warm*1e3:.3f} ms  ({st.median(colds)/warm:.1f}x)\n")
    print(f"{'cell':>22} {'median ms':>10} {'within CV':>10} {'between CV':>11} {'spread':>8} {'P95/med':>8}")
    print("-"*76)
    for k in keys:
        s = summary[k]
        print(f"{k:>22} {s['between_process_median_s']*1e3:9.3f} "
              f"{s['within_process_cv_median']:10.3f} {s['between_process_cv']:11.3f} "
              f"{s['between_process_spread']:7.2f}x {s['p95_over_median']:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
