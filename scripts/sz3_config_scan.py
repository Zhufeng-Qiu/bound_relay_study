"""A3 — SZ3 configuration scan.

SZ3 was authored by one of the groups this work is being sent to. Reporting a
number produced by whatever `pysz` defaults to would *under*-report their system,
which is worse than over-reporting it. So the SZ3 coordinate in Table B has to
come from a documented scan, not from `INTERP_LORENZO` because it happened to be
the default.

Protocol, fixed before looking at results:

* Same ``eps_i = c * std_i`` and the same bf16 byte denominator as every other
  row. SZ3 reads a lossless fp32 upcast; crediting it the wider type would hand
  it a free 2x.
* Documents, not tensors, are the split unit — tensors from one prompt are not
  independent observations.
* One global configuration per ``c``, chosen on tuning documents only and then
  frozen across K/V, layer and length. The per-tensor best is reported separately
  and labelled an oracle, because picking per tensor is not a deployable policy.

Phase A validates the script, that each configuration actually takes effect, and
the runtime budget. The *final* configuration choice waits for the larger corpus
behind Gate B0 — four documents cannot select a configuration that will hold.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boundrelay.bench.table_a import load_corpus  # noqa: E402

C_GRID = [0.01, 0.03, 0.10]
TIER1 = ["LORENZO_REG", "INTERP_LORENZO", "INTERP"]
SANITY = ["NOPRED"]                      # not a contender; proves the knob moves
TUNING_DOCS = {"p0", "p1"}               # pre-registered
HELDOUT_DOCS = {"p2", "p3"}


def sz3_once(x32: np.ndarray, eps: float, algo: str) -> dict:
    from pysz import sz, szAlgorithm, szConfig, szErrorBoundMode

    cfg = szConfig(x32.size)
    cfg.setDims(*x32.shape)
    cfg.errorBoundMode = szErrorBoundMode.ABS
    cfg.absErrorBound = eps
    cfg.cmprAlgo = getattr(szAlgorithm, algo)

    t0 = time.perf_counter()
    comp, _ = sz.compress(x32, cfg)
    enc = time.perf_counter() - t0
    dec, _ = sz.decompress(comp, x32.dtype, x32.shape)
    return {"bytes": int(comp.nbytes), "encode_s": enc,
            "max_error": float(np.abs(x32 - dec).max())}


def main() -> int:
    tensors = load_corpus(Path("results/private_raw/corpus_kv/kv"))
    out: dict = {"c_grid": C_GRID, "algorithms": TIER1 + SANITY,
                 "tuning_docs": sorted(TUNING_DOCS), "heldout_docs": sorted(HELDOUT_DOCS),
                 "note": "Phase A: script + knob validation only. Final configuration "
                         "selection waits for the Gate B0 corpus.",
                 "rows": {}}

    for name, x in tensors.items():
        doc = name.split("_")[0]
        x32 = x.float().numpy()
        sd = float(x.float().std())
        bf16_bytes = x.numel() * 2
        for c in C_GRID:
            eps = c * sd
            for algo in TIER1 + SANITY:
                r = sz3_once(x32, eps, algo)
                r["ratio"] = bf16_bytes / r["bytes"]
                r["within_bound"] = r["max_error"] <= eps * 1.001
                out["rows"][f"{name}|c{c:g}|{algo}"] = {"doc": doc, "c": c, "algo": algo, **r}

    Path("results/public/d14_table_b").mkdir(parents=True, exist_ok=True)
    Path("results/public/d14_table_b/sz3_config_scan.json").write_text(json.dumps(out, indent=2))

    def mean_ratio(algo, c, docs):
        v = [r["ratio"] for r in out["rows"].values()
             if r["algo"] == algo and r["c"] == c and r["doc"] in docs]
        return sum(v) / len(v)

    print(f"{'algorithm':>16} " + "".join(f"{'c='+f'{c:g}':>22}" for c in C_GRID))
    print(f"{'':>16} " + "".join(f"{'tuning':>11}{'held-out':>11}" for _ in C_GRID))
    print("-" * (17 + 22 * len(C_GRID)))
    for algo in TIER1 + SANITY:
        cells = ""
        for c in C_GRID:
            cells += f"{mean_ratio(algo,c,TUNING_DOCS):10.2f}x{mean_ratio(algo,c,HELDOUT_DOCS):10.2f}x"
        print(f"{algo:>16} {cells}")

    print("\nglobal choice per c, selected on tuning documents only:")
    for c in C_GRID:
        best = max(TIER1, key=lambda a: mean_ratio(a, c, TUNING_DOCS))
        dflt = mean_ratio("INTERP_LORENZO", c, HELDOUT_DOCS)
        got = mean_ratio(best, c, HELDOUT_DOCS)
        print(f"  c={c:<5g} -> {best:<16} held-out {got:5.2f}x   "
              f"(pysz default INTERP_LORENZO: {dflt:5.2f}x, {100*(got/dflt-1):+.1f}%)")

    # oracle: best algorithm per tensor -- reported, never used as the coordinate
    for c in C_GRID:
        orc = []
        for name in tensors:
            orc.append(max(out["rows"][f"{name}|c{c:g}|{a}"]["ratio"] for a in TIER1))
        print(f"  c={c:<5g}    per-tensor oracle {sum(orc)/len(orc):5.2f}x  <- not deployable")

    bad = [k for k, r in out["rows"].items() if not r["within_bound"]]
    print(f"\nbound violations: {len(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
