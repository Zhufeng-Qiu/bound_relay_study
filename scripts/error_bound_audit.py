"""Re-measure cuSZp's error bound with the corrected bridge.

The first corpus run reported cuSZp missing its bound on up to a third of tensors.
It was not: three modes produced different compressed bytes and byte-identical
reconstruction errors, which can only happen if the errors came from a stale
buffer. The destination is now poisoned before decoding and a partial decode
returns no number at all.

Run over one full cache — all 28 layers, K and V — because the failures were
perfectly deterministic per layer, concentrated on K at layers 5 and 10, and that
pattern is what has to reappear or not.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from boundrelay.codec.cuszp_bridge import roundtrip

CACHE = Path("/workspace/corpus_v2/fullcache_d00_L2048")
C_GRID, MODES = [0.01, 0.03, 0.10], ["plain", "outlier", "fixed"]

rows = []
for p in sorted(CACHE.glob("l*.pt")):
    layer = int(p.stem[1:3]); kind = p.stem[-1]
    x = torch.load(p)
    sd = float(x.float().std())
    for c in C_GRID:
        eps = c * sd
        for m in MODES:
            r = roundtrip(x, eps, m, want_decode=True, probe_writes=True)
            rows.append({"layer": layer, "kind": kind, "c": c, "mode": m, "std": sd,
                         "eps_abs": eps, **r})

Path("/workspace/error_audit.json").write_text(json.dumps(rows, indent=2))

print(f"{'mode':>9} {'c':>6} {'violations':>12} {'left unwritten':>16} {'worst err/eps':>14}")
print("-" * 62)
for m in MODES:
    for c in C_GRID:
        s = [r for r in rows if r["mode"] == m and r["c"] == c]
        bad = [r for r in s if r["within_eps_fp32"] is False]
        part = [r for r in s if r.get("unwritten_elements", 0) > 0]
        worst = max((r["max_error_fp32"]/r["eps_abs"] for r in s
                     if r["max_error_fp32"] is not None), default=0)
        print(f"{m:>9} {c:>6g} {len(bad):>6}/{len(s):<5} {len(part):>10}/{len(s):<5} {worst:13.3f}x")

print("\nsanity: do the three modes still report identical errors?")
same = 0; tot = 0
for p in sorted({(r["layer"], r["kind"]) for r in rows}):
    for c in C_GRID:
        e = [r["max_error_fp32"] for r in rows
             if (r["layer"], r["kind"]) == p and r["c"] == c]
        tot += 1
        if len(set(e)) == 1: same += 1
print(f"  identical across modes: {same}/{tot}  "
      f"({'still broken' if same > tot*0.5 else 'resolved'})")

k5 = [r for r in rows if r["kind"] == "k" and r["layer"] in (5, 10) and r["c"] == 0.01]
print(f"\nthe layers that failed before (K@5, K@10, c=0.01): "
      f"{sum(1 for r in k5 if r['within_eps_fp32'] is False)}/{len(k5)} now violate")
