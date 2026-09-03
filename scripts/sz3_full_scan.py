"""A3b — SZ3 configuration scan, over the whole configuration space pysz reaches.

The A3 pilot left two liabilities on the table, and both of them attack the same
claim: that on KV cache SZ3's *predictor* costs compression ratio, and `NOPRED`
— quantise, Huffman, zstd, no prediction — beats every predictor SZ3 offers.

1. The pilot scanned three `cmprAlgo` values. SZ3's predictor is configured by a
   good deal more than that, and all of it sits behind ``Config::loadcfg``, which
   the pilot never called. "You did not tune SZ3" is the first objection this
   result invites, and it was a fair one.
2. The pilot ran on the *superseded* corpus: a greedy-decode cache, 81 tokens,
   from an unpinned model revision. Every other claim in this project was moved
   to the Gate B0 capture (pinned `b9352fbb`, single prefill); this one was not.

So this scan does two things the pilot could not.

**It gives prediction the oracle and keeps NOPRED frozen.** For each tensor,
prediction is credited with the best result over the entire reachable grid —
algorithm, interpolation kernel, interpolation direction, Lorenzo and regression
flags — chosen per tensor, which is not a deployable policy and is not meant to
be. `NOPRED` gets one configuration, its default, everywhere. If prediction still
loses under that handicap, no amount of tuning rescues it and no train/test split
is needed to say so: an oracle cannot overfit in the direction that would flatter
the conclusion.

**It runs on Gate B0 data.** The d00 full cache is 28 layers x K/V at 2048
tokens, captured by the pinned prefill. One document, so document-level held-out
is impossible here — which is exactly why the comparison is built to need no
split. The pilot corpus is still scanned, labelled as the superseded capture, and
reported separately.

Reachability is not assumed. ``validate_knobs`` compresses one fixed field under
each INI key and records whether the output bytes move. Keys that do not move the
output are reported as unreachable rather than silently scanned: a grid axis that
does nothing would otherwise pad the oracle's config count while adding no search.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C_GRID = [0.01, 0.03, 0.10]
VIEWS = ("native_HSD", "token_major_TD", "token_contiguous_HDS")
BOUND_SLACK = 1.001          # SZ3 quantises at eps; allow rounding, nothing more


# --------------------------------------------------------------------------- #
# configuration space
# --------------------------------------------------------------------------- #

def ini_text(algo: str, algo_settings: dict, eps: float) -> str:
    body = "".join(f"{k} = {v}\n" for k, v in algo_settings.items())
    return (f"[GlobalSettings]\nCmprAlgo = ALGO_{algo}\nErrorBoundMode = ABS\n"
            f"AbsErrorBound = {eps!r}\n[AlgoSettings]\n{body}")


def sz3_run(x: np.ndarray, eps: float, algo: str, algo_settings: dict) -> dict:
    """One compress/decompress. dims are set before loadcfg; both orders were
    checked to give identical bytes, so the INI is the single source of truth."""
    from pysz import sz, szConfig

    cfg = szConfig(x.size)
    cfg.setDims(*x.shape)
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as f:
        f.write(ini_text(algo, algo_settings, eps))
        path = f.name
    try:
        cfg.loadcfg(path)
    finally:
        os.unlink(path)

    t0 = time.perf_counter()
    comp, _ = sz.compress(x, cfg)
    enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    dec, _ = sz.decompress(comp, x.dtype, x.shape)
    dcd = time.perf_counter() - t0
    err = float(np.abs(x - dec).max())
    if err > eps * BOUND_SLACK:
        raise SystemExit(f"SZ3 broke its bound: {algo} {algo_settings} "
                         f"err={err:g} eps={eps:g}")
    return {"bytes": int(comp.nbytes), "encode_s": enc, "decode_s": dcd,
            "max_error": err}


def grid(ndim: int) -> list[tuple[str, dict]]:
    """Every reachable predictor configuration, plus NOPRED.

    ``InterpolationDirection`` indexes a permutation of the axes, so its range is
    ndim! -- 6 in the native [H,S,D] view, 2 in the token-major [T,D] one. Feeding
    a 2-D tensor six directions would count four duplicate configurations into the
    oracle's search and make the handicap look larger than it is.
    """
    n_dir = {1: 1, 2: 2, 3: 6}.get(ndim, 6)
    out: list[tuple[str, dict]] = [("NOPRED", {})]
    for lz, rg in (("Yes", "Yes"), ("Yes", "No"), ("No", "Yes")):
        out.append(("LORENZO_REG", {"Lorenzo": lz, "Regression": rg}))
    for algo, kern, d in itertools.product(
            ("INTERP", "INTERP_LORENZO"),
            ("INTERP_ALGO_LINEAR", "INTERP_ALGO_CUBIC"),
            range(n_dir)):
        out.append((algo, {"InterpolationAlgo": kern, "InterpolationDirection": d}))
    return out


def tag(algo: str, s: dict) -> str:
    return algo if not s else algo + "|" + ",".join(f"{k}={v}" for k, v in s.items())


# --------------------------------------------------------------------------- #
# knob reachability
# --------------------------------------------------------------------------- #

def validate_knobs() -> dict:
    """Does each INI key move the output at all? Recorded, not assumed.

    A smooth field is used deliberately: predictor settings are what it rewards,
    so a key that fails to move the bytes *here* is not reachable, rather than
    merely irrelevant to this data.
    """
    g = np.mgrid[0:16, 0:64, 0:64].astype(np.float32)
    x = np.ascontiguousarray(np.sin(g[0] / 3) * np.cos(g[1] / 7) + 0.3 * g[2] / 64)
    eps = 0.01 * float(x.std())

    probes = {
        "BlockSize":            ("LORENZO_REG", "BlockSize", [2, 4, 6, 8, 16, 32]),
        "Lorenzo":              ("LORENZO_REG", "Lorenzo", ["Yes", "No"]),
        "Regression":           ("LORENZO_REG", "Regression", ["Yes", "No"]),
        "Lorenzo2ndOrder":      ("LORENZO_REG", "Lorenzo2ndOrder", ["Yes", "No"]),
        "Regression2ndOrder":   ("LORENZO_REG", "Regression2ndOrder", ["Yes", "No"]),
        "InterpolationAlgo":    ("INTERP", "InterpolationAlgo",
                                 ["INTERP_ALGO_LINEAR", "INTERP_ALGO_CUBIC"]),
        "InterpolationDirection": ("INTERP", "InterpolationDirection", [0, 1, 2, 3, 4, 5]),
        "QuantizationBinTotal": ("NOPRED", "QuantizationBinTotal", [256, 4096, 65536]),
    }
    out = {}
    for key, (algo, k, values) in probes.items():
        sizes = [sz3_run(x, eps, algo, {k: v})["bytes"] for v in values]
        out[key] = {"algo": algo, "values": values, "bytes": sizes,
                    "reachable": len(set(sizes)) > 1,
                    "spread_pct": 100.0 * (max(sizes) - min(sizes)) / min(sizes)}
    return out


# --------------------------------------------------------------------------- #
# corpora
# --------------------------------------------------------------------------- #

def views(x: torch.Tensor) -> dict[str, np.ndarray]:
    """The two layouts the pilot compared, both materialised contiguous.

    ``native`` is the order the tensor is actually stored and moved in. Feeding a
    compressor a permuted view is how prediction is made to look useless whether
    or not it is, so the layout is an explicit axis rather than an accident of
    whichever reshape the loader happened to do.
    """
    h, s, d = x.shape[1], x.shape[2], x.shape[3]
    xf = x.float()
    return {
        "native_HSD": np.ascontiguousarray(xf.reshape(h, s, d).numpy()),
        "token_major_TD": np.ascontiguousarray(
            xf.permute(0, 2, 1, 3).reshape(s * h, d).numpy()),
        # tokens moved to the fastest axis. The adjacent-difference statistics say
        # the token axis is the only one carrying correlation, so if the predictor
        # is losing because of where the correlation sits rather than because there
        # is none, this layout is where it should win.
        "token_contiguous_HDS": np.ascontiguousarray(
            xf.reshape(h, s, d).permute(0, 2, 1).numpy()),
    }


def axis_predictability(x: torch.Tensor) -> dict:
    """Std of adjacent differences along each axis, over the tensor's own std.

    For a white sequence this is sqrt(2). Below that means a predictor has
    something to work with along that axis; at or above it means there is nothing
    to predict, and any predictor stage is spending bits on residuals no smaller
    than the values it replaced.

    This is the mechanism test for the whole finding: the compression result says
    prediction does not pay, and this says whether that is because the data is
    unpredictable or because SZ3 is looking in the wrong place.
    """
    h, s_, d = x.shape[1], x.shape[2], x.shape[3]
    a = x.float().reshape(h, s_, d).numpy()
    sd = float(a.std())
    return {"adj_over_std_head_dim": float(np.diff(a, axis=2).std() / sd),
            "adj_over_std_tokens": float(np.diff(a, axis=1).std() / sd),
            "adj_over_std_heads": float(np.diff(a, axis=0).std() / sd),
            "white_noise_value": float(np.sqrt(2.0)), "std": sd}


def load_fullcache(root: Path) -> dict[str, torch.Tensor]:
    out = {}
    for p in sorted(root.glob("l*_?.pt")):
        out[p.stem] = torch.load(p)
    return out


def load_pilot(root: Path) -> dict[str, torch.Tensor]:
    return {p.stem: torch.load(p) for p in sorted(root.glob("p*.pt"))}


def scan(tensors: dict[str, torch.Tensor], label: str) -> dict:
    rows: dict[str, dict] = {}
    t_start = time.perf_counter()
    for i, (name, x) in enumerate(tensors.items(), 1):
        bf16_bytes = x.numel() * 2
        sd = float(x.float().std())
        for view_name, arr in views(x).items():
            configs = grid(arr.ndim)
            for c in C_GRID:
                eps = c * sd
                for algo, settings in configs:
                    r = sz3_run(arr, eps, algo, settings)
                    r["ratio"] = bf16_bytes / r["bytes"]
                    key = f"{name}|{view_name}|c{c:g}|{tag(algo, settings)}"
                    rows[key] = {"tensor": name, "view": view_name, "c": c,
                                 "algo": algo, "settings": settings, **r}
        print(f"  [{label}] {i}/{len(tensors)} {name}  "
              f"({time.perf_counter() - t_start:.0f}s)", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def oracle_vs_frozen(rows: dict, view: str) -> dict:
    """Prediction gets the best configuration per tensor; NOPRED gets its default.

    Reported per tensor, so the comparison is paired -- the ratio spread across
    layers is far larger than the gap being measured, and an unpaired mean would
    drown it.
    """
    out = {}
    for c in C_GRID:
        sel = [r for r in rows.values() if r["view"] == view and r["c"] == c]
        by_tensor: dict[str, dict] = {}
        for r in sel:
            t = by_tensor.setdefault(r["tensor"], {"pred": [], "nopred": None})
            if r["algo"] == "NOPRED":
                t["nopred"] = r["ratio"]
            else:
                t["pred"].append((r["ratio"], tag(r["algo"], r["settings"])))
        per = []
        for name, t in sorted(by_tensor.items()):
            best, which = max(t["pred"])
            per.append({"tensor": name, "nopred": t["nopred"],
                        "pred_oracle": best, "pred_best_config": which,
                        "nopred_over_oracle": t["nopred"] / best})
        wins = sum(1 for p in per if p["nopred"] > p["pred_oracle"])
        gains = [p["nopred_over_oracle"] for p in per]
        out[f"c{c:g}"] = {
            "n_tensors": len(per),
            "nopred_wins": wins,
            "nopred_over_oracle_mean": float(np.mean(gains)),
            "nopred_over_oracle_min": float(np.min(gains)),
            "nopred_over_oracle_max": float(np.max(gains)),
            "per_tensor": per,
        }
    return out


def frozen_choice(rows: dict, view: str) -> dict:
    """Best single configuration averaged over all tensors -- the deployable one."""
    out = {}
    for c in C_GRID:
        agg: dict[str, list[float]] = {}
        for r in rows.values():
            if r["view"] == view and r["c"] == c:
                agg.setdefault(tag(r["algo"], r["settings"]), []).append(r["ratio"])
        means = {k: float(np.mean(v)) for k, v in agg.items()}
        ranked = sorted(means.items(), key=lambda kv: -kv[1])
        out[f"c{c:g}"] = {"ranked": ranked[:8], "nopred": means["NOPRED"],
                          "best_predictor": next(k for k, _ in ranked if k != "NOPRED"),
                          "best_predictor_ratio": next(v for k, v in ranked if k != "NOPRED")}
    return out


def main() -> int:
    outdir = Path("results/public/a3_sz3_full")
    outdir.mkdir(parents=True, exist_ok=True)

    print("validating which INI keys reach the compressor ...", flush=True)
    knobs = validate_knobs()
    for k, v in knobs.items():
        print(f"  {k:<24} {'reachable' if v['reachable'] else 'NO EFFECT':>10}  "
              f"spread {v['spread_pct']:5.2f}%")

    corpora = {}
    fc = Path("results/private_raw/fullcache/corpus_v2/fullcache_d00_L2048")
    if fc.is_dir():
        corpora["b0_d00_fullcache"] = load_fullcache(fc)
    pilot = Path("results/private_raw/corpus_kv/kv")
    if pilot.is_dir():
        corpora["pilot_superseded"] = load_pilot(pilot)

    result = {
        "c_grid": C_GRID,
        "knob_validation": knobs,
        "protocol": {
            "denominator": "bf16 bytes",
            "eps": "eps_i = c * std_i, per tensor",
            "handicap": "prediction gets the per-tensor best over the whole reachable "
                        "grid (an oracle); NOPRED gets one default configuration",
            "why_no_split": "an oracle cannot overfit toward the conclusion being "
                            "tested, so a tuning/held-out split would not change "
                            "the direction of the comparison",
        },
        "corpora": {}, "rows": {},
    }

    for label, tensors in corpora.items():
        print(f"\nscanning {label}: {len(tensors)} tensors", flush=True)
        rows = scan(tensors, label)
        result["rows"].update({f"{label}|{k}": v for k, v in rows.items()})
        stats = {n: axis_predictability(x) for n, x in tensors.items()}
        result["corpora"][label] = {
            "axis_predictability": stats,
            "corr_nopred_gain_vs_token_axis": {
                f"c{c:g}": float(np.corrcoef(
                    [100 * (p["nopred_over_oracle"] - 1) for p in
                     oracle_vs_frozen(rows, "native_HSD")[f"c{c:g}"]["per_tensor"]],
                    [stats[p["tensor"]]["adj_over_std_tokens"] for p in
                     oracle_vs_frozen(rows, "native_HSD")[f"c{c:g}"]["per_tensor"]],
                )[0, 1]) for c in C_GRID},
            "n_tensors": len(tensors),
            "n_configs_native": len(grid(3)),
            "n_configs_token_major": len(grid(2)),
            "oracle_vs_frozen": {v: oracle_vs_frozen(rows, v)
                                 for v in VIEWS},
            "frozen_choice": {v: frozen_choice(rows, v)
                              for v in VIEWS},
        }

    (outdir / "sz3_full_scan.json").write_text(json.dumps(result, indent=2))

    for label, summ in result["corpora"].items():
        print(f"\n=== {label} ({summ['n_tensors']} tensors, "
              f"{summ['n_configs_native']} configs native) ===")
        for view in VIEWS:
            print(f"  {view}:")
            ov = summ["oracle_vs_frozen"][view]
            fz = summ["frozen_choice"][view]
            for c in C_GRID:
                k = f"c{c:g}"
                print(f"    c={c:<5g} NOPRED beats the per-tensor oracle on "
                      f"{ov[k]['nopred_wins']}/{ov[k]['n_tensors']} tensors, "
                      f"by {100*(ov[k]['nopred_over_oracle_mean']-1):+.1f}% mean "
                      f"[{100*(ov[k]['nopred_over_oracle_min']-1):+.1f}%, "
                      f"{100*(ov[k]['nopred_over_oracle_max']-1):+.1f}%]")
                print(f"           frozen: NOPRED {fz[k]['nopred']:.2f}x vs "
                      f"{fz[k]['best_predictor']} {fz[k]['best_predictor_ratio']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
