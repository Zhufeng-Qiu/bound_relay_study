"""B1 — how much of cuSZp's compression ratio is predictable before compressing?

The question a transport decision actually faces: at the moment you must choose
compress or bypass, you do not yet know the compressed size. So the features are
split by what it costs to obtain them.

``metadata``    layer depth, K/V, sequence length, eps_rel. Free — known from the
                request. This is what a deployable policy could use.
``statistics``  metadata plus per-tensor statistics (std, range, outlier rate,
                near-zero fraction, adjacent-difference spread). Not free: it
                requires a pass over the tensor, and that pass is reported as a
                cost rather than waved through as a "feature".
``oracle``      the measured compressed bytes. Not a predictor; the upper bound.

Splits group by **document**. Six layers x two kinds x four lengths from one
article are not independent observations, and a random split over tensors would
leak the same article across train and test and report an accuracy that does not
exist. Every hyperparameter is chosen inside training folds.

The target is log ratio rather than ratio: ratio is bounded below by ~1 and
right-skewed, and squared error on it would weight the large-ratio tail in a way
that has nothing to do with the transport decision.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C_GRID = [0.01, 0.03, 0.10]
MODE = "fixed"                      # leads on ratio; all modes share a median cost
META = ["layer_frac", "is_k", "log_seq", "log_c"]
STATS = ["log_std", "log_range_over_std", "outlier_rate", "near_zero_frac",
         "adjdiff_over_std"]


def build(manifest: Path):
    d = json.loads(manifest.read_text())
    n_layers = d["n_layers"]
    X, y, groups, rows = [], [], [], []
    for o in d["observations"]:
        if "cuszp" not in o or o["std"] <= 0:
            continue
        for c in C_GRID:
            k = f"c{c:g}_{MODE}"
            if k not in o["cuszp"]:
                continue
            r = o["cuszp"][k]["ratio"]
            sd = o["std"]
            X.append([
                o["layer"] / (n_layers - 1), 1.0 if o["kind"] == "k" else 0.0,
                np.log(o["seq_len"]), np.log(c),
                np.log(sd), np.log(max(o["range"] / sd, 1e-9)),
                o["outlier_rate"], o["near_zero_frac"], o["adjdiff_over_std"],
            ])
            y.append(np.log(r))
            groups.append(o["doc"])
            rows.append({"doc": o["doc"], "layer": o["layer"], "kind": o["kind"],
                         "seq_len": o["seq_len"], "c": c, "ratio": r,
                         "bytes": o["cuszp"][k]["cmp_bytes"], "bf16": o["bf16_bytes"]})
    return np.array(X), np.array(y), np.array(groups), rows


def evaluate(X, y, groups, cols, name, rows):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    idx = [META + STATS].pop()
    use = [idx.index(c) for c in cols]
    Xs = X[:, use]

    gkf = GroupKFold(n_splits=6)
    per_doc: dict[str, list] = {}
    ae_ratio, ae_bytes = [], []
    for tr, te in gkf.split(Xs, y, groups):
        # both model families are fitted inside the fold; the one with better
        # in-fold cross-validated error is used on the held-out documents, so no
        # model choice ever sees test documents
        cands = [RidgeCV(alphas=np.logspace(-3, 3, 13)),
                 RandomForestRegressor(n_estimators=200, max_depth=4,
                                       min_samples_leaf=8, random_state=0)]
        inner = GroupKFold(n_splits=3)
        best, best_err = None, np.inf
        for m in cands:
            errs = []
            for itr, ite in inner.split(Xs[tr], y[tr], groups[tr]):
                m.fit(Xs[tr][itr], y[tr][itr])
                errs.append(np.abs(m.predict(Xs[tr][ite]) - y[tr][ite]).mean())
            if np.mean(errs) < best_err:
                best, best_err = m, np.mean(errs)
        best.fit(Xs[tr], y[tr])
        pred = best.predict(Xs[te])
        for i, j in enumerate(te):
            pr, ar = float(np.exp(pred[i])), float(np.exp(y[j]))
            ae_ratio.append(abs(pr - ar))
            pb = rows[j]["bf16"] / max(pr, 1e-9)
            ae_bytes.append(abs(pb - rows[j]["bytes"]) / rows[j]["bytes"])
            per_doc.setdefault(rows[j]["doc"], []).append(abs(pr - ar))

    doc_mae = {k: float(np.mean(v)) for k, v in per_doc.items()}
    boot = [float(np.mean([doc_mae[d] for d in np.random.choice(
        list(doc_mae), len(doc_mae), replace=True)])) for _ in range(2000)]
    return {
        "features": name, "n": len(y), "n_docs": len(set(groups)),
        "ratio_mae": float(np.mean(ae_ratio)),
        "ratio_mae_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "byte_mape": float(np.mean(ae_bytes)),
        "per_document_mae": doc_mae,
    }


def main() -> int:
    np.random.seed(0)
    X, y, groups, rows = build(Path("results/public/b0_corpus/manifest.json"))
    ratios = np.exp(y)
    print(f"{len(y)} observations, {len(set(groups))} documents, mode={MODE}")
    print(f"ratio spread: {ratios.min():.2f}x – {ratios.max():.2f}x  "
          f"(mean {ratios.mean():.2f}x, sd {ratios.std():.3f})\n")

    out = {"mode": MODE, "c_grid": C_GRID, "models": []}
    for cols, name in ((META, "metadata-only"), (META + STATS, "statistics-aware")):
        r = evaluate(X, y, groups, cols, name, rows)
        out["models"].append(r)
        lo, hi = r["ratio_mae_ci"]
        print(f"  {name:>18}: ratio MAE {r['ratio_mae']:.4f}x  "
              f"[{lo:.4f}, {hi:.4f}]   byte MAPE {100*r['byte_mape']:.2f}%")

    # the null a predictor has to beat: predict the training-fold mean
    base = float(np.mean(np.abs(ratios - ratios.mean())))
    out["constant_baseline_mae"] = base
    print(f"  {'constant (mean)':>18}: ratio MAE {base:.4f}x   <- the bar\n")

    a, b = out["models"][0]["ratio_mae"], out["models"][1]["ratio_mae"]
    print(f"statistics buy {100*(a-b)/a:+.1f}% over metadata alone; "
          f"metadata buys {100*(base-a)/base:+.1f}% over the constant.")
    Path("results/public/b1_predictability").mkdir(parents=True, exist_ok=True)
    Path("results/public/b1_predictability/predictability.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
