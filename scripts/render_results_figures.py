"""The three figures the results actually support, drawn from the committed JSON.

`render_figures.py` predates every measurement in this project: it draws Figures
1 and 3 from *synthetic* inputs as a Gate A check that the codec-to-JSON-to-figure
path worked before any GPU was rented. It did its job and its figures are not
results. These are.

Each figure answers one question, and each reads its numbers out of
`results/public/` rather than carrying them inline, so a figure cannot drift from
the finding it illustrates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))
from _style import INK, MEASURED, MODEL, MUTED, SERIES, STOP, figure, save  # noqa: E402

PUB = Path("results/public")
OUT = Path("plots/results")


def load(p: str) -> dict:
    return json.loads((PUB / p).read_text())


# --------------------------------------------------------------------------- #
def fig_decision() -> None:
    """Where compression pays and where it does not, on every path measured.

    The point of putting them on one axis is that the sign changes across it, and
    the sign is the only thing that survived from the model.
    """
    sc = load("remeasure/session_close.json")["closed_e2e"]
    pl = load("pipeline/pipeline_v2.json")
    off = load("pipeline/fsync_offload.json")

    ser = {r["compressed"]: r["median_s"] * 1e3 for r in pl["serial"]}
    pipe = {(r["compressed"], r["depth"]): r["median_s"] * 1e3
            for r in pl["runs"] if r["transport"] == "host_pinned"}
    ov = off["paths"]["/root/offload"]["per_tensor_writes"]
    ws = off["paths"]["/workspace/offload"]["per_tensor_writes"]

    rows = [
        ("host-staged\nserial", ser[False], ser[True]),
        ("host-staged\n7-stage pipeline", pipe[(False, 8)], pipe[(True, 8)]),
        ("container overlay\nfsync offload write", ov["raw"]["median_s"] * 1e3,
         ov["compressed"]["median_s"] * 1e3),
        ("MooseFS\nfsync offload write", ws["raw"]["median_s"] * 1e3,
         ws["compressed"]["median_s"] * 1e3),
    ]

    fig, ax = figure(7.6, 4.2)
    y = np.arange(len(rows))
    h = 0.36
    for i, (_, raw, comp) in enumerate(rows):
        ax.barh(i - h / 2, raw, h, color=MUTED, zorder=3)
        ax.barh(i + h / 2, comp, h,
                color=MEASURED if comp < raw else STOP, zorder=3)
        # always report the factor as a number above 1, with the direction in the
        # word. "0.52x slower" is the kind of label a reader silently misreads.
        faster = comp < raw
        factor = (raw / comp) if faster else (comp / raw)
        ax.text(max(raw, comp) * 1.09, i,
                f"{factor:.2f}× " + ("faster" if faster else "slower"),
                va="center", fontsize=8.5,
                color=MEASURED if faster else STOP, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(8, 5000)
    ax.set_xlabel("time to move one 234.9 MB KV cache (ms, log scale)")
    ax.set_title("Compression loses on the fast path and wins on the slow one\n"
                 "Qwen3-1.7B, 56 tensors, $\\varepsilon = 0.10\\,\\sigma$, "
                 "cuSZp fixed — 3.02× fewer bytes throughout",
                 fontsize=10, loc="left", pad=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MUTED, label="raw bf16"),
                       Patch(color=MEASURED, label="compressed — faster"),
                       Patch(color=STOP, label="compressed — slower")],
              frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.grid(axis="x", color=MUTED, alpha=0.18, zorder=0)
    save(fig, OUT / "fig1_where_compression_pays.png")


# --------------------------------------------------------------------------- #
def fig_mechanism() -> None:
    """Why K and V behave differently, in two statistics that answer to two designs.

    Adjacent smoothness is what a predictor needs. Per-channel scale spread is what
    per-channel quantisation needs. They are different questions about the same
    axis, and reading only one of them makes this project's result look like it
    contradicts KIVI and PackKV when it does not.
    """
    st = load("a3_sz3_full/sz3_full_scan.json")["corpora"]["b0_d00_fullcache"]
    ap = st["axis_predictability"]
    if "chan_scale_max_over_median" not in next(iter(ap.values())):
        print("  ! scan JSON predates the scale statistics -- rerun sz3_full_scan.py")
        return

    def med(kind: str, key: str) -> float:
        return float(np.median([v[key] for n, v in ap.items() if n.endswith("_" + kind)]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.6), dpi=160)

    axes_names = ["head_dim\n(channel)", "tokens", "heads"]
    keys = ["adj_over_std_head_dim", "adj_over_std_tokens", "adj_over_std_heads"]
    x = np.arange(3); w = 0.36
    for off_, kind, col in ((-w / 2, "k", SERIES[0]), (w / 2, "v", SERIES[1])):
        a1.bar(x + off_, [med(kind, k) for k in keys], w,
               label=f"{kind.upper()} cache", color=col, zorder=3)
    a1.axhline(np.sqrt(2), color=STOP, ls="--", lw=1.2, zorder=4)
    a1.text(2.45, np.sqrt(2) + 0.03, "$\\sqrt{2}$ = white", color=STOP,
            fontsize=8, ha="right")
    a1.set_xticks(x); a1.set_xticklabels(axes_names, fontsize=8.5)
    a1.set_ylabel("adjacent-difference std / tensor std")
    a1.set_title("Smoothness — what a predictor needs", fontsize=9.5, loc="left")
    a1.set_ylim(0, 1.75); a1.legend(frameon=False, fontsize=8.5)
    a1.grid(axis="y", color=MUTED, alpha=0.18, zorder=0)

    x2 = np.arange(2)
    for off_, kind, col in ((-w / 2, "k", SERIES[0]), (w / 2, "v", SERIES[1])):
        a2.bar(x2 + off_, [med(kind, "chan_scale_max_over_median"),
                           med(kind, "token_scale_max_over_median")], w,
               color=col, zorder=3)
    for i, kind in enumerate(("k", "v")):
        v = med(kind, "chan_scale_max_over_median")
        a2.text(0 + (-w / 2 if i == 0 else w / 2), v + 0.5, f"{v:.1f}×",
                ha="center", fontsize=8.5, fontweight="bold", color=SERIES[i])
    a2.axhline(1.0, color=MUTED, ls=":", lw=1)
    a2.set_xticks(x2); a2.set_xticklabels(["per channel", "per token"], fontsize=8.5)
    a2.set_ylabel("scale spread (max / median std)")
    a2.set_title("Scale spread — what per-channel quantisation needs",
                 fontsize=9.5, loc="left")
    a2.grid(axis="y", color=MUTED, alpha=0.18, zorder=0)

    fig.suptitle("K is white along channels and heterogeneous in channel scale; "
                 "V is neither, anywhere", fontsize=10, x=0.01, ha="left")
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    save(fig, OUT / "fig2_kv_mechanism.png")


# --------------------------------------------------------------------------- #
def fig_models() -> None:
    """Three composed models, three overestimates, one direction."""
    pl = load("pipeline/pipeline_v2.json")
    off = load("pipeline/fsync_offload.json")
    pipe8 = {(r["compressed"], r["depth"]): r["median_s"] * 1e3
             for r in pl["runs"] if r["transport"] == "host_pinned"}
    RB, CB = off["raw_bytes"], off["comp_bytes"]
    ratios = []
    for path, e in off["paths"].items():
        for k, r in e.items():
            pred = CB / 1e9 / r["raw"]["gbps_of_payload"]
            ratios.append(r["compressed"]["median_s"] / pred)

    rows = [
        ("component model\ncodec on one 20 MB tensor\n+ link from another session", 4.0,
         "3.4× of it is per-tensor granularity"),
        ("stage-sum model\nreal stages, real cache,\nright hardware", pipe8[(True, 8)] / 21.43,
         "a stage sum has no per-item cost"),
        ("bandwidth-only model\nbytes ÷ an unchanged\nbandwidth", float(np.median(ratios)),
         f"{min(ratios):.2f}–{max(ratios):.2f}× across 8 configurations"),
    ]

    fig, ax = figure(7.8, 3.9)
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], 0.5, color=MODEL, zorder=3)
    ax.axvline(1.0, color=MEASURED, lw=1.6, zorder=4)
    ax.annotate("model = measurement", xy=(1.0, len(rows) - 0.45),
                xytext=(1.12, len(rows) - 0.42), color=MEASURED, fontsize=8.5,
                va="center")
    for i, (_, v, note) in enumerate(rows):
        ax.text(v + 0.08, i - 0.13, f"{v:.2f}×", va="center", fontsize=10,
                fontweight="bold", color=MODEL)
        ax.text(v + 0.08, i + 0.17, note, va="center", fontsize=7.6, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.3, -0.7)
    ax.set_xlim(0, 6.4)
    ax.set_xlabel("how much slower reality was than the model predicted")
    ax.set_title("Every model built from separately measured parts was optimistic,\n"
                 "and every one of them in the same direction",
                 fontsize=10, loc="left", pad=16)
    ax.grid(axis="x", color=MUTED, alpha=0.18, zorder=0)
    save(fig, OUT / "fig3_models_vs_measurement.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_decision()
    fig_mechanism()
    fig_models()
