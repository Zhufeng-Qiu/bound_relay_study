"""Figure 1 -- break-even map.

x: effective bandwidth, log, 0.1 -> 1000 GB/s (four orders of magnitude)
y: net latency benefit, with the zero line drawn

Two rules this figure lives or dies by:

1. Measured points are markers with error bars; the cost model is a line, and
   the legend says which is which. Blur that and it stops being analysis.
2. Real fabric positions are annotated. A few axvlines -- and it is what lets one
   figure answer three different audiences at once.
"""

from __future__ import annotations

import json
from pathlib import Path

from _style import MEASURED, MODEL, MUTED, STOP, figure, save
from boundrelay.policy.cost_model import FABRIC_GBPS


def render(sweep_json: Path, measured_json: Path | None, out: Path) -> None:
    s = json.loads(Path(sweep_json).read_text())
    fig, ax = figure()

    y_ms = [v * 1e3 for v in s["net_benefit_s"]]
    ax.plot(s["bw_gbps"], y_ms, color=MODEL, lw=2.0, label="cost model", zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.fill_between(s["bw_gbps"], 0, y_ms, where=[v > 0 for v in y_ms],
                    color=MEASURED, alpha=0.12, zorder=0)

    be = s["break_even_gbps"]
    if 0 < be < float("inf"):
        ax.axvline(be, color=STOP, lw=1.2, ls="--", zorder=2)
        ax.annotate(f"break-even\n{be:.1f} GB/s", (be, 0), textcoords="offset points",
                    xytext=(6, -34), color=STOP, fontsize=8)

    for name, bw in FABRIC_GBPS.items():
        ax.axvline(bw, color=MUTED, lw=0.7, ls=":", alpha=0.7, zorder=1)
        ax.annotate(name, (bw, ax.get_ylim()[1]), rotation=90, va="top", ha="right",
                    fontsize=7, color=MUTED, textcoords="offset points", xytext=(-2, -4))

    if measured_json and Path(measured_json).exists():
        m = json.loads(Path(measured_json).read_text())
        ax.errorbar(m["bw_gbps"], [v * 1e3 for v in m["net_benefit_s"]],
                    yerr=[v * 1e3 for v in m.get("err_s", [0] * len(m["bw_gbps"]))],
                    fmt="o", ms=4.5, color=MEASURED, capsize=3, lw=1.2,
                    label="measured", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("effective bandwidth (GB/s)")
    ax.set_ylabel("net latency benefit (ms)")
    ax.set_title(f"Break-even: {s['original_bytes']/1e6:.0f} MB payload, "
                 f"{s['original_bytes']/s['compressed_bytes']:.1f}x, "
                 f"codec {s['codec_cost_s']*1e3:.2f} ms", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, out)


if __name__ == "__main__":
    import sys
    render(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None,
           Path(sys.argv[-1]))
