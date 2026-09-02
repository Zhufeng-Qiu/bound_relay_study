"""Figure 4 — what the controller decides, over payload × bandwidth.

Every cell is a decision taken from measured codec cost (A40, D9–D15) and a
bandwidth axis whose two marked points were measured on the same hardware (D6).
The staircase is the break-even boundary seen from the controller's side: the
same curve as Figure 1, read as an operating decision rather than a latency delta.

The honest headline is the colour balance. Compression is the exception — the
controller bypasses in most of the grid — and the interesting region is narrow
and sits exactly where the real hardware does.
"""

from __future__ import annotations

import json
from pathlib import Path

from _style import MEASURED, MUTED, STOP, figure, save


def render(eval_json: Path, out: Path) -> None:
    d = json.loads(Path(eval_json).read_text())
    bws = d["bw_points_gbps"]
    rows = [r for r in d["decisions_margin0"]]
    payloads = sorted({r["payload"] for r in rows},
                      key=lambda s: int(s.strip("()").split(",")[0]))
    mb = {p: int(p.strip("()").split(",")[0]) * 2048 * 2 / 1e6 for p in payloads}

    figure()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=160)

    for yi, p in enumerate(payloads):
        for xi, bw in enumerate(bws):
            r = next(x for x in rows if x["payload"] == p and x["bw_gbps"] == bw)
            comp = r["choice"] != "bf16_passthrough"
            ax.add_patch(plt.Rectangle(
                (xi - .5, yi - .5), 1, 1,
                facecolor=MEASURED if comp else "none",
                alpha=0.85 if comp else 1.0,
                edgecolor=MUTED, linewidth=0.5))
            ax.text(xi, yi, r["choice"].split("@")[0] if comp else "bypass",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if comp else MUTED)

    for bw in d["measured_points"]:
        xi = bws.index(bw)
        ax.axvline(xi, color=STOP, lw=1.6, ls="--", zorder=5)
        ax.annotate(f"measured\n{bw:g} GB/s", (xi, len(payloads) - 0.35),
                    ha="center", va="bottom", fontsize=7, color=STOP)

    ax.set_xticks(range(len(bws)))
    ax.set_xticklabels([f"{b:g}" for b in bws])
    ax.set_yticks(range(len(payloads)))
    ax.set_yticklabels([f"{mb[p]:.2f} MB" for p in payloads])
    ax.set_xlim(-.5, len(bws) - .5)
    ax.set_ylim(-.5, len(payloads) - .5 + 0.45)
    ax.set_xlabel("effective bandwidth (GB/s)")
    ax.set_ylabel("payload")
    ax.set_title("controller decision, ε = 0.15, measured A40 codec cost",
                 fontsize=9, loc="left")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    save(fig, out)


if __name__ == "__main__":
    import sys
    render(Path(sys.argv[1]), Path(sys.argv[2]))
