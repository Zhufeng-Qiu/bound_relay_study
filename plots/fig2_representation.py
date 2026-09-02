"""Figure 2 — where the error budget should be spent, measured on real KV cache.

Ratio at matched eps for {blockwise, per-channel, per-token}, split by tensor
kind (keys vs values) and by depth.

This figure is about where *this* codec should spend its error budget. It is not
evidence that general error-bounded compressors do poorly here — D14 measured SZ3
compressing these same tensors 5x better, so that reading would be wrong.

Keys carry strong per-channel structure — at layer 0,
blockwise and per-token allocation achieve no compression at all while
per-channel reaches 3.5x. Values behave the opposite way, with per-channel often
the worst choice. The right axis is a property of the tensor, not of the codec,
which is exactly why a fixed representation underperforms and an adaptive one has
something to adapt to.

Entirely offline, so it cannot be lost with a failed GPU session.
"""

from __future__ import annotations

import json
from pathlib import Path

from _style import SERIES, figure, save


def render(by_layer_json: Path, out: Path, eps_index: int = 1) -> None:
    d = json.loads(Path(by_layer_json).read_text())
    allocs, eps = d["allocations"], d["eps_values"]
    rows = d["mean_ratio"]

    layers = sorted({int(k.split("_l")[1]) for k in rows})
    figure()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5), sharey=True, dpi=160)

    for ax, kind, title in zip(axes, ("k", "v"), ("keys", "values")):
        w, n = 0.26, len(allocs)
        for i, a in enumerate(allocs):
            xs = [j + (i - (n - 1) / 2) * w for j in range(len(layers))]
            ys = [rows[f"{kind}_l{l}"][a][eps_index] for l in layers]
            ax.bar(xs, ys, w * 0.9, label=a.replace("_", "-"), color=SERIES[i])
            for x, y in zip(xs, ys):
                if y < 1.05:                      # no compression at all
                    ax.annotate("none", (x, y), ha="center", va="bottom",
                                fontsize=6, color=SERIES[i], rotation=90,
                                textcoords="offset points", xytext=(0, 2))
        ax.axhline(1.0, color="#7F8C89", lw=0.8, ls=":")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"layer {l}" for l in layers])
        ax.set_title(title, fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("compression ratio (vs bf16)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{d['model']} KV cache, ε = {eps[eps_index]:g}",
                 fontsize=9, x=0.005, ha="left")
    save(fig, out)


if __name__ == "__main__":
    import sys
    render(Path(sys.argv[1]), Path(sys.argv[2]))
