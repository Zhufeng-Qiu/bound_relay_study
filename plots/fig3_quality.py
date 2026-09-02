"""Figure 3 — the quality–compression frontier, measured.

x: error bound ε (log)
left y: perplexity cost, as a percentage of the uncompressed baseline
right y: compression ratio actually achieved at that ε

Semantics are prefill → transport → decode: a prefix is prefilled, its KV cache
round-tripped through the codec, and the continuation scored against the
reconstructed cache. That is the disaggregated-serving scenario, not
"compress on every cache update", which no deployment does.

The controller's operating point is marked, because a frontier without the chosen
point on it invites the reader to pick their own and argue with a number nobody
proposed.
"""

from __future__ import annotations

import json
from pathlib import Path

from _style import MEASURED, MODEL, MUTED, STOP, figure, save

OPERATING_EPS = 0.15   # what the decision map (D21) actually selects


def render(quality_json: Path, out: Path) -> None:
    d = json.loads(Path(quality_json).read_text())
    eps = d["eps"]
    base = d["baseline_ppl"]
    pct = [100.0 * v / base for v in d["ppl_delta"]]

    fig, ax = figure(w=6.8, h=3.9)
    ax.plot(eps, pct, "o-", color=MEASURED, lw=1.9, ms=5,
            label="perplexity cost", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("error bound ε")
    ax.set_ylabel("perplexity increase (% of baseline)")

    ax2 = ax.twinx()
    ax2.plot(eps, d["ratio"], "s--", color=MODEL, lw=1.6, ms=4.5,
             label="compression ratio", zorder=3)
    ax2.set_ylabel("compression ratio (vs bf16)", color=MODEL)
    ax2.tick_params(axis="y", colors=MODEL)
    ax2.spines[["top"]].set_visible(False)

    if OPERATING_EPS in eps:
        i = eps.index(OPERATING_EPS)
        ax.axvline(OPERATING_EPS, color=STOP, lw=1.2, ls="--", zorder=2)
        ax.annotate(f"controller operating point\nε={OPERATING_EPS:g}: "
                    f"{d['ratio'][i]:.2f}×  for  +{pct[i]:.1f}%",
                    (OPERATING_EPS, pct[i]), fontsize=7.5, color=STOP,
                    textcoords="offset points", xytext=(8, 14))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"{d['model']}, WikiText-2, {d['n_seq']} sequences · "
                 f"prefill {d['prefix_len']} → decode {d['seq_len']-d['prefix_len']} · "
                 f"baseline ppl {base:.1f}", fontsize=8.5, loc="left", color=MUTED)
    save(fig, out)


if __name__ == "__main__":
    import sys
    render(Path(sys.argv[1]), Path(sys.argv[2]))
