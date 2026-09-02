"""Shared plotting style. One scale, one palette, readable in print."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED = "#17211F", "#7F8C89"
MODEL, MEASURED, STOP = "#A96C15", "#1B6863", "#A2372B"
SERIES = ["#1B6863", "#A96C15", "#3C6E9F", "#8A5A9B"]


def figure(w=7.2, h=4.0):
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160,
    })
    return plt.subplots(figsize=(w, h))


def save(fig, out):
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
