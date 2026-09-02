"""Gate A check: all three figures must render before any GPU spend.

Figures 1 and 3 are drawn from synthetic inputs here and are clearly labelled as
such in the emitted JSON. Figure 2 is drawn from the *real* reference codec on
synthetic tensors -- the numbers are genuine codec output, only the tensors are
stand-ins for the corpus that D7-D8 will capture.

The point of running this on day 4 is to prove the whole path -- codec to JSON to
figure -- works while it is still free to fix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))

from boundrelay.bench.table_a import allocation_study, int8_baseline  # noqa: E402
from boundrelay.policy.cost_model import CostInputs, sweep  # noqa: E402
from tests.conftest import hidden_state, with_channel_outliers  # noqa: E402

OUT = Path("results/public/preflight")
FIGS = Path("results/public/preflight/figures")
EPS = [0.05, 0.15, 0.5]


def main() -> None:
    g = torch.Generator().manual_seed(20260902)
    clean = hidden_state(tokens=64, gen=g)
    tensors = {"clean": clean, "outlier": with_channel_outliers(clean)}

    # --- Figure 2: real codec output on stand-in tensors -------------------
    alloc = OUT / "allocation_study.json"
    allocation_study(tensors, EPS, alloc)

    for name, x in tensors.items():
        for pc in (False, True):
            r, e = int8_baseline(x, pc)
            tag = "int8_per_channel" if pc else "int8_per_tensor"
            print(f"  {name:>8} {tag:>18}: {r:.2f}x  max_err={e:.4f}")

    import fig2_representation
    fig2_representation.render(alloc, FIGS / "fig2_representation.png")

    # --- Figure 1: cost model, synthetic timings --------------------------
    c = CostInputs(original_bytes=20_000_000, compressed_bytes=8_000_000,
                   bw_gbps=50.0, t_encode_s=0.4e-3, t_decode_s=0.3e-3)
    s = sweep(c)
    s["SYNTHETIC"] = "codec timings are placeholders until the A40/H100 benchmark"
    (OUT / "sweep.json").write_text(json.dumps(s, indent=2))
    measured = {"SYNTHETIC": True, "bw_gbps": [7.0, 12.5, 50.0],
                "net_benefit_s": [1.02e-3, 0.26e-3, -0.46e-3],
                "err_s": [0.04e-3, 0.03e-3, 0.02e-3]}
    (OUT / "measured.json").write_text(json.dumps(measured, indent=2))

    import fig1_breakeven
    fig1_breakeven.render(OUT / "sweep.json", OUT / "measured.json",
                          FIGS / "fig1_breakeven.png")

    # --- Figure 3: synthetic quality curve --------------------------------
    q = {"SYNTHETIC": "placeholder until D20 perplexity sweep",
         "eps": [0.02, 0.05, 0.15, 0.5, 1.0],
         "ppl_delta": [0.01, 0.04, 0.18, 0.9, 3.1],
         "wer_delta": None, "ratio": [1.9, 2.1, 2.5, 3.7, 4.6], "gate": 0.5}
    (OUT / "quality.json").write_text(json.dumps(q, indent=2))

    import fig3_quality
    fig3_quality.render(OUT / "quality.json", FIGS / "fig3_quality.png")

    print("\nGate A: all three figures render.")


if __name__ == "__main__":
    main()
