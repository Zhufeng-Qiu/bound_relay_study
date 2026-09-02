"""D21 — evaluate the decision rule on measured costs.

Candidates come from the A40 cost table (D9–D15); bandwidths span the measured
fabric points and the modelled range around them. The controller is scored
against a zero-margin oracle, and what is being measured is the price of the
safety margin: how much latency it gives up, and how many chunks it flips from
compress to bypass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boundrelay.codec.contract import Mode  # noqa: E402
from boundrelay.policy.cost_model import FABRIC_GBPS, CostInputs, net_benefit_s  # noqa: E402
from boundrelay.policy.decision import Candidate, decide, oracle  # noqa: E402

MARGINS_MS = [0.0, 0.10, 0.25]
BW_POINTS = [0.5, 2.0, 7.0, 8.45, 12.5, 23.0, 50.0, 64.0, 450.0]   # incl. measured 8.45 / 23


def candidates_for(table: dict, payload_key: str) -> tuple[int, list[Candidate]]:
    """Every measured (bit width) option at one payload size."""
    out, original = [], None
    for k, r in table.items():
        w, shape = k.split("_", 1)
        if shape != payload_key:
            continue
        original = r["original_bytes"]
        out.append(Candidate(
            mode=Mode.ERROR_BOUNDED, label=f"{w}@eps0.15",
            compressed_bytes=r["compressed_bytes"],
            t_encode_s=r["kernel_s"], t_decode_s=r["decode_kernel_s"],
            eps=0.15))
    return original, out


def main() -> int:
    d = json.loads(Path("results/private_raw/gpu_session/cost_table.json").read_text())
    table = d["table"]
    payloads = sorted({k.split("_", 1)[1] for k in table},
                      key=lambda s: int(s.strip("()").split(",")[0]))

    rows, summary = [], {}
    for margin_ms in MARGINS_MS:
        margin_s = margin_ms / 1e3
        n_compress = n_bypass = n_flip = 0
        regret_ms = 0.0
        for pk in payloads:
            original, cands = candidates_for(table, pk)
            for bw in BW_POINTS:
                dec = decide(original, bw, cands, margin_s)
                orc = oracle(original, bw, cands)
                # realised benefit of each choice, scored without the margin
                def realised(x):
                    if x.mode is Mode.BF16_PASSTHROUGH:
                        return 0.0
                    c = next(c for c in cands if c.label == x.label)
                    return net_benefit_s(CostInputs(original, c.compressed_bytes, bw,
                                                    c.t_encode_s, c.t_decode_s))
                r_dec, r_orc = realised(dec), realised(orc)
                regret_ms += (r_orc - r_dec) * 1e3
                if dec.mode is Mode.BF16_PASSTHROUGH:
                    n_bypass += 1
                else:
                    n_compress += 1
                if dec.label != orc.label:
                    n_flip += 1
                if margin_ms == 0.0:
                    rows.append({"payload": pk, "bw_gbps": bw, "choice": dec.label,
                                 "net_ms": round(r_dec * 1e3, 4),
                                 "bypass_reason": dec.bypass_reason.value
                                 if dec.bypass_reason else None})
        total = len(payloads) * len(BW_POINTS)
        summary[f"margin_{margin_ms:g}ms"] = {
            "compress": n_compress, "bypass": n_bypass,
            "bypass_rate": round(n_bypass / total, 3),
            "flipped_vs_oracle": n_flip,
            "total_regret_ms": round(regret_ms, 4),
            "mean_regret_ms": round(regret_ms / total, 5),
        }

    out = Path("results/public/d21_policy"); out.mkdir(parents=True, exist_ok=True)
    (out / "policy_eval.json").write_text(json.dumps(
        {"bw_points_gbps": BW_POINTS, "measured_points": [8.45, 23.0],
         "fabric": FABRIC_GBPS, "summary": summary, "decisions_margin0": rows}, indent=2))

    print(f"{'margin':>8} {'compress':>9} {'bypass':>7} {'bypass rate':>12} "
          f"{'flipped':>8} {'mean regret':>12}")
    print("-" * 62)
    for k, v in summary.items():
        print(f"{k.split('_')[1]:>8} {v['compress']:>9} {v['bypass']:>7} "
              f"{v['bypass_rate']:>12.1%} {v['flipped_vs_oracle']:>8} "
              f"{v['mean_regret_ms']:>11.4f}ms")

    print(f"\n{'payload':>16} " + "".join(f"{b:>9g}" for b in BW_POINTS))
    print("-" * (17 + 9 * len(BW_POINTS)))
    for pk in payloads:
        mb = next(r["original_bytes"] for k, r in table.items()
                  if k.split("_", 1)[1] == pk) / 1e6
        cells = []
        for bw in BW_POINTS:
            r = next(x for x in rows if x["payload"] == pk and x["bw_gbps"] == bw)
            cells.append("bypass" if r["choice"] == "bf16_passthrough" else r["choice"].split("@")[0])
        print(f"{mb:13.2f} MB " + "".join(f"{c:>9}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
