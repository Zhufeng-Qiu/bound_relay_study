"""Why do a full forward and a cached forward disagree, and by how much?

B1's preflight measured a 5.29e-03 mean-NLL gap between scoring the continuation
from a prefix cache and scoring it inside one long forward. The protocol calls
anything above 1e-3 an investigation trigger rather than a failure, and this is the
investigation. It was first run ad hoc with the output going only to a terminal;
this version saves the per-token record so the conclusion can be checked instead of
taken.

Two candidate causes leave different fingerprints:

* a **position or offset error** is systematic -- the cached scores would line up
  with the full scores one position over, so shifting the comparison would *reduce*
  the disagreement;
* **bf16 kernel selection** varying with sequence length is diffuse -- it touches
  every token a little, and disappears in fp32.

Both are tested, on the two development articles only. No held-out article is used:
this measures the harness, not the corpus.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL = "Qwen/Qwen3-1.7B"
REV = "b9352fbb8ce704292730cf54b3b1dceb2a808738"
PREFIX, SCORED = 1024, 256


def per_token(model, ids):
    """Cached and full per-token NLL over the same 256 target positions."""
    with torch.inference_mode():
        pre = ids[:, :PREFIX]
        cont_in = ids[:, PREFIX:PREFIX + SCORED]
        tgt = ids[:, PREFIX + 1:PREFIX + 1 + SCORED]
        o = model(pre, use_cache=True)
        lg = model(cont_in, past_key_values=o.past_key_values).logits
        cached = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]),
                                 tgt.reshape(-1), reduction="none")
        full_lg = model(ids[:, :PREFIX + SCORED]).logits[:, PREFIX:PREFIX + SCORED]
        full = F.cross_entropy(full_lg.float().reshape(-1, full_lg.shape[-1]),
                               tgt.reshape(-1), reduction="none")
    return cached.float().cpu().numpy(), full.float().cpu().numpy()


def shift_test(a: np.ndarray, b: np.ndarray) -> dict:
    """If the two are misaligned by k positions, shifting by k reduces the gap."""
    out = {}
    for k in (-2, -1, 0, 1, 2):
        if k == 0:
            out["0"] = float(np.abs(a - b).mean())
            continue
        lo, hi = max(0, k), len(a) + min(0, k)
        out[str(k)] = float(np.abs(a[lo:hi] - b[max(0, -k):len(b) + min(0, -k)]).mean())
    out["best_shift"] = min(out, key=lambda kk: out[kk] if kk != "best_shift" else 9e9)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/out/alignment")
    ap.add_argument("--docs", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    raw = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                 split="train")["text"])
    p = re.split(r"\n\s=\s([^=][^\n]*?)\s=\s\n", raw)
    arts = [(p[i].strip(), p[i + 1]) for i in range(1, len(p) - 1, 2)
            if len(p[i + 1]) >= 12000][:a.docs]

    jl = (out / "alignment_per_token.jsonl").open("w")
    summary = {"model": MODEL, "revision": REV, "prefix": PREFIX, "scored": SCORED,
               "documents": [], "note":
               "development articles only; this measures the harness, not the corpus"}

    for dtype in (torch.bfloat16, torch.float32):
        name = str(dtype).rsplit(".", 1)[-1]
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REV, dtype=dtype).to("cuda:0").eval()
        for di, (title, body) in enumerate(arts):
            ids = tok(body, return_tensors="pt").input_ids[:, :PREFIX + 1 + SCORED]
            if ids.shape[1] < PREFIX + 1 + SCORED:
                continue
            ids = ids.cuda()
            cached, full = per_token(model, ids)
            d = np.abs(cached - full)
            rec = {
                "dtype": name, "doc": di, "title": title,
                "mean_abs_token_diff": float(d.mean()),
                "median_abs_token_diff": float(np.median(d)),
                "max_abs_token_diff": float(d.max()),
                "n_tokens_over_0p1": int((d > 0.1).sum()),
                "mean_nll_cached": float(cached.mean()),
                "mean_nll_full": float(full.mean()),
                "mean_nll_gap": float(abs(cached.mean() - full.mean())),
                "shift_test_mean_abs_diff": shift_test(cached, full),
            }
            summary["documents"].append(rec)
            jl.write(json.dumps({**rec, "per_token_abs_diff":
                                 [round(float(v), 6) for v in d]}) + "\n")
            jl.flush()
            st = rec["shift_test_mean_abs_diff"]
            print(f"{name:<9} doc{di}: mean |diff| {d.mean():.3e}  "
                  f"gap {rec['mean_nll_gap']:.3e}  >0.1: {rec['n_tokens_over_0p1']}/256"
                  f"  best shift {st['best_shift']} "
                  f"(0:{st['0']:.3e}  -1:{st['-1']:.3e}  +1:{st['+1' if '+1' in st else '1']:.3e})",
                  flush=True)
        del model
        torch.cuda.empty_cache()
    jl.close()

    bf = [d for d in summary["documents"] if d["dtype"] == "bfloat16"]
    f32 = [d for d in summary["documents"] if d["dtype"] == "float32"]
    summary["verdict"] = {
        "bf16_mean_gap": float(np.mean([d["mean_nll_gap"] for d in bf])),
        "fp32_mean_gap": float(np.mean([d["mean_nll_gap"] for d in f32])),
        "fp32_shrinks_gap_by": (float(np.mean([d["mean_nll_gap"] for d in bf]))
                                / max(float(np.mean([d["mean_nll_gap"] for d in f32])), 1e-30)),
        "any_shift_improves": any(d["shift_test_mean_abs_diff"]["best_shift"] != "0"
                                  for d in summary["documents"]),
        "reading": "a position error would make a non-zero shift the best one and "
                   "would not vanish in fp32; kernel-selection noise does the "
                   "opposite",
    }
    (Path(a.out) / "alignment_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nverdict: {json.dumps(summary['verdict'], indent=1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
