"""B1 -- what does compressing the KV cache cost, on articles this project has
never read?

The K/V asymmetry this project reported came from sixteen articles that were also
used to develop it. That is the finding most worth re-testing and the one least
able to carry its own weight, so it is re-run here on the thirty-two held-out
articles frozen in `heldout_manifest.json` before any of these numbers existed.

Two corrections to how the earlier version scored.

**Every scored position now uses the reconstructed cache.** The old arrangement
scored the first continuation token from the prefill's last logit -- produced
*before* the cache was touched -- so one of its 128 positions was structurally
immune to compression. Taking 257 continuation tokens and scoring the last 256
against the first 256 removes that.

**Each arm re-prefills.** Arms that modify a cache in place cannot share one, and
an arm that inherits the previous arm's reconstruction is measuring the wrong
thing twice.

Improvement is not a pass condition. A confidence interval that crosses zero says
the direction is undetermined; it does not say compression is safe.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/boundrelay")
from boundrelay.codec.cuszp_bridge import _MANGLED, TAU, lib  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
REV = "b9352fbb8ce704292730cf54b3b1dceb2a808738"
MODE = "fixed"
PREFIX, SCORED = 1024, 256
ARMS = [("raw", None, None), ("K_only", 0.10, None),
        ("V_only", None, 0.10), ("K_and_V", 0.10, 0.10)]


def kv_layers(pkv):
    if hasattr(pkv, "layers"):
        return [(l.keys, l.values) for l in pkv.layers]
    return [(k, v) for k, v in pkv]


def roundtrip(L, t: torch.Tensor, eps: float) -> tuple[torch.Tensor, int, dict]:
    """One tensor through the codec, with the full three-way error accounting."""
    f = t.float().contiguous().reshape(-1)
    n = f.numel()
    cmp_ = torch.zeros(n * 4 + 4096, dtype=torch.uint8, device=f.device)
    dec = torch.zeros(n, dtype=torch.float32, device=f.device)
    m = ctypes.c_size_t(0)
    getattr(L, _MANGLED[("compress", MODE)])(
        ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(cmp_.data_ptr()),
        ctypes.c_size_t(n), ctypes.byref(m), ctypes.c_float(eps), None)
    torch.cuda.synchronize()
    nb = int(m.value)
    getattr(L, _MANGLED[("decompress", MODE)])(
        ctypes.c_void_p(dec.data_ptr()), ctypes.c_void_p(cmp_.data_ptr()),
        ctypes.c_size_t(n), ctypes.c_size_t(nb), ctypes.c_float(eps), None)
    torch.cuda.synchronize()
    z = dec.to(torch.bfloat16)
    x64, y64, z64 = f.double().cpu(), dec.double().cpu(), z.double().cpu()
    finite = bool(torch.isfinite(dec).all() and torch.isfinite(z).all())
    e32 = float((y64 - x64).abs().max())
    ebf = float((z64 - x64).abs().max())
    rnd = float((z64 - y64).abs().max())
    # The pre-declared fp32 criterion is `E32 <= eps + TAU` with TAU absolute, and
    # B0 measured why that is the wrong shape. Across its 448 combinations the
    # codec's error reaches 1.000000578 x eps -- it does exceed the bound, by a few
    # float32 ulps *of eps*, because the quantiser works from a float32 1/eps. That
    # excess is relative, so at the small epsilons B0 saw (up to 2.2) it stays under
    # 1e-6 absolute and passes, and at a larger eps the same relative excess crosses
    # it and fails. B1 hit that at eps = 1.465.
    #
    # TAU is not widened here. Widening a threshold because it failed is the move
    # this round's protocol exists to prevent, and doing it silently would be worse
    # than the original loose `eps * 1.001`. Instead the fp32 margin becomes a
    # *recorded measurement* on every tensor, and the abort is reserved for damage:
    # a non-finite value, or an excess three orders of magnitude beyond the observed
    # float-noise floor, which is what real corruption looks like (the truncated
    # payload in B0 came back at 245 x eps).
    #
    # The consequence is stated rather than hidden: this run does not certify
    # `E32 <= eps + 1e-6`. It reports the largest margin it saw.
    ratio = e32 / eps if eps > 0 else float("inf")
    if not finite or ratio > 1.001:
        raise SystemExit(f"codec returned damage: E32={e32:g} eps={eps:g} "
                         f"E32/eps={ratio:g} finite={finite}")
    return z.reshape(t.shape), nb, {"eps": eps, "E32": e32, "Ebf": ebf, "R": rnd,
                                    "E32_over_eps": ratio,
                                    "within_eps_plus_tau": e32 <= eps + TAU}


@torch.inference_mode()
def score(model, L, ids: torch.Tensor, cK, cV) -> dict:
    """Prefill, replace the prefix cache, then score 256 teacher-forced positions.

    Only the prefix K/V are replaced. K/V the continuation itself produces stay
    raw, so this measures the cost of one cache reconstruction rather than of
    compressing every state the model goes on to make.
    """
    pre = ids[:, :PREFIX]
    cont_in = ids[:, PREFIX:PREFIX + SCORED]
    cont_tgt = ids[:, PREFIX + 1:PREFIX + 1 + SCORED]
    o = model(pre, use_cache=True)

    payload, raw_bytes, tensors = 0, 0, []
    for li, (k_, v_) in enumerate(kv_layers(o.past_key_values)):
        for kind, t in (("k", k_), ("v", v_)):
            c = cK if kind == "k" else cV
            raw_bytes += t.numel() * 2
            if c is None:
                # the uncompressed half of a one-sided arm still has to ship
                payload += t.numel() * 2
                tensors.append({"layer": li, "kind": kind, "c": None,
                                "payload_bytes": t.numel() * 2,
                                "raw_bytes": t.numel() * 2})
                continue
            eps = float(torch.tensor(c * float(t.float().std(correction=1)),
                                     dtype=torch.float32))
            rec, nb, e = roundtrip(L, t, eps)
            t.copy_(rec)
            payload += nb
            tensors.append({"layer": li, "kind": kind, "c": c, "payload_bytes": nb,
                            "raw_bytes": t.numel() * 2, **e})

    logits = model(cont_in, past_key_values=o.past_key_values).logits
    nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                          cont_tgt.reshape(-1), reduction="none")
    finite = bool(torch.isfinite(logits).all() and torch.isfinite(nll).all())
    return {"nll_sum": float(nll.sum()), "ntok": int(nll.numel()),
            "mean_nll": float(nll.mean()), "finite": finite,
            "payload_bytes": payload, "raw_bytes": raw_bytes,
            "per_token_nll": [round(v, 6) for v in nll.tolist()],
            "tensors": tensors}


@torch.inference_mode()
def preflight(model, tok, dev_bodies: list[str]) -> dict:
    """Two checks on development articles, before any held-out article is touched.

    1. Replacing the cache with a bit-identical copy must not move the score. If it
       does, the harness is wrong and nothing downstream means anything.
    2. A full forward and a cached forward should agree on the same positions. bf16
       kernels differ with sequence length, so a gap here is a prompt to
       investigate alignment, not a proof of error.
    """
    out = []
    for i, body in enumerate(dev_bodies):
        ids = tok(body, return_tensors="pt").input_ids[:, :PREFIX + 1 + SCORED].cuda()
        if ids.shape[1] < PREFIX + 1 + SCORED:
            continue
        pre, cont_in = ids[:, :PREFIX], ids[:, PREFIX:PREFIX + SCORED]
        cont_tgt = ids[:, PREFIX + 1:PREFIX + 1 + SCORED]

        def cached_nll(replace: bool):
            o = model(pre, use_cache=True)
            if replace:
                for k_, v_ in kv_layers(o.past_key_values):
                    k_.copy_(k_.clone())
                    v_.copy_(v_.clone())
            lg = model(cont_in, past_key_values=o.past_key_values).logits
            return F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]),
                                   cont_tgt.reshape(-1), reduction="none")

        a, b = cached_nll(False), cached_nll(True)
        replace_max = float((a - b).abs().max())

        full = model(ids[:, :PREFIX + SCORED]).logits[:, PREFIX:PREFIX + SCORED]
        fn = F.cross_entropy(full.float().reshape(-1, full.shape[-1]),
                             cont_tgt.reshape(-1), reduction="none")
        align = abs(float(fn.mean()) - float(a.mean()))
        out.append({"dev_doc": i, "replace_max_token_nll_diff": replace_max,
                    "replace_ok": replace_max <= 1e-6,
                    "full_vs_cached_mean_diff": align,
                    "alignment_investigate": align > 1e-3,
                    "finite": bool(torch.isfinite(a).all() and torch.isfinite(fn).all())})
        print(f"  dev{i}: replace {replace_max:.2e} "
              f"{'ok' if replace_max <= 1e-6 else 'FAIL'} | "
              f"full-vs-cached {align:.2e}"
              f"{'  <- investigate' if align > 1e-3 else ''}", flush=True)
    return {"checks": out, "all_replace_ok": all(c["replace_ok"] for c in out)}


def bootstrap(delta: dict[str, list[float]], docs: list[str], seed: int,
              n: int = 5000) -> dict:
    """Resample articles, carrying every arm for an article together.

    The independent unit is the article, not the token and not the tensor. Keeping
    the arms together is what makes the interval a paired one.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(docs))
    out = {}
    for arm, d in delta.items():
        arr = np.asarray(d)
        means = np.array([arr[rng.choice(idx, len(idx), replace=True)].mean()
                          for _ in range(n)])
        lo, hi = np.percentile(means, [2.5, 97.5])
        out[arm] = {"delta_nll": float(arr.mean()),
                    "ci95": [float(lo), float(hi)],
                    "ppl_pct": 100 * (float(np.exp(arr.mean())) - 1),
                    "ppl_pct_ci95": [100 * (float(np.exp(lo)) - 1),
                                     100 * (float(np.exp(hi)) - 1)],
                    "crosses_zero": bool(lo < 0 < hi),
                    "n_docs": len(arr),
                    "n_docs_worse": int((arr > 0).sum())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/workspace/boundrelay/results/public/"
                                          "protocol_2026_09_17/heldout_manifest.json")
    ap.add_argument("--out", default="/workspace/out/b1")
    ap.add_argument("--skip-preflight", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    man = json.loads(Path(a.manifest).read_text())
    want = {d["title"]: d for d in man["documents"]}
    print(f"held-out articles frozen: {man['frozen_n']}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, dtype=torch.bfloat16).to("cuda:0").eval()
    L = lib()

    def split_articles(split):
        raw = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                     split=split)["text"])
        p = re.split(r"\n\s=\s([^=][^\n]*?)\s=\s\n", raw)
        return [(p[i].strip(), p[i + 1]) for i in range(1, len(p) - 1, 2)]

    if not a.skip_preflight:
        print("\npreflight on development articles ...", flush=True)
        dev = [b for _, b in split_articles("train") if len(b) >= 12000][:2]
        pre = preflight(model, tok, dev)
        (out / "b1_preflight.json").write_text(json.dumps(pre, indent=2))
        if not pre["all_replace_ok"]:
            raise SystemExit("cache replacement changes the score; harness is wrong")

    bodies = {t: b for t, b in split_articles("validation") if t in want}
    missing = set(want) - set(bodies)
    if missing:
        raise SystemExit(f"frozen articles not found in the corpus: {sorted(missing)}")

    jl = (out / "quality_documents.jsonl").open("w")
    delta: dict[str, list[float]] = {a_[0]: [] for a_ in ARMS if a_[0] != "raw"}
    raw_nll: dict[str, float] = {}
    bytes_tot: dict[str, list[int]] = {a_[0]: [] for a_ in ARMS}
    raw_tot: list[int] = []
    docs: list[str] = []
    per_arm_all: list[dict] = []
    t0 = time.perf_counter()

    for title, meta in sorted(want.items(), key=lambda kv: kv[1]["doc"]):
        ids = tok(bodies[title], return_tensors="pt").input_ids
        ids = ids[:, :PREFIX + 1 + SCORED].cuda()
        if ids.shape[1] < PREFIX + 1 + SCORED:
            raise SystemExit(f"{meta['doc']} too short at run time: {ids.shape[1]}")
        per_arm = {}
        for arm, cK, cV in ARMS:
            r = score(model, L, ids, cK, cV)
            if not r["finite"] or r["ntok"] != SCORED:
                raise SystemExit(f"{meta['doc']} {arm}: finite={r['finite']} "
                                 f"ntok={r['ntok']}")
            per_arm[arm] = r
            per_arm_all.append(r)
            bytes_tot[arm].append(r["payload_bytes"])
        raw_tot.append(per_arm["raw"]["raw_bytes"])
        docs.append(meta["doc"])
        raw_nll[meta["doc"]] = per_arm["raw"]["mean_nll"]
        for arm in delta:
            delta[arm].append(per_arm[arm]["mean_nll"] - per_arm["raw"]["mean_nll"])
        jl.write(json.dumps({"doc": meta["doc"], "title": title,
                             "corpus_index": meta["corpus_index"],
                             "arms": per_arm}) + "\n")
        jl.flush()
        d = {k: per_arm[k]["mean_nll"] - per_arm["raw"]["mean_nll"] for k in delta}
        print(f"  {meta['doc']} {title[:34]:<34} raw {per_arm['raw']['mean_nll']:.4f} "
              + "  ".join(f"{k} {v:+.4f}" for k, v in d.items())
              + f"   ({time.perf_counter()-t0:.0f}s)", flush=True)
    jl.close()

    stats = bootstrap(delta, docs, json.loads(
        Path(a.manifest).with_name("protocol.json").read_text())["seeds"]["bootstrap"])
    margins = [t["E32_over_eps"] for r in per_arm_all for t in r["tensors"]
               if "E32_over_eps" in t]
    n_over_tau = sum(1 for r in per_arm_all for t in r["tensors"]
                     if "within_eps_plus_tau" in t and not t["within_eps_plus_tau"])
    summary = {"n_documents": len(docs), "documents": docs,
               "fp32_margin": {
                   "max_E32_over_eps": max(margins) if margins else None,
                   "n_compressed_tensors": len(margins),
                   "n_exceeding_eps_plus_tau": n_over_tau,
                   "tau": TAU,
                   "note": "the pre-declared criterion E32 <= eps + TAU (absolute) is "
                           "not met by every tensor. The excess is relative -- a few "
                           "float32 ulps of eps -- so it crosses an absolute tolerance "
                           "only at large eps. TAU was not widened; this run does not "
                           "certify the absolute criterion and reports the margin it "
                           "measured instead."},
               "scored_tokens_per_arm": SCORED, "arms": [a_[0] for a_ in ARMS],
               "delta_vs_raw": stats,
               "payload_over_raw": {k: sum(v) / sum(raw_tot)
                                    for k, v in bytes_tot.items()},
               "note": "payload counts the uncompressed side of a one-sided arm; "
                       "metadata is not serialised, so these are tensor payload "
                       "bytes, not a recoverable file size",
               "elapsed_s": time.perf_counter() - t0}
    (out / "quality_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'arm':<10} {'dNLL':>9} {'95% CI':>22} {'PPL %':>9} {'bytes/raw':>10} "
          f"{'worse':>7}")
    for arm, s in stats.items():
        print(f"{arm:<10} {s['delta_nll']:+9.4f} "
              f"[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] {s['ppl_pct']:+8.2f}% "
              f"{summary['payload_over_raw'][arm]:10.3f} "
              f"{s['n_docs_worse']:>3}/{s['n_docs']}"
              + ("   CI crosses 0" if s["crosses_zero"] else ""))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
