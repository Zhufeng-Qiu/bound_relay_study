"""Formal 2x2 interaction contrast, plus the c_V boundary.

Two things the earlier grid could not support:

* The interaction was reported as "7.7x the additive prediction", which is not a
  standard estimator. The contrast is
  ``NLL(K,V) - NLL(K only) - NLL(V only) + NLL(raw)``, bootstrapped over documents.
* `K-only at c=0.005` was never measured, so the claim that this bound is
  individually indistinguishable from lossless went beyond the data. It is
  measured here.

And the boundary: V is safe at 0.03 and catastrophic at 0.10, with nothing in
between. This walks it.
"""
from __future__ import annotations
import ctypes, json, re, statistics as st, sys
from pathlib import Path
sys.path.insert(0, "/workspace/boundrelay")
import numpy as np, torch
from boundrelay.codec.cuszp_bridge import _MANGLED, lib
from boundrelay.bench.quality import kv_layers

MODEL, REV = "Qwen/Qwen3-1.7B", "b9352fbb8ce704292730cf54b3b1dceb2a808738"
PREFIX, CONT, NDOC, MODE = 1024, 128, 16, "fixed"

# (c_K, c_V); None means that kind ships raw
CONFIGS = [
    (None, None),                                   # baseline
    (0.005, None), (None, 0.10), (0.005, 0.10),     # contrast cell A
    (0.10,  None),                (0.10,  0.10),    # contrast cell B (V-only shared)
    (0.01, 0.03), (0.01, 0.04), (0.01, 0.05),       # c_V boundary walk
    (0.01, 0.06), (0.01, 0.08), (0.01, 0.10),
]

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
raw = "\n".join(load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")["text"])
parts = re.split(r"\n\s=\s([^=][^\n]*?)\s=\s\n", raw)
docs = [(parts[i].strip(), parts[i+1]) for i in range(1, len(parts)-1, 2)
        if len(parts[i+1]) >= 12000][:NDOC]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV,
                                             dtype=torch.bfloat16).to("cuda:0").eval()
L = lib()


def rt(t, eps):
    f = t.float().contiguous().reshape(-1)
    # zeroed, both of them: cuSZp leaves elements it expects to be zero unwritten,
    # and reads past cmpSize. With torch.empty the allocator's leftovers land in the
    # reconstruction, and the looser the bound the more of them there are -- 50 of 56
    # tensors at c=0.10 against 5 at c=0.01. Every earlier quality result used
    # torch.empty here.
    scr = torch.zeros(f.numel()*4+4096, dtype=torch.uint8, device="cuda:0")
    n = ctypes.c_size_t(0)
    getattr(L,_MANGLED[("compress",MODE)])(ctypes.c_void_p(f.data_ptr()),
        ctypes.c_void_p(scr.data_ptr()), ctypes.c_size_t(f.numel()),
        ctypes.byref(n), ctypes.c_float(eps), None)
    torch.cuda.synchronize(0)
    out = torch.zeros(f.numel(), dtype=torch.float32, device="cuda:0")
    getattr(L,_MANGLED[("decompress",MODE)])(ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(scr.data_ptr()), ctypes.c_size_t(f.numel()),
        ctypes.c_size_t(int(n.value)), ctypes.c_float(eps), None)
    torch.cuda.synchronize(0)
    err = float((f - out).abs().max())
    assert err <= eps * 1.001, f"cuSZp exceeded its bound: {err} > {eps}"
    return out.reshape(t.shape).to(torch.bfloat16), int(n.value), err


@torch.inference_mode()
def run(ids, cK, cV):
    pre, cont = ids[:, :PREFIX], ids[:, PREFIX:PREFIX+CONT]
    o = model(pre, use_cache=True)
    payload, worst = 0, 0.0
    for k_, v_ in kv_layers(o.past_key_values):
        for kind, t in (("k", k_), ("v", v_)):
            cc = cK if kind == "k" else cV
            if cc is None:
                payload += t.numel()*2; continue
            eps = cc * float(t.float().std())
            rec, nb, err = rt(t, eps)
            t.copy_(rec); payload += nb; worst = max(worst, err/eps)
    lg = model(cont, past_key_values=o.past_key_values, use_cache=True).logits
    sh = torch.cat([o.logits[:, -1:], lg[:, :-1]], 1).float()
    pt = torch.nn.functional.cross_entropy(sh.reshape(-1, sh.shape[-1]),
                                           cont.reshape(-1), reduction="none")
    return float(pt.sum()), int(pt.numel()), payload, worst


rows = []
for di, (title, body) in enumerate(docs):
    a = tok(body, return_tensors="pt").input_ids[0]
    if a.numel() < PREFIX+CONT: continue
    ids = a[:PREFIX+CONT].unsqueeze(0).to("cuda:0")
    for cfg in CONFIGS:
        s, n, p, w = run(ids, *cfg)
        rows.append({"doc": f"d{di:02d}", "cK": cfg[0], "cV": cfg[1],
                     "nll_sum": s, "ntok": n, "bytes": p, "err_over_eps": w})
    print(f"  d{di:02d} done", flush=True)

DOCS = sorted({r["doc"] for r in rows})
def per_doc(cfg):
    return {r["doc"]: r["nll_sum"]/r["ntok"] for r in rows
            if (r["cK"], r["cV"]) == cfg}
base = per_doc((None, None))
def agg(cfg):
    r = [x for x in rows if (x["cK"], x["cV"]) == cfg]
    return sum(x["nll_sum"] for x in r)/sum(x["ntok"] for x in r), st.mean(x["bytes"] for x in r)

out = {"configs": [list(c) for c in CONFIGS], "rows": rows, "summary": [], "contrasts": []}
b_nll, b_bytes = agg((None, None))
print(f"\nbaseline ppl {np.exp(b_nll):.3f}, payload {b_bytes/1e6:.1f} MB, {len(DOCS)} documents\n")
print(f"{'c_K':>7} {'c_V':>7} {'MB':>8} {'vs raw':>7} {'ppl':>10} {'dNLL':>9} {'95% CI':>21} {'err/eps':>8}")
print("-"*82)
for cfg in CONFIGS[1:]:
    nl, by = agg(cfg); pd = per_doc(cfg)
    d = [pd[k]-base[k] for k in DOCS]
    boot = [float(np.mean(np.random.choice(d, len(d), True))) for _ in range(4000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    w = max(r["err_over_eps"] for r in rows if (r["cK"], r["cV"]) == cfg)
    out["summary"].append({"cK": cfg[0], "cV": cfg[1], "bytes": by, "ppl": float(np.exp(nl)),
                           "dnll": nl-b_nll, "ci": [float(lo), float(hi)],
                           "significant": not (lo < 0 < hi), "max_err_over_eps": w})
    print(f"{str(cfg[0]):>7} {str(cfg[1]):>7} {by/1e6:7.1f} {by/b_bytes:6.2f}x {np.exp(nl):10.3f} "
          f"{nl-b_nll:+9.4f} [{lo:+.4f},{hi:+.4f}]{'*' if not (lo<0<hi) else ' '} {w:7.3f}")

print("\n=== 2x2 interaction contrasts: NLL(K,V) - NLL(K) - NLL(V) + NLL(raw) ===")
for cK, cV in ((0.005, 0.10), (0.10, 0.10)):
    kv, k1, v1 = per_doc((cK, cV)), per_doc((cK, None)), per_doc((None, cV))
    d = [kv[x] - k1[x] - v1[x] + base[x] for x in DOCS]
    boot = [float(np.mean(np.random.choice(d, len(d), True))) for _ in range(4000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out["contrasts"].append({"cK": cK, "cV": cV, "contrast": float(np.mean(d)),
                             "ci": [float(lo), float(hi)]})
    print(f"  c_K={cK:<6g} c_V={cV:<5g}  contrast {np.mean(d):+7.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]")
Path("/workspace/quality_contrast.json").write_text(json.dumps(out, indent=2))
print("\nwrote /workspace/quality_contrast.json")
