"""Phase D — where should the error go, at a matched byte budget?

Four static strategies over a real cuSZp round trip ending in bf16: no
compression, uniform K+V, K-only, V-only. The uncompressed half of a one-sided
strategy still ships, so its raw bytes count toward the payload -- comparing
K-only against uniform on ratio alone would credit K-only for bytes it never
saved.

Quality is the NLL of a continuation decoded against the *reconstructed* cache,
paired per document against the same document's uncompressed run. Perplexity is
exponentiated from the aggregate mean NLL; averaging per-document perplexities
would weight documents by how surprising they are rather than by their length.

Documents are the bootstrap unit, because six layers of one article are not six
observations.
"""
import ctypes, json, statistics as st, sys, time
sys.path.insert(0, "/workspace/boundrelay")
import numpy as np, torch
from pathlib import Path
from boundrelay.codec.cuszp_bridge import _MANGLED, lib
from boundrelay.bench.quality import kv_layers

MODEL = "Qwen/Qwen3-1.7B"; REV = "b9352fbb8ce704292730cf54b3b1dceb2a808738"
PREFIX, CONT, NDOC = 1024, 128, 16
C_GRID = [0.01, 0.03, 0.10]
STRATS = {"uniform": lambda k: True, "k_only": lambda k: k == "k",
          "v_only": lambda k: k == "v"}

from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from datasets import load_dataset
raw = "\n".join(load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")["text"])
parts = re.split(r"\n\s=\s([^=][^\n]*?)\s=\s\n", raw)
docs = [(parts[i].strip(), parts[i+1]) for i in range(1,len(parts)-1,2) if len(parts[i+1])>=12000][:NDOC]

tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).to("cuda:0").eval()
L = lib()

def roundtrip_(t, eps, mode="fixed"):
    f = t.float().contiguous().reshape(-1)
    scr = torch.empty(f.numel()*4+4096, dtype=torch.uint8, device="cuda:0")
    n = ctypes.c_size_t(0)
    getattr(L,_MANGLED[("compress",mode)])(ctypes.c_void_p(f.data_ptr()),
        ctypes.c_void_p(scr.data_ptr()), ctypes.c_size_t(f.numel()),
        ctypes.byref(n), ctypes.c_float(eps), None)
    torch.cuda.synchronize(0)
    out = torch.empty(f.numel(), dtype=torch.float32, device="cuda:0")
    getattr(L,_MANGLED[("decompress",mode)])(ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(scr.data_ptr()), ctypes.c_size_t(f.numel()),
        ctypes.c_size_t(int(n.value)), ctypes.c_float(eps), None)
    torch.cuda.synchronize(0)
    return out.reshape(t.shape).to(torch.bfloat16), int(n.value)

@torch.inference_mode()
def nll(ids, strat=None, c=None):
    pre, cont = ids[:, :PREFIX], ids[:, PREFIX:PREFIX+CONT]
    o = model(pre, use_cache=True)
    payload = 0
    if strat:
        for k_, v_ in kv_layers(o.past_key_values):
            for kind, t in (("k",k_), ("v",v_)):
                if STRATS[strat](kind):
                    eps = c * float(t.float().std())
                    rec, nb = roundtrip_(t, eps); t.copy_(rec); payload += nb
                else:
                    payload += t.numel()*2          # ships uncompressed
    else:
        payload = sum(t.numel()*2 for kv in kv_layers(o.past_key_values) for t in kv)
    lg = model(cont, past_key_values=o.past_key_values, use_cache=True).logits
    sh = torch.cat([o.logits[:,-1:], lg[:,:-1]], 1).float()
    per_tok = torch.nn.functional.cross_entropy(
        sh.reshape(-1, sh.shape[-1]), cont.reshape(-1), reduction="none")
    return float(per_tok.sum()), int(per_tok.numel()), payload

rows=[]
for di,(title, body) in enumerate(docs):
    ids_all = tok(body, return_tensors="pt").input_ids[0]
    if ids_all.numel() < PREFIX+CONT: continue
    ids = ids_all[:PREFIX+CONT].unsqueeze(0).to("cuda:0")
    s0,n0,p0 = nll(ids)
    rows.append({"doc":f"d{di:02d}","strategy":"none","c":None,"nll_sum":s0,"ntok":n0,"bytes":p0})
    for strat in STRATS:
        for c in C_GRID:
            s,n,p = nll(ids, strat, c)
            rows.append({"doc":f"d{di:02d}","strategy":strat,"c":c,"nll_sum":s,"ntok":n,"bytes":p})
    print(f"  d{di:02d} {title[:38]:<38} done", flush=True)

def agg(sel):
    r=[x for x in rows if sel(x)]
    return sum(x["nll_sum"] for x in r)/sum(x["ntok"] for x in r), st.mean(x["bytes"] for x in r)
base_nll, base_bytes = agg(lambda x: x["strategy"]=="none")
print(f"\nbaseline: ppl {np.exp(base_nll):.3f}, payload {base_bytes/1e6:.1f} MB, "
      f"{len(set(r['doc'] for r in rows))} documents\n")
print(f"{'strategy':>10} {'c':>6} {'payload MB':>11} {'vs raw':>8} {'ppl':>8} {'dppl':>8} {'d%':>7}")
print("-"*62)
out={"baseline_nll":base_nll,"baseline_bytes":base_bytes,"rows":rows,"summary":[]}
for strat in STRATS:
    for c in C_GRID:
        nl, by = agg(lambda x,s=strat,cc=c: x["strategy"]==s and x["c"]==cc)
        d = float(np.exp(nl)-np.exp(base_nll))
        # per-document paired deltas -> cluster bootstrap
        pd=[]
        for doc in sorted(set(r["doc"] for r in rows)):
            a=[x for x in rows if x["doc"]==doc and x["strategy"]=="none"][0]
            b=[x for x in rows if x["doc"]==doc and x["strategy"]==strat and x["c"]==c][0]
            pd.append(b["nll_sum"]/b["ntok"] - a["nll_sum"]/a["ntok"])
        boot=[float(np.mean(np.random.choice(pd,len(pd),True))) for _ in range(2000)]
        out["summary"].append({"strategy":strat,"c":c,"bytes":by,"ppl":float(np.exp(nl)),
            "dppl":d,"dnll":nl-base_nll,
            "dnll_ci":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))]})
        print(f"{strat:>10} {c:>6g} {by/1e6:10.1f} {by/base_bytes:7.2f}x {np.exp(nl):8.3f} "
              f"{d:+8.3f} {100*d/np.exp(base_nll):+6.2f}%")
json.dump(out, open("/workspace/quality_d.json","w"), indent=2)
