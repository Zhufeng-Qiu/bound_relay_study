"""Quality lane — Figure 3. Deliberately independent of any GPU session.

Semantics: **prefill, transport, decode.** A prefix is prefilled to produce a KV
cache; that cache is round-tripped through the codec at a given ε; the
continuation is then scored using the *reconstructed* cache. This is the
disaggregated-serving scenario — KV computed on one worker, moved, consumed on
another — rather than "compress on every cache update", which no deployment does.

Reported as a delta in negative log-likelihood over the continuation tokens
against an uncompressed run on the same inputs. Paired by construction: the same
prefix, the same continuation, the same order, one variable.

This exists because the paired-WER result depends on the runtime integration,
which is a stretch goal. Making the entire error-to-quality story hostage to the
riskiest step would be a planning error, since error propagation to a downstream
metric is the part of the work closest to Sian Jin's own line. So it has a path
that cannot fail: it runs on a laptop.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from boundrelay.codec import Allocation, CodecConfig, reference


def kv_layers(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer (keys, values) across transformers Cache API versions."""
    def pick(obj, names):
        for n in names:
            v = getattr(obj, n, None)
            if torch.is_tensor(v):
                return v
        raise AttributeError(f"no tensor among {names} on {type(obj).__name__}")

    if hasattr(cache, "layers"):
        return [(pick(L, ("keys", "key_cache", "k")),
                 pick(L, ("values", "value_cache", "v"))) for L in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def roundtrip_cache_(cache, eps: float, alloc: Allocation) -> dict:
    """Compress and reconstruct every layer's cache in place.

    Returns the realised compression ratio and worst-case error, so the quality
    number is always reported next to what was paid for it.
    """
    tot_o = tot_c = 0
    worst = 0.0
    for k, v in kv_layers(cache):
        for t in (k, v):
            x = t.detach().to(torch.bfloat16).cpu()
            flat = x.permute(0, 2, 1, 3).reshape(-1, x.shape[-1]).contiguous()
            buf, st = reference.encode(flat, CodecConfig(eps=eps, allocation=alloc))
            rec = reference.decode(buf)
            worst = max(worst, reference.max_abs_error(flat, rec))
            back = rec.reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[3]) \
                      .permute(0, 2, 1, 3).contiguous()
            t.copy_(back.to(t.dtype).to(t.device))
            tot_o += st.original_bytes
            tot_c += st.compressed_bytes
    return {"ratio": tot_o / max(tot_c, 1), "max_error": worst}


@torch.inference_mode()
def _nll(model, ids: torch.Tensor, prefix_len: int, eps: float | None,
         alloc: Allocation) -> tuple[float, dict]:
    """Mean NLL over continuation tokens, optionally through a compressed cache."""
    pre, cont = ids[:, :prefix_len], ids[:, prefix_len:]
    out = model(pre, use_cache=True)
    stats = {"ratio": 1.0, "max_error": 0.0}
    if eps is not None:
        stats = roundtrip_cache_(out.past_key_values, eps, alloc)

    logits = model(cont, past_key_values=out.past_key_values, use_cache=True).logits
    # predict token t from position t-1; first continuation token is predicted
    # by the last prefix logit, which lives in the prefill output
    shift = torch.cat([out.logits[:, -1:], logits[:, :-1]], dim=1).float()
    nll = torch.nn.functional.cross_entropy(
        shift.reshape(-1, shift.shape[-1]), cont.reshape(-1), reduction="mean")
    return float(nll), stats


def run(model_id: str, eps_values: list[float], out: Path,
        device: str = "mps", n_seq: int = 12, seq_len: int = 256,
        prefix_len: int = 128, alloc: Allocation = Allocation.PER_CHANNEL) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device).eval()

    corpus = _corpus()
    enc = tok(corpus, return_tensors="pt").input_ids[0]
    seqs = [enc[i * seq_len:(i + 1) * seq_len].unsqueeze(0)
            for i in range(min(n_seq, enc.numel() // seq_len))]
    print(f"model={model_id} device={device} seqs={len(seqs)} "
          f"prefix={prefix_len} cont={seq_len - prefix_len}", flush=True)

    res = {"model": model_id, "allocation": alloc.value, "eps": [],
           "ppl_delta": [], "nll_delta": [], "ratio": [], "max_error": [],
           "n_seq": len(seqs), "prefix_len": prefix_len, "seq_len": seq_len}

    base = [_nll(model, s.to(device), prefix_len, None, alloc)[0] for s in seqs]
    base_mean = sum(base) / len(base)
    res["baseline_nll"] = base_mean
    res["baseline_ppl"] = float(torch.tensor(base_mean).exp())
    print(f"baseline nll={base_mean:.4f} ppl={res['baseline_ppl']:.3f}", flush=True)

    for eps in eps_values:
        nlls, ratios, errs = [], [], []
        for s in seqs:
            n, st = _nll(model, s.to(device), prefix_len, eps, alloc)
            nlls.append(n); ratios.append(st["ratio"]); errs.append(st["max_error"])
        m = sum(nlls) / len(nlls)
        res["eps"].append(eps)
        res["nll_delta"].append(m - base_mean)
        res["ppl_delta"].append(float(torch.tensor(m).exp() - torch.tensor(base_mean).exp()))
        res["ratio"].append(sum(ratios) / len(ratios))
        res["max_error"].append(max(errs))
        print(f"  eps={eps:<6g} ratio={res['ratio'][-1]:5.2f}x  "
              f"nll+{res['nll_delta'][-1]:+.5f}  ppl+{res['ppl_delta'][-1]:+.4f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return res


def _corpus() -> str:
    """WikiText-2 test split — the standard perplexity corpus.

    A first attempt used one hand-written paragraph repeated to length. The model
    memorised it after the first copy and baseline perplexity came out at 1.03,
    which leaves no headroom for a compression effect to show in: the metric was
    measuring memorisation, not reconstruction quality. Natural, non-repeating
    text is a requirement here, not a nicety.
    """
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(t for t in ds["text"] if len(t.strip()) > 64)
