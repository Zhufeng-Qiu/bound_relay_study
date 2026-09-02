"""D8 — capture a real KV-cache corpus, and immediately test the allocation
hypothesis on it.

Two jobs in one pass, because the pod is billing:

1. Dump KV-cache tensors from a real decoder into a replay manifest. This is the
   second tensor family, and the bridge to disaggregated-serving KV transport —
   the object PackKV and CacheGen both work on.
2. Run the blockwise / per-channel / per-token comparison on those tensors.

The second job is the one that matters. Every ratio in the project so far comes
from Gaussian activations with outliers injected by hand, chosen because that is
what the literature says LLM activations look like. Whether real KV cache
actually carries per-channel structure is an empirical question, and until it is
answered on captured tensors, Figure 2 is a statement about a synthetic
distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boundrelay.codec import Allocation, CodecConfig, reference, serialize  # noqa: E402
from boundrelay.codec.contract import TensorSpec  # noqa: E402
from boundrelay.trace.manifest import CaptureEnv, Manifest  # noqa: E402

PROMPTS = [
    "Explain why moving bytes between two GPUs is not always faster than recomputing them.",
    "Summarise the trade-off between quantisation error and inference latency.",
    "Describe how a streaming vocoder consumes tokens from an autoregressive model.",
    "What determines whether compressing an activation tensor pays for itself?",
]
EPS_VALUES = [0.05, 0.15, 0.5]


def kv_layers(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer (keys, values) across transformers Cache API versions.

    transformers 5.x replaced the legacy tuple-of-tuples with a ``Cache`` holding
    ``.layers``; ``to_legacy_cache()`` no longer returns something subscriptable.
    Candidate attribute names are probed and accepted only when they are actually
    tensors, so a renamed accessor fails loudly here instead of silently
    capturing a bound method.
    """
    def pick(obj, names):
        for n in names:
            v = getattr(obj, n, None)
            if torch.is_tensor(v):
                return v
        raise AttributeError(f"no tensor among {names} on {type(obj).__name__}")

    if hasattr(cache, "layers"):
        return [(pick(L, ("keys", "key_cache", "k")),
                 pick(L, ("values", "value_cache", "v"))) for L in cache.layers]
    if torch.is_tensor(getattr(cache, "key_cache", [None])[0]):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(k, v) for k, v in cache]


def raw_pct(buf: bytes) -> float:
    h, body = serialize.read(buf)
    if h.n_groups == 0:
        return 100.0
    w = torch.frombuffer(bytearray(body[: h.n_groups]), dtype=torch.uint8)
    return 100.0 * float((w == 0).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--fallback", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", type=Path, default=Path("/workspace/corpus/kv"))
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = args.model
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda")
    except Exception as e:                                   # noqa: BLE001
        print(f"{model_id} unavailable ({type(e).__name__}); falling back", flush=True)
        model_id = args.fallback
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    cfgm = model.config
    print(f"model={model_id} layers={cfgm.num_hidden_layers} "
          f"kv_heads={getattr(cfgm,'num_key_value_heads','?')} "
          f"head_dim={getattr(cfgm,'head_dim', cfgm.hidden_size//cfgm.num_attention_heads)}",
          flush=True)

    man = Manifest(corpus_id="kv_cache", env=CaptureEnv(
        gpu=torch.cuda.get_device_name(0), gpu_count=1,
        driver="see environment.json", cuda=torch.version.cuda,
        torch=torch.__version__, model_id=model_id,
        model_revision="main", weight_dtype="bf16",
        sglang_omni_sha="n/a", image_digest="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        notes="KV cache captured during greedy decode; second tensor family"))
    mpath = args.out / "manifest.json"

    tensors: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for pi, prompt in enumerate(PROMPTS):
            ids = tok(prompt, return_tensors="pt").to("cuda")
            out = model.generate(**ids, max_new_tokens=args.max_new,
                                 do_sample=False, return_dict_in_generate=True)
            legacy = kv_layers(out.past_key_values)
            # Sample early / middle / late layers rather than all of them: the
            # question is whether structure exists and how it varies with depth,
            # not to mirror the whole cache onto disk.
            picks = [0, len(legacy) // 2, len(legacy) - 1]
            for li in picks:
                for kind, t in (("k", legacy[li][0]), ("v", legacy[li][1])):
                    x = t.detach().to(torch.bfloat16).cpu().contiguous()
                    name = f"p{pi}_l{li}_{kind}"
                    torch.save(x, args.out / f"{name}.pt")
                    # [batch, kv_heads, seq, head_dim] -> [tokens, head_dim]
                    flat = x.permute(0, 2, 1, 3).reshape(-1, x.shape[-1]).contiguous()
                    tensors[name] = flat
                    man.append(mpath, TensorSpec(
                        name=name, shape=tuple(x.shape), dtype=str(x.dtype),
                        contiguous=True, producer=f"layer{li}", consumer="attention",
                        family="kv_cache", bytes_per_request=x.numel() * 2,
                        extras={"kind": kind, "layer": li, "prompt": pi,
                                "flat_shape": list(flat.shape)}))
            print(f"  prompt {pi}: captured {len(picks)*2} tensors", flush=True)

    del model
    torch.cuda.empty_cache()

    # ---- the hypothesis test, on real tensors ----
    print("\n=== allocation study on captured KV cache ===", flush=True)
    study = {"model": model_id, "eps_values": EPS_VALUES,
             "allocations": [a.value for a in Allocation], "tensors": {}}
    for name, x in sorted(tensors.items()):
        study["tensors"][name] = {"shape": list(x.shape)}
        for a in Allocation:
            ratios, raws = [], []
            for eps in EPS_VALUES:
                buf, st = reference.encode(x, CodecConfig(eps=eps, allocation=a))
                err = reference.max_abs_error(x, reference.decode(buf))
                assert err <= eps + 1e-6, (name, a, eps, err)
                ratios.append(round(st.ratio, 3))
                raws.append(round(raw_pct(buf), 2))
            study["tensors"][name][a.value] = {"ratio": ratios, "raw_pct": raws}

    (args.out / "allocation_study_kv.json").write_text(json.dumps(study, indent=2))

    print(f"\n{'tensor':>14} {'alloc':>12} " +
          "  ".join(f"eps={e:<5g}" for e in EPS_VALUES))
    print("-" * 62)
    for name in sorted(study["tensors"])[:6]:
        for a in Allocation:
            r = study["tensors"][name][a.value]["ratio"]
            print(f"{name:>14} {a.value:>12} " + "  ".join(f"{v:9.2f}x" for v in r))
    print(f"\nwrote {args.out/'allocation_study_kv.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
