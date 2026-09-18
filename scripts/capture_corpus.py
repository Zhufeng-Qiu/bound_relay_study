"""B0 — capture the frozen KV corpus.

Replaces the earlier capture, which had three problems that would have propagated
into every downstream number:

* It used ``generate(max_new_tokens=64)`` from short prompts. That produces a
  decode-time cache, not the prefill cache a disaggregated transfer actually
  moves, and the two are different distributions. This uses one
  ``model(input_ids, use_cache=True)`` call.
* It fell back to a different model on failure and pinned nothing. A silent
  substitution mid-capture would poison the corpus, so revisions are pinned and
  failure is fatal.
* It treated tensors as independent samples. They are not: six layers times two
  kinds times four lengths all come from one document. Documents are the
  independent unit and are recorded as such, so splits can group by them.

Documents are real WikiText-2 articles, cut on the corpus's own ``= Title =``
markers. Concatenating the corpus and slicing it into equal lengths would
manufacture "documents" that share content and defeat the grouping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "b9352fbb8ce704292730cf54b3b1dceb2a808738"   # pinned: no silent drift
LENGTHS = [64, 256, 1024, 2048]
CHAR_LAYERS = [0, 5, 10, 14, 20, 27]


def articles(min_chars: int = 12000) -> list[tuple[str, str]]:
    """Real WikiText-2 articles, split on the corpus's own headings."""
    from datasets import load_dataset

    raw = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                 split="train")["text"])
    parts = re.split(r"\n\s=\s([^=][^\n]*?)\s=\s\n", raw)
    out = []
    for i in range(1, len(parts) - 1, 2):
        title, body = parts[i].strip(), parts[i + 1]
        if len(body) >= min_chars:
            out.append((title, body))
    return out


def stats(x: torch.Tensor) -> dict:
    """Per-tensor statistics. These are the RQ1 features, so they are recorded
    for every observation even though most tensors are never persisted."""
    f = x.float()
    flat = f.reshape(-1)
    absf = flat.abs()
    sd = float(flat.std())
    # adjacent differences along the last (channel) axis -- the axis a predictor
    # would exploit if the structure were there
    d = (f[..., 1:] - f[..., :-1]).reshape(-1)
    return {
        "std": sd,
        "mean": float(flat.mean()),
        "range": float(flat.max() - flat.min()),
        "absmax": float(absf.max()),
        "outlier_rate": float((absf > 4 * sd).float().mean()) if sd > 0 else 0.0,
        "near_zero_frac": float((absf < 0.01 * sd).float().mean()) if sd > 0 else 1.0,
        "adjdiff_std": float(d.std()),
        "adjdiff_over_std": float(d.std() / sd) if sd > 0 else 0.0,
        "nonfinite": int((~torch.isfinite(flat)).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/workspace/corpus_v2"))
    ap.add_argument("--docs", type=int, default=24)
    ap.add_argument("--full-cache-docs", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    ap.add_argument("--full-cache-lengths", type=int, nargs="+", default=None,
                    help="persist a full 28-layer cache at each of these lengths. "
                         "Default: the longest requested length only, which is what "
                         "this script used to do unconditionally.")
    ap.add_argument("--smoke", action="store_true", help="local shape/contract check only")
    ap.add_argument("--cuszp", action="store_true",
                    help="measure cuSZp ratio inline, while each tensor is still in hand")
    ap.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.03, 0.10])
    ap.add_argument("--modes", nargs="+", default=["plain", "outlier", "fixed"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16).to(args.device).eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    print(f"{MODEL_ID}@{MODEL_REVISION} · transformers {transformers.__version__} · "
          f"{n_layers} layers · {cfg.num_key_value_heads} kv heads · device {args.device}",
          flush=True)

    arts = articles()
    print(f"{len(arts)} candidate articles; taking {args.docs}", flush=True)
    if len(arts) < args.docs:
        raise SystemExit(f"only {len(arts)} long-enough articles; refusing to pad")

    from boundrelay.bench.quality import kv_layers

    manifest = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "lengths": args.lengths, "char_layers": CHAR_LAYERS, "n_layers": n_layers,
        "capture": "single prefill, model(input_ids, use_cache=True)",
        "documents": [], "observations": [],
    }
    full_docs = set(range(args.full_cache_docs))   # pre-registered, first N
    full_lengths = set(args.full_cache_lengths or [max(args.lengths)])

    with torch.inference_mode():
        for di, (title, body) in enumerate(arts[: args.docs]):
            ids_all = tok(body, return_tensors="pt").input_ids[0]
            if ids_all.numel() < max(args.lengths):
                continue
            manifest["documents"].append({
                "doc": f"d{di:02d}", "title": title,
                "sha1": hashlib.sha1(body.encode()).hexdigest()[:16],
                "n_tokens_available": int(ids_all.numel()),
                "full_cache": di in full_docs,
            })
            for L in args.lengths:
                ids = ids_all[:L].unsqueeze(0).to(args.device)
                assert ids.shape[1] == L, f"got {ids.shape[1]} tokens, wanted {L}"
                kv = kv_layers(model(ids, use_cache=True).past_key_values)
                assert len(kv) == n_layers
                for li in CHAR_LAYERS:
                    for kind, t in (("k", kv[li][0]), ("v", kv[li][1])):
                        x = t.detach().to(torch.bfloat16).cpu()
                        assert x.shape[2] == L, f"cache seq {x.shape[2]} != {L}"
                        s = stats(x)
                        obs = {
                            "doc": f"d{di:02d}", "layer": li, "kind": kind, "seq_len": L,
                            "shape": list(x.shape), "contiguous": bool(x.is_contiguous()),
                            "codec_dims_native": [x.shape[1], x.shape[2], x.shape[3]],
                            "bf16_bytes": x.numel() * 2, **s,
                        }
                        if args.cuszp and s["std"] > 0:
                            # Measured while the tensor is in hand: the corpus is not
                            # persisted, so a ratio not taken now cannot be taken later.
                            from boundrelay.codec.cuszp_bridge import roundtrip
                            obs["cuszp"] = {}
                            for cc in args.c_grid:
                                eps = cc * s["std"]
                                for mode in args.modes:
                                    r = roundtrip(x, eps, mode, want_decode=True)
                                    obs["cuszp"][f"c{cc:g}_{mode}"] = {
                                        "eps_abs": eps, "cmp_bytes": r["cmp_bytes"],
                                        "ratio": r["ratio_vs_bf16"],
                                        "max_error_fp32": r["max_error_fp32"],
                                        "max_error_bf16": r["max_error_bf16"],
                                        "within_eps_fp32": r["within_eps_fp32"],
                                    }
                        manifest["observations"].append(obs)
                # Previously `L == max(args.lengths)` -- so asking for [1024, 2048]
                # silently produced only the 2048 cache, and a directory named for
                # one length could hold another. B2 needs both lengths, and a cache
                # mislabelled by length is a wrong answer that looks like a right one.
                if di in full_docs and L in full_lengths:
                    fc = args.out / f"fullcache_d{di:02d}_L{L}"
                    fc.mkdir(exist_ok=True)
                    for li in range(n_layers):
                        for kind, t in (("k", kv[li][0]), ("v", kv[li][1])):
                            torch.save(t.detach().to(torch.bfloat16).cpu(),
                                       fc / f"l{li:02d}_{kind}.pt")
                    print(f"  d{di:02d} full cache: {2*n_layers} tensors", flush=True)
                del kv
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            print(f"  d{di:02d} {title[:44]:<44} "
                  f"{len(manifest['observations'])} obs", flush=True)
            if args.smoke and di >= 1:
                break

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest['documents'])} documents, "
          f"{len(manifest['observations'])} observations -> {args.out/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
