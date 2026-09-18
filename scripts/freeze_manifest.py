"""Freeze the experiment list before any of it runs.

Everything a later result could be accused of having been chosen after the fact
is decided here and written down: which articles count as already seen, which
held-out articles B1 will score, the seeds, and the acceptance thresholds.

The one rule this file exists to enforce: **the held-out list is frozen before a
single compressed-quality number is produced on it.** If fewer than the target
qualify, the count is frozen here too, at whatever it is, rather than after
someone has seen how sixteen articles came out.

"Not used in this project's development" is what the exclusion establishes. It is
not "the model never saw it" -- WikiText-2 is public and Qwen3 was trained before
any of this ran. Split membership is not evidence of independence either, because
earlier phases of this project read from train *and* test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3-1.7B"
REV = "b9352fbb8ce704292730cf54b3b1dceb2a808738"
MIN_CHARS = 12_000          # the threshold every earlier phase used
MIN_TOKENS = 1281           # 1024 prefill + 257 continuation
TARGET = 32
FLOOR = 16
SEED_RUN = 20260917
SEED_BOOTSTRAP = 20260918
TAU = 1e-6
DELTA_NLL_TOLERANCE = 0.01  # declared here, before results, or not used at all


def h16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def articles(split: str) -> list[tuple[str, str]]:
    """WikiText articles, cut on the corpus's own `= Title =` markers.

    The same split regex every earlier phase used, so that "already seen" is
    decided by the same notion of an article that produced the seen set.
    """
    from datasets import load_dataset
    raw = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                 split=split)["text"])
    parts = re.split(r"\n\s=\s([^=][^\n]*?)\s=\s\n", raw)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def seen_documents() -> dict:
    """Every article this project has read while developing or tuning.

    Sources, and why each counts:

    * the 24-document B0 corpus (train) -- every characterisation number
    * the first 16 train articles over the character threshold -- the quality
      experiments, including the K/V asymmetry this round is re-testing
    * the four pilot prompts p0-p3 -- the superseded decode-time capture

    The quality set is reconstructed by re-running its selection rule rather than
    read from a file, because it was never written down as a list. That is exactly
    the bookkeeping gap this manifest exists to close.
    """
    b0 = json.loads(Path("results/public/b0_corpus/manifest.json").read_text())
    seen = {}
    for d in b0["documents"]:
        seen[d["title"]] = {"title": d["title"], "body_sha1_16": d["sha1"],
                            "split": "train", "used_by": ["b0_corpus"]}

    train = articles("train")
    quality = [(t, b) for t, b in train if len(b) >= MIN_CHARS][:16]
    for t, b in quality:
        e = seen.setdefault(t, {"title": t, "body_sha1_16": h16(b),
                                "split": "train", "used_by": []})
        e["used_by"].append("quality_contrast")
    return {"n": len(seen), "documents": sorted(seen.values(), key=lambda d: d["title"]),
            "note": "Articles read during this project's development. Excluding them "
                    "establishes 'not used in development', not 'unseen by the model'."}


def build_heldout(seen: dict, tok) -> dict:
    """Validation articles that qualify, in corpus order, first TARGET of them."""
    seen_titles = {d["title"] for d in seen["documents"]}
    seen_bodies = {d["body_sha1_16"] for d in seen["documents"]}

    picked, rejected, bodies_taken = [], [], set()
    for idx, (title, body) in enumerate(articles("validation")):
        bh = h16(body)
        norm = h16(re.sub(r"\s+", " ", body).strip())
        why = None
        if len(body) < MIN_CHARS:
            why = "below character threshold"
        elif title in seen_titles:
            why = "title used in development"
        elif bh in seen_bodies:
            why = "body used in development"
        elif bh in bodies_taken:
            why = "duplicate of an article already picked"
        if why is None:
            ids = tok(body, return_tensors=None)["input_ids"]
            if len(ids) < MIN_TOKENS:
                why = f"only {len(ids)} tokens, needs {MIN_TOKENS}"
            else:
                bodies_taken.add(bh)
                picked.append({
                    "doc": f"h{len(picked):02d}", "title": title, "split": "validation",
                    "corpus_index": idx, "n_chars": len(body),
                    "n_tokens_available": len(ids),
                    "body_sha1_16": bh, "body_normalised_sha1_16": norm,
                    "prefix_1281_token_sha1_16": h16(",".join(map(str, ids[:MIN_TOKENS]))),
                })
                if len(picked) >= TARGET:
                    break
                continue
        rejected.append({"title": title, "corpus_index": idx, "reason": why})

    return {"frozen_n": len(picked), "target": TARGET, "floor": FLOOR,
            "selection_rule": f"WikiText-2 validation, corpus order, body >= "
                              f"{MIN_CHARS} chars and >= {MIN_TOKENS} tokens under "
                              f"the pinned tokenizer, not used in development, "
                              f"no duplicate bodies; first {TARGET}",
            "documents": picked, "rejected": rejected[:80]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/public/protocol_2026_09_17")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)

    print("enumerating articles this project has already read ...", flush=True)
    seen = seen_documents()
    (out / "seen_documents.json").write_text(json.dumps(seen, indent=2))
    print(f"  {seen['n']} seen articles")

    print("selecting held-out articles ...", flush=True)
    held = build_heldout(seen, tok)
    (out / "heldout_manifest.json").write_text(json.dumps(held, indent=2))
    print(f"  {held['frozen_n']} qualify (target {TARGET}, floor {FLOOR})")
    if held["frozen_n"] < FLOOR:
        raise SystemExit(f"only {held['frozen_n']} qualify, below the floor of {FLOOR}")

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    protocol = {
        "frozen_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "code_sha": git.stdout.strip(),
        "working_tree_clean": dirty.stdout.strip() == "",
        "uncommitted": dirty.stdout.strip().splitlines(),
        "model": {"id": MODEL, "revision": REV, "dtype": "bfloat16", "mode": "eval"},
        "codec": {"name": "cuSZp", "mode": "fixed",
                  "commit": "RECORDED AT RUN TIME BY environment.json"},
        "eps": "eps_i = float32(c * std(x_i.float(), correction=1)), per tensor",
        "seeds": {"run": SEED_RUN, "bootstrap": SEED_BOOTSTRAP,
                  "bootstrap_resamples": 5000},
        "tolerances": {
            "tau_absolute": TAU,
            "fp32_acceptance": "E32 <= eps + tau",
            "bf16_posterior": "Ebf <= eps + tau + R, with R the measured rounding",
            "bf16_guarantee": "none claimed",
            "scoring_replacement_max_nll_diff": 1e-6,
            "full_vs_cached_alignment_investigate_above": 1e-3,
            "delta_nll_tolerance_if_used": DELTA_NLL_TOLERANCE,
        },
        "B0": {"caches": "2 dev docs x {1024, 2048} tokens", "tensors_per_cache": 56,
               "c_grid": [0.01, 0.10], "fresh_combinations": 4 * 56 * 2,
               "reuse_passes_per_c": 20,
               "acceptance": ["raw byte-identical", "legal reuse == fresh bitwise",
                              "0 non-finite", "0 fp32 bound failures",
                              "all injected faults detected",
                              "56 GPU1 outputs simultaneously live"]},
        "B1": {"arms": ["raw", "K-only c=0.10", "V-only c=0.10", "K+V c=0.10"],
               "prefill_tokens": 1024, "scored_tokens": 256,
               "continuation_tokens_taken": 257,
               "documents": held["frozen_n"],
               "results_expected": held["frozen_n"] * 4,
               "statistic": "paired per-document delta NLL, article-level bootstrap CI",
               "note": "improvement is not a pass condition; a CI crossing zero means "
                       "the direction is undetermined, not that compression is safe"},
        "B2": {"paths": ["host-staged serial", "host-staged pipeline depth=8",
                         "single-file write + one fsync"],
               "inputs": 4, "paired_configurations": 12,
               "pairs_per_configuration": 30, "segments": 3,
               "timed_runs": 12 * 30 * 2,
               "move_endpoint": "GPU0 raw cache ready -> 56 independent bf16 "
                                "outputs live on GPU1",
               "write_endpoint": "GPU0 raw cache ready -> payload written, fsync "
                                 "returned, file closed",
               "statistic": "paired R = exp(mean(log(T_comp/T_raw))), bootstrap CI",
               "note": "no required speedup; slow samples are not removed"},
        "budget_usd": {"cap": 100, "experiments": 70, "debug_reserve": 20,
                       "storage_transfer": 10},
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(f"\nfrozen at {protocol['code_sha'][:12]}"
          f"{'' if protocol['working_tree_clean'] else ' (DIRTY TREE)'}")
    print(f"wrote {out}/protocol.json, seen_documents.json, heldout_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
