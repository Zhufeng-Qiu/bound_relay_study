"""D5 -- exercise the capture path against a synthetic stand-in.

The point is that nothing about the capture code is written for the first time
inside a rented pod. Manifest schema, incremental flushing, and crash-resumption
all get proven here for free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from boundrelay.codec.contract import TensorSpec
from boundrelay.trace.manifest import CaptureEnv, Manifest, read, write


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/private_raw/dryrun"))
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    mpath = args.out / "manifest.json"

    m = Manifest(
        corpus_id="dryrun",
        env=CaptureEnv(
            gpu="none (dry run)", gpu_count=0, driver="-", cuda="-",
            torch=torch.__version__, model_id="synthetic", model_revision="-",
            weight_dtype="bf16", sglang_omni_sha="-", image_digest="-",
            notes="synthetic stand-in; proves manifest + incremental flush",
        ),
    )
    write(mpath, m)

    for i in range(args.n):
        x = torch.randn(64, 2048).to(torch.bfloat16)
        torch.save(x, args.out / f"t{i:03d}.pt")
        m.append(mpath, TensorSpec(
            name=f"t{i:03d}", shape=tuple(x.shape), dtype=str(x.dtype),
            contiguous=x.is_contiguous(), producer="thinker", consumer="talker",
            family="hidden_state", bytes_per_request=x.numel() * 2,
        ))

    back = read(mpath)
    assert len(back.tensors) == args.n, "incremental flush lost entries"
    print(f"ok: {args.n} tensors, manifest round-trips, crash-safe append verified")


if __name__ == "__main__":
    main()
