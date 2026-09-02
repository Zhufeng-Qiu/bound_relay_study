"""One scripted GPU session: verify, then measure. Never the other way round.

Runbook D9-D11 + D15. The whole point of scripting it is that the pod is billing
while it runs, so nothing here is decided interactively.

Order is deliberate:

1. **Equivalence first.** A fast kernel that disagrees with the reference oracle
   is worth nothing, so the session aborts before spending a second on timing.
2. Throughput by (shape, bit width) -> the cost-model lookup table.
3. Everything lands in JSON under results/, which is what the atlas reads.

Exit code is non-zero if step 1 fails, so a wrapper can tear the pod down.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boundrelay.codec import Allocation, CodecConfig, gpu  # noqa: E402

SHAPES = [
    (64, 2048),      # ~0.25 MB  — a thinker hidden-state chunk
    (512, 2048),     # ~2 MB     — a short audio-encoder embedding
    (2560, 2048),    # ~10 MB    — near the predicted break-even payload
    (5120, 2048),    # ~20 MB    — a video-encoder embedding
]
WIDTHS = [4, 6, 8]
EPS = 0.15


def environment() -> dict:
    return {
        "host": platform.node(),
        "torch": torch.__version__,
        "cuda_built": torch.version.cuda,
        "driver_gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "triton": __import__("triton").__version__,
        "gpu_count": torch.cuda.device_count(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/private_raw/gpu_session"))
    ap.add_argument("--skip-tests", action="store_true",
                    help="only for re-running timings after a verified session")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = environment()
    print(json.dumps(env, indent=2), flush=True)
    (args.out / "environment.json").write_text(json.dumps(env, indent=2))

    if not gpu.available():
        print("FATAL: Triton/CUDA unavailable", file=sys.stderr)
        return 2

    if not args.skip_tests:
        print("\n=== step 1/2: equivalence vs reference ===", flush=True)
        rc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_gpu_equivalence.py"],
            cwd=Path(__file__).resolve().parents[1],
        ).returncode
        if rc != 0:
            print("\nFATAL: GPU path disagrees with the reference. "
                  "No timing is worth collecting until this passes.", file=sys.stderr)
            return 1
        print("equivalence: PASS", flush=True)

    print("\n=== step 2/2: throughput -> cost lookup table ===", flush=True)
    table: dict[str, dict] = {}
    for w in WIDTHS:
        cfg = CodecConfig(eps=EPS, allocation=Allocation.BLOCKWISE, bit_width=w)
        res = gpu.benchmark(SHAPES, cfg)
        for shape, row in res.items():
            table[f"w{w}_{shape}"] = row
            print(f"  w={w} {shape:>16}  {row['ratio']:.2f}x  "
                  f"{row['encode_GBps']:7.2f} GB/s  {row['encode_s']*1e3:7.3f} ms",
                  flush=True)

    (args.out / "cost_table.json").write_text(json.dumps(
        {"environment": env, "eps": EPS, "table": table}, indent=2))
    print(f"\nwrote {args.out/'cost_table.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
