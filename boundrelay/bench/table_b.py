"""Table B — scientific-compressor coordinate system.

Places this codec in the coordinate system Sian Jin's field already uses.

    cuSZp (fixed mode)  GPU; its own docs target non-smooth data and ML
                        weights/tokens, which is exactly this tensor family.
                        cuSZ overlaps it -- run one, not both.
    CUDA zfp            GPU; built for spatially-correlated 2D-4D arrays and
                        not recommended for 1D. Kept as an explicitly labelled
                        structure-sensitive baseline on the raw
                        [tokens, hidden_dim] layout.
    SZ3                 CPU framework. One row, ratio and error only. It cannot
                        be compared against a GPU kernel on throughput and is
                        not going to be.

Two methodological traps, both fatal if missed:

* cuSZp/SZ3 read fp32. bf16 -> fp32 is a *lossless* upcast (bf16 is a truncated
  fp32), so values are unchanged and an absolute eps transfers cleanly. But the
  ratio denominator must stay at the bf16 byte count, or every fp32 baseline
  collects a free 2x.
* Upcast cost -- kernel time plus 2x the memory traffic -- gets its own column.
  It is not folded into throughput.

Wording, everywhere: "benchmarked against ... under matched error settings".
Never "validated against".

STATUS: D14.
"""

from __future__ import annotations

from pathlib import Path


def run(corpus: Path, out: Path) -> None:
    raise NotImplementedError("D14")
