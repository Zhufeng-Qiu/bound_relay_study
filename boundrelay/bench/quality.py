"""Quality lane — Figure 3. Deliberately independent of the H100 session.

Round-trips a small model's KV cache through the codec and measures perplexity
delta against eps. Implementation route: a ``transformers`` ``Cache`` subclass
that compresses on write and decompresses on read -- roughly a hundred lines.
Do not modify attention kernels; that is a different project.

This exists because the paired-WER result depends on the runtime integration,
which is a 15% stretch goal. Making the entire error-to-quality story hostage to
the riskiest step would be a planning error: quality propagation is the part of
the work closest to Sian Jin's own line, so it needs a path that cannot fail.

Two legs, then:
    perplexity delta   always obtained (D20)
    paired WER/CER     only if the H100 session lands (D25)

STATUS: D20.
"""

from __future__ import annotations

from pathlib import Path


def run(model_id: str, eps_values: list[float], out: Path) -> None:
    raise NotImplementedError("D20")
