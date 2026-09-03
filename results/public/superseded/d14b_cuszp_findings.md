# D14b — cuSZp on the same tensors: this codec is beaten on both axes

cuSZp built from source on 1×A40 (sm_86, CUDA 12.8), benchmarked through a small
C++ driver against the same 24 captured Qwen3-1.7B KV tensors. Session cost ~$0.20.
Raw: `results/public/superseded/d14b_cuszp/`.

> **Corrected in Phase A.** The comparison below used a fixed absolute ε across
> tensors and is withdrawn as an aggregate; the per-tensor conclusion (cuSZp wins
> on both ratio and speed) is unaffected, but the *size* of the gap was measured in
> coordinates that exaggerate it. cuSZp under normalised ε is re-measured on the
> Gate B0 corpus, not re-labelled from the old runs.

## The result

Same 24 tensors, same absolute error bounds, ratios against bf16 bytes:

| ε | cuSZp (fixed) | this codec, Triton (best width) |
|---|---|---|
| 0.05 | **2.30×** @ 0.128 ms | 1.24× @ 1.089 ms |
| 0.15 | **3.09×** @ 0.120 ms | 1.49× @ 1.168 ms |
| 0.5 | **6.20×** @ 0.112 ms | 1.93× @ 0.981 ms |

**cuSZp compresses 2–3× better and runs ~9× faster.** It is strictly better on
this data, on both axes at once. There is no trade being made.

## What this retracts

D14 concluded that "the advantage is implementation, not algorithm" — that a
simple codec mapping cleanly onto a GPU beats a better algorithm that does not.
**That was wrong, and cuSZp is the counterexample.** It is a better algorithm
*and* it maps onto a GPU, so it wins outright. `d14_table_b_findings.md` framed
SZ3's CPU-only cost as the reason prediction-based compression loses the
transport decision; with a GPU implementation in hand, that reason evaporates.

The retraction was foreseeable and was flagged in D14 as the biggest open
limitation. Running it was worth ~$0.20 precisely because it could overturn the
headline — publishing a conclusion that a twenty-minute experiment refutes would
have been the worse outcome by a wide margin.

## What this means for the project

**The codec is not the contribution, and should stop being presented as one.**
A homemade blockwise quantiser is not competitive with a mature GPU error-bounded
compressor on the target data, and saying otherwise would not survive a reviewer
with a GPU and an afternoon.

What survives is codec-agnostic and, if anything, stronger for being instantiated
with the strongest tested baseline rather than with a self-built component:

* **The transport-decision framework.** Measure codec cost on the target hardware,
  measure the link, compute the break-even boundary, and bypass when the payload
  is too small or the link too fast. Nothing in that argument depends on whose
  codec fills the slot — and the boundary can now be drawn using cuSZp's numbers.
* **The measurement discipline that produced the correction.** Topology cannot be
  inferred from an API (D6), ratio cannot be inferred from a synthetic
  distribution (D8), and a codec's standing cannot be inferred from its own
  benchmark (this document).
* **The finding that compression is usually the wrong call** (D21: 69% bypass).
  A better codec moves the boundary; it does not remove it.

## Immediate consequences

1. Figure 1 and Figure 4 should be redrawn with cuSZp as the codec. Its cost at
   ε = 0.5 on real tensors is 0.112 ms encode + 0.080 ms decode at 6.20×, which
   pushes the break-even bandwidth substantially higher than this codec's
   19.6 GB/s — the region where compression pays gets *wider*, not narrower.
2. The CV bullet claiming a GPU codec must be rewritten. What was built and
   measured is a transport-decision framework and a benchmark harness; the codec
   is the reference implementation those were developed against.

## Method note: a flawed large-payload construction

The 20 MB throughput comparison tiled all 24 tensors into one array. That
concatenation creates discontinuities at every tensor boundary, which penalises
cuSZp's 1D predictor — it scored 2.50–2.62× on the tiled array against 3.09× on
the individual tensors. The per-tensor numbers above are the fair comparison; the
tiled figures are reported in the raw JSON but should not be quoted.

`fixed` mode appeared to show 3–6× encode-time variance across processes at tight ε
(**withdrawn — see `b0_corpus_findings.md`**: under a protocol separating cold start
from warm cost, all modes have the same ~0.88 ms median and the spread is the
measurement environment, not the mode)
(3.08 ms vs 0.49 ms at ε = 0.05, median of 7 runs); `plain` and `outlier` were
stable near 40 GB/s. Data-dependent cost is expected for a predictor, but the
spread has not been characterised properly and no throughput claim should rest on
a single `fixed`-mode number.
