# D14 — Table B, the scientific-compressor coordinate system

24 captured Qwen3-1.7B KV tensors, matched absolute error bounds, all ratios
against **bf16 bytes** (2 B/element). SZ3 and zfp read a lossless bf16→fp32
upcast; reporting their ratio against those fp32 bytes would hand each a free 2×.
Run entirely on the laptop, **$0**. Raw: `results/public/d14_table_b/table_b_kv.json`.

## Compression ratio — SZ3 wins, decisively

| ε | SZ3 (INTERP_LORENZO) | zfp (fixed-accuracy) | ours, blockwise | ours, per-channel |
|---|---|---|---|---|
| 0.05 | **2.59×** | 1.48× | 1.44× | 1.82× |
| 0.15 | **4.07×** | 1.84× | 1.85× | 2.31× |
| 0.5 | **12.63×** | 2.45× | 2.42× | 3.12× |

SZ3 also spends its budget exactly — max error lands on ε to four decimals —
while this codec is conservative (0.373 at ε = 0.5), because widths round up to
the next supported 4/6/8 and the bf16 half-ulp is subtracted from the budget up
front. Both are real inefficiencies, not measurement artefacts.

## A claim from D8 that this retracts

D8 concluded that a general error-bounded compressor underperforms on these
tensors because its inductive bias assumes one exploitable structure. **That is
wrong on ratio, and this measurement is what shows it.** SZ3's prediction is
perfectly effective on KV cache — 5× better than this codec at ε = 0.5. The
assumption that LLM intermediate state is "not smooth enough to predict" did not
survive contact with a real predictor. `d8_kv_findings.md` is corrected.

What survives from D8 is narrower and still holds: *within this codec*, which
axis the error budget follows changes the ratio a lot, and the best axis differs
between keys and values and with depth.

## Transport decision — the ranking inverts

For a 21 MB payload at the 8.45 GB/s link measured in D6, encode+decode scaled
from the per-tensor measurements:

| method | ratio | codec cost | transfer saved | **net** | break-even |
|---|---|---|---|---|---|
| **ours, Triton on A40** | 2.55× | 0.65 ms | 1.51 ms | **+0.86 ms** | **19.6 GB/s** |
| zfp, CPU | 2.45× | 115.9 ms | 1.47 ms | −114.4 ms | 0.11 GB/s |
| SZ3, CPU | 12.63× | 196.7 ms | 2.29 ms | **−194.4 ms** | 0.10 GB/s |

**SZ3 compresses five times better and loses the decision by 195 ms.** Its
break-even sits at 0.10 GB/s: it would only pay on a link slower than 100 MB/s.

This is the project's thesis stated against a mature, well-regarded compressor
rather than against a strawman. Ratio is not the objective; end-to-end time is.
The right codec for a transport path is not the one with the best rate–distortion
curve — it is the one whose cost fits inside the transfer it saves.

## The advantage here is implementation, not algorithm

Worth stating plainly, because the table invites the opposite reading. On CPU,
this codec's reference implementation manages 0.027 GB/s against SZ3's 0.213 and
zfp's 0.362 — it is the **slowest** of the three. The entire transport advantage
comes from the Triton kernel at 39 GB/s, roughly 180× the CPU reference.

So the finding is not "this compression algorithm is better". It is that a simple
algorithm that maps cleanly onto a GPU beats a much better algorithm that does
not, *for this particular decision*.

## The limitation that matters most

**SZ3 is measured here CPU-only**, through `pysz`. GPU implementations of the SZ
family exist — cuSZ and cuSZp — and are not benchmarked in this pass. cuSZp's own
documentation targets non-smooth data and ML weights/tokens, which is exactly
this tensor family.

The comparison above is therefore between *this SZ3 build* and a GPU codec, not
between prediction-based compression and quantisation. A GPU SZ implementation
could plausibly keep most of the 5× ratio advantage at a fraction of the cost,
and if it does, the transport decision flips back. **That makes cuSZp the highest-value
remaining experiment in the project**, and until it is run, no claim should be
made about the SZ family as a whole.

## Other limits

* One tensor family (KV cache), one model, 24 tensors.
* CPU timings are single-threaded on Apple silicon; SZ3 and zfp both support
  OpenMP and would improve with it. The 180× gap to the GPU kernel is wide enough
  that threading does not change the conclusion, but the CPU numbers are a floor.
* zfp is applied to the raw `[tokens, head_dim]` 2D layout and labelled a
  structure-sensitive baseline; its documentation does not recommend it for data
  without spatial correlation.
