# A3b — SZ3's predictor on KV cache: a real effect, a third the size, and only on V

Gate B0 d00 full cache: 28 layers × K/V at 2048 tokens, Qwen3-1.7B pinned at
`b9352fbb`, single prefill. `eps_i = c · std_i`, ratios against bf16 bytes.
`pysz` on CPU, **$0**. Raw: `results/public/a3_sz3_full/sz3_full_scan.json`.
Script: `scripts/sz3_full_scan.py`. Supersedes `superseded/a3_sz3_config_findings.md`.

The pilot reported that turning SZ3's prediction off beats every predictor by
**14.4%**, and left two liabilities against that claim. It scanned three `cmprAlgo`
values out of a configuration space that lives behind `Config::loadcfg` and was
never opened — "you did not tune SZ3" was a fair objection. And it ran on the
superseded corpus: an 81-token greedy-decode cache from an unpinned revision, while
every other claim in this project had moved to the Gate B0 prefill capture.

Both are closed here, and the claim comes back smaller and much more specific.

## The scan gives prediction an oracle and keeps NOPRED frozen

For every tensor, prediction is credited with the best result over the whole
reachable grid — algorithm, interpolation kernel, interpolation direction, Lorenzo
and regression flags, 28 configurations in the 3-D layouts. `NOPRED` gets its one
default configuration everywhere. Picking per tensor is not a deployable policy and
is not offered as one; the point is that an oracle cannot overfit *toward* the
conclusion being tested, so if prediction still loses under that handicap, no
train/test split is needed to say so.

Which INI keys actually reach the compressor is measured, not assumed. Each key is
swept over its range on one fixed smooth field, where predictor settings are what
the data rewards:

| key | reaches the compressor | byte spread over its range |
|---|---|---|
| `InterpolationAlgo` | yes | 131.7% |
| `InterpolationDirection` | yes | 36.7% |
| `Lorenzo` | yes | 28.2% |
| `Regression` | yes | 0.2% |
| `QuantizationBinTotal` | nominally | 0.03% — three bytes, a header field |
| **`BlockSize`** | **no** | 0.00%, byte-identical from 2 to 32 |
| **`Lorenzo2ndOrder`** | **no** | 0.00% |
| **`Regression2ndOrder`** | **no** | 0.00% |

Three keys parse without error and change nothing in the output. They are reported
as unreachable rather than swept, so that "the whole configuration space" means
what it says: through this binding, on this build, block size is not tunable.

## The effect is real and about a third of the reported size

Native `[H, S, D]` layout — the order the tensor is actually stored and moved in —
over all 56 tensors:

| | c = 0.01 | c = 0.03 | c = 0.10 |
|---|---|---|---|
| NOPRED beats the per-tensor prediction oracle on | 34 / 56 | 33 / 56 | 34 / 56 |
| by, on average | **+2.0%** | **+1.8%** | **+1.9%** |
| range across tensors | −2.8% … +49.3% | −3.9% … +50.1% | −5.7% … +49.3% |
| one frozen configuration: NOPRED | 2.263× | 2.924× | 4.327× |
| one frozen configuration: best predictor (`INTERP`) | 2.218× | 2.867× | 4.206× |

The pilot's three-algorithm numbers reproduce here exactly — NOPRED 5.046×,
`LORENZO_REG` 4.411×, `INTERP` and `INTERP_LORENZO` 4.139× — so the difference
below is the scan, not a different measurement of the same thing. Where the 14.4%
went, in two steps:

| | c = 0.10, native, frozen |
|---|---|
| pilot corpus, pilot's three algorithms | **+14.4%** (5.046× / 4.411×) |
| pilot corpus, whole reachable grid | **+9.0%** (5.046× / 4.630×) |
| Gate B0 prefill corpus, whole reachable grid | **+2.9%** (4.327× / 4.206×) |

Roughly a third of the inflation was configuration space and two thirds was the
corpus. Both halves were the pilot's own stated limits; neither was free.

## It is not a property of KV cache. It is a property of V

The mean hides the result. Split by kind, at `c = 0.10`, native layout:

| | NOPRED beats the prediction oracle on | mean |
|---|---|---|
| **V** | **28 / 28** | **+5.0%** |
| **K** | 6 / 28 | **−1.1%** |

On values, no configuration SZ3 offers beats no-prediction on any tensor at any `c`.
On keys, prediction wins on 22 of 28 — and it wins with depth: NOPRED is +2.6% ahead
in layers 0–8 and 3.4% behind in layers 9–18.

"SZ3's prediction stage costs ratio on KV cache" is therefore false as stated. It
costs ratio on **V**, it pays on **K** below the first few layers, and the pilot's
corpus — three layers, 81 tokens — could not see the difference.

## Why: the cache is white on every axis but one, and only for keys

Std of adjacent differences divided by the tensor's own std, per axis. A white
sequence gives √2 = 1.414; below it means there is something for a predictor to
work with.

Medians over 28 tensors of each kind. Medians rather than means throughout, because
the scale statistic below is strongly skewed — K's channel spread has mean 25.4×
against median 15.4× — and one document should not mix estimators.

| | along `head_dim` | along tokens | across heads |
|---|---|---|---|
| K (28 tensors) | 1.417 | **0.50** | 1.401 |
| V (28 tensors) | 1.413 | **1.10** | 1.419 |

The KV cache is indistinguishable from white noise along the contiguous axis and
across heads, for both kinds. Correlation exists on exactly one axis — tokens — and
it is strong for keys and weak for values. Across tensors, NOPRED's advantage
correlates with the token-axis statistic at **+0.43** (c = 0.01), **+0.46** (c = 0.03)
and **+0.42** (c = 0.10): the more token-correlated a tensor, the less no-prediction wins.

That is the mechanism, and it makes the K/V split a prediction rather than an
observation: a key at token *t* is close to the key at *t−1*; a value is much less
so; and nothing in either is predictable along the head or channel axes.

### Smoothness is not the only structure, and not the one KIVI and PackKV use

The table above measures **smoothness** — whether stepping one position along an
axis lands near where you were. That is what a predictor needs, and along channels
there is none, for either kind.

It is not what per-channel quantisation needs. That wins when some channels are
consistently *larger* than others, however uncorrelated neighbouring channels are —
a scale claim, not a smoothness one. KIVI quantises the K cache channel-wise and
V token-wise, citing channel correlations, and PackKV inherits the split. Read
against the smoothness row alone, this project's result looks like it contradicts
them. Measured side by side it does not:

| | K | V |
|---|---|---|
| adjacent-difference std / σ, **along channels** | 1.417 — white | 1.413 — white |
| adjacent-difference std / σ, **along tokens** | **0.50** | 1.10 |
| **per-channel scale spread** (max / median channel σ) | **15.4×** | 1.54× |
| per-token scale spread | 1.19× | 1.55× |

**The K cache's channel axis is white and 15× heterogeneous in scale at the same
time.** Both designs are right and they exploit different structure on one axis:
per-channel quantisation takes the scale, and a predictor finds nothing to take.
That is why no-prediction wins here without the channel-wise quantisation
literature being wrong.

V is flat in all four cells — 1.413, 1.10, 1.54×, 1.55× — which is why `NOPRED`
beats every predictor on 28 of 28 value tensors, and why PackKV can pick token-wise
for V on purely computational grounds without paying for it statistically.

Layout tests it. `c = 0.10`:

| layout | K: NOPRED wins | K mean | V: NOPRED wins | V mean | NOPRED ratio |
|---|---|---|---|---|---|
| native `[H, S, D]` | 6 / 28 | −1.1% | 28 / 28 | +5.0% | 4.327× |
| token-major `[T, D]` (2-D) | **28 / 28** | +3.2% | 28 / 28 | +2.2% | 4.313× |
| tokens contiguous `[H, D, S]` | 11 / 28 | +2.3% | 28 / 28 | +3.2% | **4.441×** |

Fold the token axis away — token-major puts `(t, h)` next to `(t, h+1)`, which are
not token-adjacent — and prediction loses on keys too, unanimously. Restore it and
prediction recovers. Which axis is *fastest* barely matters, because SZ3's 3-D
stencils reach all three; what matters is whether the token axis survives the
reshape at all. **Table B had been feeding SZ3 the 2-D token-major view**, which is
the layout in which prediction is guaranteed to look useless.

One incidental result worth keeping: tokens-contiguous `[H, D, S]` gives the best
absolute ratio for the no-prediction path, 4.441× against 4.327× native — a free
2.6% for a codec that does no prediction at all, presumably because its entropy
stage sees a more repetitive byte stream.

## What this does and does not establish

Established, on this corpus: with SZ3 given every configuration `pysz` can reach,
no-prediction still compresses V better than any predictor on every one of 28
tensors at every `c`, and the single best deployable configuration is `NOPRED` at
every `c` and in every layout tested. The token axis carries all the exploitable
correlation, and keys carry far more of it than values.

Not established:

* **One document.** The d00 full cache is the only Gate B0 capture whose raw
  tensors survive locally, so the depth axis is complete (28 layers) and the
  document axis is not. The comparison is built to need no train/test split, but
  it cannot speak to across-document variance. The pilot's four-document result
  points the same way and is on a corpus this one supersedes.
* **One model.** Whether `adj/std ≈ 0.50` along tokens for keys is a Qwen3
  property, a GQA property, or general is untested.
* **Ratio only, and CPU only.** `pysz` contributes no throughput number here by
  design.
* **`BlockSize` is unreachable through this binding**, not shown to be irrelevant.
  A build exposing it might move the predictor numbers.
* **The scale statistic is descriptive, not a compression result.** It says the K
  cache's channels differ in magnitude by 15×; it does not measure what a
  per-channel quantiser achieves on this corpus, and no comparison against KIVI or
  PackKV was run. It is reported to place this project's smoothness result beside
  theirs, not to adjudicate between them.
