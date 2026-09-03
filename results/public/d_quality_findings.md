# Phase D — keys and values are not equally safe to compress

16 documents, 1024-token prefill → 128-token continuation, real cuSZp round trip
ending in bf16, Qwen3-1.7B pinned. Four static strategies at `eps_i = c·std_i`.
The uncompressed half of a one-sided strategy still ships, so its raw bytes count
toward the payload. Raw: `results/public/d_quality/quality_d.json`.

Baseline perplexity **15.665**, full-cache payload **117.4 MB**.

| strategy | c | payload | vs raw | Δ perplexity | Δ % |
|---|---|---|---|---|---|
| uniform K+V | 0.01 | 62.7 MB | **0.53×** | +0.014 | **+0.09%** |
| uniform K+V | 0.03 | 51.2 MB | 0.44× | +1.032 | +6.59% |
| uniform K+V | 0.10 | 38.8 MB | 0.33× | +11821 | **+75467%** |
| K-only | 0.01 | 89.9 MB | 0.77× | +0.188 | +1.20% |
| K-only | 0.03 | 84.1 MB | 0.72× | +0.616 | +3.93% |
| K-only | 0.10 | **78.0 MB** | 0.66× | +21.68 | **+138%** |
| V-only | 0.01 | 90.2 MB | 0.77× | +0.019 | +0.12% |
| V-only | 0.03 | 84.5 MB | 0.72× | +0.035 | +0.23% |
| V-only | 0.10 | **78.3 MB** | 0.67× | −0.119 | **−0.76%** |

## At a matched byte budget, only one of them degrades measurably

The two c = 0.10 one-sided rows ship almost identical payloads — 78.0 MB against
78.3 MB, 0.4% apart — which makes them a matched-budget comparison without any
bisection needed.

| | ΔNLL | 95% CI (document bootstrap) | |
|---|---|---|---|
| K-only | +0.869 | [+0.692, +1.041] | large, significant |
| V-only | −0.008 | [−0.023, +0.008] | **crosses zero** |

**K-only compression caused a large and statistically significant degradation;
V-only showed no detectable degradation.** That is the claim the data supports.

An earlier version of this document reported a "180× difference" from the ratio of
the point estimates (+138% against −0.76%). That ratio is not defensible: its
denominator is statistically indistinguishable from zero, so the quotient is
arbitrary. Nor did V-only *improve* quality — the negative point estimate is
inside its own interval.

The direction holds at the tighter bound too, where V-only's effect is also small
(+0.23% against K-only's +3.93%), but the same caution applies to reading a ratio
from it.

That ordering is mechanically unsurprising once stated. Keys enter attention
through a dot product inside a softmax, so perturbing them moves *where* the model
attends and the exponential amplifies it. Values are combined linearly by weights
the keys already fixed, so an error there is averaged rather than amplified.
Uniform compression at c = 0.10 inherits the key sensitivity and collapses
entirely — perplexity 15.7 → 11838.

## But the best strategy is not the one-sided one

At any acceptable quality, **uniform at a tight bound dominates**:

| | payload | Δ % |
|---|---|---|
| **uniform, c = 0.01** | **0.53×** | **+0.09%** |
| V-only, c = 0.10 | 0.67× | −0.76% |
| V-only, c = 0.01 | 0.77× | +0.12% |

V-only can never ship less than the keys' share of the cache, so its payload floor
is around 0.66×. Compressing both halves gently beats compressing one half hard.
The useful conclusion is not "compress V only" — it is that **the error budget
should be spent where the model is insensitive, and the way to exploit that is an
asymmetric bound, not a one-sided strategy.** Testing per-kind bounds
(`c_K ≪ c_V`) is the obvious next experiment and was not run.

## Limits

* One model, 16 documents, one prefill/continuation split. Whether the K/V
  asymmetry holds across scales is untested.
* Three bounds per strategy. The frontier is sparse and the c = 0.10 uniform point
  is far outside any usable region.
* Perplexity on a teacher-forced continuation, not free generation.
* Asymmetric per-kind bounds — the strategy this result actually points at — were
  not measured.
