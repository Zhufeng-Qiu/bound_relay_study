# D20 — quality–compression frontier, measured on a laptop

Qwen3-0.6B, WikiText-2 test split, 16 sequences of 256 tokens. Prefill 128 →
decode 128. Per-channel allocation. fp32 on Apple MPS, **$0 of GPU rental**.
Raw: `results/public/d20_quality/quality_kv.json`.

> **Superseded in Phase A/D.** The sweep below used a fixed absolute ε applied
> uniformly to the whole cache, and reconstructed through this project's reference
> codec rather than through a real cuSZp round trip ending in bf16. Phase D
> re-measures it with `eps_i = c · std_i`, a real codec round trip, final-bf16
> error checking, and document-level bootstrap CIs.

## Semantics: prefill → transport → decode

A prefix is prefilled to produce a KV cache; that cache is round-tripped through
the codec at ε; the continuation is then scored against the *reconstructed*
cache. This is the disaggregated-serving scenario — KV computed on one worker,
moved, consumed on another — not "compress on every cache update", which no
deployment does. Paired by construction: same prefix, same continuation, same
order, one variable.

## Result

Baseline NLL 3.4706, perplexity **32.155**.

| ε | ratio | Δ perplexity | relative | verdict |
|---|---|---|---|---|
| 0.05 | 1.54× | +0.007 | **+0.02%** | free |
| **0.15** | **2.23×** | **+0.920** | **+2.9%** | controller's operating point |
| 0.5 | 3.24× | +2.042 | +6.4% | |
| 1.0 | 3.62× | +4.190 | +13.0% | |
| 2.0 | 3.81× | +6.342 | +19.7% | past the knee |

Two things the curve says plainly:

**There is a free region.** At ε = 0.05 the codec returns 1.54× for a 0.02%
perplexity cost — indistinguishable from free. A system unwilling to spend any
quality still has compression available to it.

**Returns collapse past ε ≈ 1.** Going from ε = 1.0 to ε = 2.0 buys 5% more
compression (3.62× → 3.81×) for 6.7 points more perplexity. The knee sits between
ε = 0.15 and ε = 0.5, which is where the controller operates.

## Why this ran locally, and why that matters

The paired-WER measurement depends on the SGLang-Omni integration, a 15% stretch
item that has already been aborted once. Leaving the entire error-to-quality story
hostage to the riskiest step would have been a planning error, since propagation
from an element-wise bound to a downstream metric is the part of this work closest
to the compression literature it is positioned against. So this lane was built to
run on a laptop, and it does.

## A methodological note worth keeping

The first attempt used a hand-written paragraph repeated to length as the corpus.
Baseline perplexity came out at **1.03**: the model memorised the text after one
copy, leaving no headroom for a compression effect to register. The numbers looked
plausible and meant nothing. Natural, non-repeating text is a requirement for this
measurement, not a nicety — and a suspiciously perfect baseline is the tell.

## Limits

* Qwen3-0.6B, not the 1.7B whose KV cache the allocation study used, and not
  Qwen3-Omni. Ratio and sensitivity both depend on the model.
* 16 sequences, one prefix/continuation split. No bootstrap CI yet.
* The quality gate is **not yet frozen from measured baseline variance**. This run
  is deterministic — no sampling, fixed inputs, so run-to-run variance is zero and
  the paired delta is the meaningful quantity — but a gate stated as "acceptable
  degradation" still needs a variance estimate behind it before any threshold is
  claimed as passed.
* The whole cache is compressed uniformly. A deployment would likely be selective
  by layer, and D8 showed the structure differs sharply between keys and values
  and with depth.
