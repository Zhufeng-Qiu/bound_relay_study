# Asymmetric bounds — K and V interact super-additively, and the obvious rule is wrong

16 documents, 1024 → 128 tokens, real cuSZp round trip ending in bf16, document
cluster bootstrap. Baseline perplexity 15.665, full payload 117.4 MB.
Raw: `results/public/session_close/quality_asym.json`, `quality_reconcile.json`.

Phase D found K-only compression significantly damaging and V-only harmless, and
pointed at `c_K ≪ c_V` as the design rule. **That rule is wrong, and the reason is
more interesting than the rule.**

> **Withdrawn in full.** Every catastrophic result below came from a
> `torch.empty` reconstruction buffer. cuSZp does not write elements it expects to
> be zero, and leaves more of them the looser the bound — which is exactly why the
> collapse appeared at large `c_V`. Re-measured against zeroed buffers the 2×2
> interaction contrasts are +0.0012 [−0.0030, +0.0052] and −0.0027 [−0.0081,
> +0.0021]: **there is no interaction**, and nothing in the grid is catastrophic.
> See `remeasure_findings.md`. Kept as the record of a result that did not survive
> its own audit.

## The grid

| c_K | c_V | payload | vs raw | ΔNLL | 95% CI | |
|---|---|---|---|---|---|---|
| 0.01 | 0.01 | 62.7 MB | 0.53× | +0.0009 | [−0.0028, +0.0047] | no damage |
| **0.01** | **0.03** | **56.9 MB** | **0.48×** | **−0.0007** | [−0.0056, +0.0050] | **no damage, best payload** |
| 0.005 | 0.03 | 60.6 MB | 0.52× | −0.0003 | [−0.0040, +0.0033] | no damage |
| 0.03 | 0.03 | 51.2 MB | 0.44× | +0.0638 | [+0.0139, +0.1472] | significant |
| 0.005 | 0.10 | 54.4 MB | 0.46× | **+5.979** | [+5.385, +6.646] | **catastrophic** |
| 0.01 | 0.10 | 50.7 MB | 0.43× | **+6.297** | [+5.677, +6.980] | **catastrophic** |

## The interaction, and why it is not a bug

The catastrophic rows contradicted Phase D, where V at c = 0.10 with K left raw was
harmless. Both were re-run **inside the same code path** to distinguish an
interaction from a defect:

| configuration | ΔNLL | |
|---|---|---|
| K raw, V at 0.10 | −0.0076 | reproduces Phase D's `v_only` exactly |
| K at 0.10, V raw | +0.8688 | reproduces Phase D's `k_only` exactly |
| K at 0.005, V at 0.10 | **+5.979** | |
| K at 0.10, V at 0.10 | **+6.628** | |

Both one-sided results reproduce. **The measurements agree; the effect is real.**

Individually, K at 0.10 costs +0.87 and V at 0.10 costs −0.008. An additive model
predicts **+0.86** for both together. Measured: **+6.63 — 7.7× the additive
prediction.** And compressing K at 0.005, a bound at which K alone is
indistinguishable from lossless, still turns a harmless V compression into a
collapse: perplexity 15.7 → 6191.

Perplexities in the thousands are a breakdown, not a degradation — consistent with
error compounding across 28 layers of a jointly perturbed cache rather than with
graceful loss. Attention selects with K and averages with V; when only one is
perturbed the other still anchors the operation, and when both are, wrong weights
multiply wrong values. That is a hypothesis this experiment does not test.

## What the design rule actually is

Not "spend the budget on V". The usable region is bounded by **c_V ≤ 0.03**
regardless of how tight K is made — 0.005 does not buy permission to loosen V.

There is a real but modest asymmetric gain: **(c_K = 0.01, c_V = 0.03) ships 0.48×
the bytes at no detectable quality cost**, against 0.53× for the symmetric
c = 0.01. About 10% fewer bytes, free.

## The methodological point

**One-at-a-time sensitivity did not predict joint behaviour.** Phase D measured
each kind's marginal effect correctly and the rule inferred from those margins was
wrong by a factor of eight. Any error-budget allocator built from per-tensor
sensitivity — measured the way Phase D measured it — would have chosen
(0.005, 0.10): the second-best payload in this grid, and a configuration that
destroys the model.

## Limits

* One model, 16 documents, one prefill/continuation split.
* Six grid points. The boundary in c_V lies somewhere between 0.03 and 0.10 and was
  not located.
* The mechanism is untested. Whether the collapse is compounding across layers, a
  numerical range failure in bf16, or something else is open.
