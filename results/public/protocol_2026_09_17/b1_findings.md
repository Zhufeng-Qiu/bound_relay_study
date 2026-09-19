# B1 — the K/V asymmetry replicates on 32 articles this project had never read

> ### Status: complete, with a recorded protocol deviation
>
> This run **did not pass the error criterion the protocol declared in advance.**
> Two of 3,584 compressed tensors exceeded `E32 ≤ ε + 1e-6`, both of them the same
> tensor appearing in two different arms. The protocol's stop rule said to halt the
> affected path on a failure; the rule was changed mid-round instead — the abort
> threshold was moved to damage (`>1.001 × ε`) and the margin was recorded per
> tensor rather than enforced.
>
> The reasoning is in `b0_findings.md`: on the tensors measured here the excess
> scales with ε rather than sitting under a fixed absolute bound, so an absolute
> tolerance gets harder to satisfy as ε grows. That is an observation about **this
> codec on this data**, not a proof about conforming codecs in general — nothing
> here establishes what any other implementation would do, and two failures out of
> 3,584 is a thin basis for a claim about a class.
>
> Either way the reasoning does not change what happened: **a pre-declared
> acceptance criterion was relaxed after it failed, during the run it governed.**
> That is the thing the protocol was written to prevent.
>
> So this document reports the quality results as measured and does **not** claim
> the original numerical contract was met. Re-running would not fix it: the codec
> would fail the same criterion again, because the criterion is the wrong shape.
> What would settle it is a round conducted under a relative criterion fixed before
> any tensor is compressed.

Qwen3-1.7B pinned at `b9352fbb`, 1×A40, cuSZp `f581dcf3` `fixed`, `c = 0.10`.
**32 held-out WikiText-2 validation articles**, frozen in `heldout_manifest.json`
before any of these numbers existed, with zero title or body overlap with the 24
articles used to develop this project. Four arms × 32 articles = **128 results**,
**256 scored tokens each**. Raw: `b1/quality_documents.jsonl`,
`b1/quality_summary.json`.

The K/V asymmetry was this project's most-quoted result and its least defensible:
it came from sixteen articles that were also used to develop it. This is the
re-test.

## It replicates, in all three directions

| arm | ΔNLL | 95% CI (paired, by article) | PPL | payload / raw | articles worse |
|---|---|---|---|---|---|
| **K-only** `c=0.10` | **+0.0155** | [+0.0068, +0.0235] | **+1.56%** | 0.664 | **26 / 32** |
| **V-only** `c=0.10` | **−0.0278** | [−0.0320, −0.0234] | **−2.74%** | 0.667 | **1 / 32** |
| **K+V** `c=0.10` | **−0.0134** | [−0.0235, −0.0042] | −1.33% | **0.330** | 11 / 32 |

All three intervals exclude zero. Against the development-set numbers:

| | development (16 articles, 128 tokens) | held-out (32 articles, 256 tokens) |
|---|---|---|
| K-only | +0.0228 | **+0.0155** |
| V-only | −0.0323, 16/16 improved | **−0.0278, 31/32 improved** |
| K+V | −0.0122, CI crossed zero | **−0.0134, CI excludes zero** |

Same direction, similar magnitude, on articles the project had never read. The K+V
arm is now significant where it previously was not — a larger sample, not a larger
effect.

**Keys and values are not equally safe to compress, and the ordering is not a
quirk of the sixteen articles it was found on.** At payloads within 0.5% of each
other — 0.664 against 0.667 — compressing keys costs 1.56% of perplexity and
compressing values does not cost anything.

## The V improvement replicates too, and is still unexplained

V-only at `c = 0.10` lowers the mean teacher-forced NLL by 0.0278 on these 32
articles — a 2.74% change when converted through `exp(ΔNLL) − 1` — with 31 of the 32
improving. The development set showed the same direction on its own 16.

Two disjoint article sets pointing the same way makes it unlikely that *one
particular sample* produced it. It does not rule out sample effects in general: both
sets are WikiText articles scored the same way under one model and one bound, and a
different corpus, model or bound could behave differently.

It is still not evidence that lossy compression improves a language model. One
model, one corpus, 256 scored tokens per article, one bound, teacher-forced
scoring of a single cache reconstruction. Noise regularisation, value-outlier
suppression and an interaction with bf16 rounding all remain consistent with it and
none has been tested. The claim the byte argument needs is the weaker one: **on these 32 held-out
articles, compressing V at `c = 0.10` did not raise the mean teacher-forced NLL.**
That is an average over articles and over 256 positions each — it is not a statement
that any individual article, or any individual token, was unharmed.

## What the preflight established, and what it flagged

**Cache replacement is exact.** Replacing the prefix cache with a bit-identical
copy moved the per-token NLL by `0.00e+00` on both development articles. The
scoring harness does what it claims.

**The full-forward / cached-forward gap is bf16 numerics, not misalignment.** The
preflight measured a 5.29e-03 mean-NLL gap, above the 1e-3 the protocol set as an
investigation trigger. Investigated:

| | mean \|per-token diff\| | cached − full gap | tokens differing > 0.1 |
|---|---|---|---|
| bf16 | 3.74e-02 | 5.29e-03 | 33 / 256 |
| **fp32** | **7.51e-06** | **4.77e-07** | **0 / 256** |

*(document 0; on document 1 the fp32 mean-NLL gap is exactly 0. Full per-token
record in `alignment_findings.md`.)*

In fp32 the two paths agree to seven decimal places, so they are computing the same
function; the gap is bf16 kernels selecting differently for a 1280-token forward
than for a 256-token forward over a 1024-token cache. Shifting the comparison by
±1 position makes the disagreement **90× worse** in both precisions, which rules out
a position offset.

**What follows from that, and what does not.** All four arms are scored through
the cached path, so no ΔNLL here is a cached-against-full comparison and this gap
is not one of the quantities being differenced. That is the claim the result needs
and it holds.

An earlier draft went further and said the gap "sits identically in every arm and
cancels in every ΔNLL". That was not measured and is not safe to assert: the arms
feed *different* values through the same kernels — one arm's cache is raw, another's
is a reconstruction — so the bf16 kernel noise is of the same kind in each arm but
its magnitude need not be identical, and nothing here bounds the residual. What can
be said is that the effect is ~5e-03 on a mean NLL while the smallest effect being
reported is 1.34e-02, and that it applies to every arm rather than to one; it is not
established that it cancels exactly.

## The fp32 margin, measured rather than certified

Across **3,584 compressed tensors**, the largest `E32/ε` was **1.0000014646**, and
**2 tensors exceeded `ε + 1e-6`** — 0.056%. The protocol's absolute τ is not met by
every tensor, τ was not widened, and this run therefore does not certify
`E32 ≤ ε + 1e-6`. See `b0_findings.md`: the excess is relative, a few float32 ulps
of ε, so it crosses an absolute tolerance only once ε exceeds roughly 0.68.

## What this does not establish

* One model, one bound (`c = 0.10`), one codec. Nothing here says where the
  quality cost becomes unacceptable, only that at this bound K costs and V does not.
* Teacher-forced scoring of **one** cache reconstruction. The continuation's own
  K/V stay raw, so this is not the cost of compressing every state a long
  generation produces.
* 32 articles is 32. The interval is over articles and is honest about that; it is
  not a claim about every corpus.
* "Not used in this project's development" is what the held-out set establishes.
  WikiText-2 is public and Qwen3 was trained before any of this ran.
