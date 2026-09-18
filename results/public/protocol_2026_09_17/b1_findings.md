# B1 — the K/V asymmetry replicates on 32 articles this project had never read

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

V-only at `c = 0.10` improves next-token perplexity by 2.74%, with 31 of 32
articles improving. The development set said the same thing (16/16) and it was
recorded then as observed and unexplained. It is now observed twice, on disjoint
article sets, which removes *sample* as the explanation and removes nothing else.

It is still not evidence that lossy compression improves a language model. One
model, one corpus, 256 scored tokens per article, one bound, teacher-forced
scoring of a single cache reconstruction. Noise regularisation, value-outlier
suppression and an interaction with bf16 rounding all remain consistent with it and
none has been tested. The claim the byte argument needs is the weaker one that is
firmly established: **on this corpus the error budget on V costs nothing.**

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
| **fp32** | **7.51e-06** | **7.15e-07** | **0 / 256** |

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
