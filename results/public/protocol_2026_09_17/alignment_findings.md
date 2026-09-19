# Scoring alignment — the full/cached gap is bf16 kernel selection, not misalignment

Qwen3-1.7B pinned at `b9352fbb`, 1×A40, the **two development articles only**.
1024-token prefix, 256 scored positions. Raw: `alignment/alignment_per_token.jsonl`
(per-token differences for every document and dtype), `alignment/alignment_summary.json`.
Script: `scripts/scoring_alignment_diag.py`. **$0** — ran alongside the verification
session.

B1's preflight found that scoring a continuation from a prefix cache and scoring it
inside one long forward disagree by 5.29e-03 in mean NLL. The protocol calls
anything above 1e-3 an investigation trigger rather than a failure. This is the
investigation.

It is also a correction of process: the first time this was looked at, the numbers
went to a terminal and the write-up asserted a conclusion with no record behind it.
The per-token differences are now saved.

## Two causes, two fingerprints

A **position or offset error** is systematic — the cached scores would line up with
the full scores one position over, so shifting the comparison would *reduce* the
disagreement. **bf16 kernel selection** varying with sequence length is diffuse —
it touches every token a little and vanishes in fp32.

Mean absolute per-token difference, by how far the comparison is shifted:

| | shift −2 | shift −1 | **aligned** | shift +1 | shift +2 |
|---|---|---|---|---|---|
| bf16, doc 0 | 3.641 | 3.425 | **0.0374** | 3.429 | 3.639 |
| bf16, doc 1 | 3.744 | 3.636 | **0.0395** | 3.638 | 3.747 |
| fp32, doc 0 | 3.636 | 3.423 | **7.51e-06** | 3.423 | 3.636 |
| fp32, doc 1 | 3.740 | 3.633 | **7.46e-06** | 3.633 | 3.740 |

**Aligned is best in all four cases, by roughly 90×.** There is no offset.

## It is precision, and it is not small in bf16

| | mean \|diff\| | median | max | mean-NLL gap | tokens > 0.1 |
|---|---|---|---|---|---|
| bf16, doc 0 | 3.74e-02 | 1.39e-02 | 0.248 | 5.29e-03 | 33 / 256 |
| bf16, doc 1 | 3.95e-02 | 1.43e-02 | 0.262 | 3.33e-03 | 38 / 256 |
| **fp32, doc 0** | **7.51e-06** | 4.53e-06 | 4.96e-05 | **4.77e-07** | **0 / 256** |
| **fp32, doc 1** | **7.46e-06** | 4.29e-06 | 4.39e-05 | **0.00e+00** | **0 / 256** |

In fp32 the two paths compute the same function — on doc 1 the mean NLL agrees to
the last bit. Moving to bf16 multiplies the mean-NLL gap by **18,079×**, and it is
not a uniform bias: individual tokens move by up to 0.26 NLL and about one in seven
moves by more than 0.1.

So a 1280-token forward and a 256-token forward over a 1024-token cache select
different kernels, or reduce in a different order, and in bf16 that is visible.

## What this licenses, and what it does not

**Licensed.** B1's arms are all scored through the cached path, so no ΔNLL it
reports is a cached-against-full comparison, and this gap is not among the
quantities being differenced. B1's conclusions do not rest on the two paths
agreeing.

**Not licensed.** That the gap *cancels* between arms. It is the same kind of effect
in each arm and need not be the same size: the arms push different values through
the same kernels — one cache is raw, another is a reconstruction — and nothing here
bounds the per-arm residual. The available comparison is crude: the effect is
~4e-03 on a mean NLL while B1's smallest reported effect is 1.34e-02.

**Not licensed either.** Any claim about fp32 inference quality. This ran fp32 only
to identify the cause. Every quality number in B1 is bf16, which is what the model
is served in.

## Limits

* Two articles, one model, one prefix length, one GPU. The mechanism is identified;
  its magnitude on other shapes is not characterised.
* Kernel selection is inferred from the precision dependence and the absence of an
  offset. No profiler was run and no specific kernel is named.
* This measures the harness, not the corpus. No held-out article was touched.
