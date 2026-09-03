# Phase E — a narrow microbenchmark, not a verdict on pipelining

Stretch phase, stopped at its feasibility gate. Raw:
`results/public/e_pipeline/pipeline.json`.

## Why it was worth trying

Phase C put cuSZp at **76% of the compressed path** — 19.89 ms encode plus
17.07 ms decode against 8.99 ms of transfer — and the whole path 2.1× slower than
moving the cache raw. If those stages could run concurrently the steady state
would be `max(encode, transfer, decode)` ≈ 19.9 ms rather than their 45.9 ms sum,
which lands under the 23.47 ms raw baseline. **Overlap was the one mechanism that
could have flipped the decision.**

The natural streaming unit was already there: the cache's own 56 tensors, each
compressed whole, so nothing would be given up in ratio.

## What was actually measured, and what it does not cover

The pipeline this phase set out to test was

    GPU0 encode(i)  ||  copy(i-1)  ||  GPU1 decode(i-2)

which never requires two encodes to run at once. **What the gate measured was two
concurrent encodes on GPU0** — a cheaper proxy, chosen because it fails fast. It
is not the same thing, and the distinction matters for what may be concluded.

## The gate

Two encodes, issued on two CUDA streams on the same device, against the same two
encodes issued serially:

| | median |
|---|---|
| serial, two encodes | **0.81 ms** |
| separate streams, two encodes | **4.14 ms** |
| overlap efficiency | **−415%** |

They do not merely fail to overlap — **they are 5× slower when issued
concurrently.** The gate threshold was +10%, so the pipeline was not implemented,
per the phase's own stop rule.

**What this establishes:** cuSZp `fixed` has no usable same-GPU encode/encode
concurrency on this A40.

**What it does not establish:** that a cross-GPU transport pipeline is infeasible.
Encode-on-0 overlapping decode-on-1 and a host copy is a different question and
was not tested. The claim in an earlier version of this document — that the one
mechanism able to flip the transport decision is unavailable — went beyond the
measurement and is withdrawn.

## Why, and why it matters beyond this project

cuSZp's kernel signatures carry `volatile unsigned int* cmpOffset`,
`volatile unsigned int* locOffset` and `volatile int* flag` — the shape of a
decoupled-lookback prefix scan, which needs all its thread blocks resident and
communicating. A kernel like that effectively owns the device; a second instance
on another stream contends for the same SMs and each spins waiting for blocks that
cannot be scheduled.

That is a **mechanism hypothesis inferred from the kernel signatures**, not
something this experiment demonstrated. It is consistent with the timing, and it
would explain it, but nothing here rules out a simpler cause such as launch
serialisation or resource contention that a larger device would absorb.

The useful, defensible version: a compressor that gives up some ratio for stream
concurrency could hide behind the transfer, and whether cuSZp can be made to do
that is a question for the compressor rather than the transport layer. Answering
it needs the cross-GPU double-buffered pipeline this phase did not build.

## What this does not establish

* Whether cuSZp *decode* has the same property, or whether encode-on-0 could
  overlap decode-on-1 across two devices. Only same-device encode/encode was
  tested, because it is the cheapest gate and it failed.
* Whether a different mode (`plain`, `outlier`) behaves differently. `fixed` was
  used throughout.
* Anything about pipelining in general. This is one compressor's kernel structure.
