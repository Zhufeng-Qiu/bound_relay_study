# Phase E — cuSZp's compression kernel is not stream-concurrent

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

## The gate

Two encodes, issued on two CUDA streams on the same device, against the same two
encodes issued serially:

| | median |
|---|---|
| serial, two encodes | **0.81 ms** |
| separate streams, two encodes | **4.14 ms** |
| overlap efficiency | **−415%** |

They do not merely fail to overlap — **they are 5× slower when issued
concurrently.** The gate threshold was +10%; this is far outside it, so the
pipeline was not implemented, per the phase's own stop rule.

## Why, and why it matters beyond this project

cuSZp's kernel signatures carry `volatile unsigned int* cmpOffset`,
`volatile unsigned int* locOffset` and `volatile int* flag` — the shape of a
decoupled-lookback prefix scan, which needs all its thread blocks resident and
communicating. A kernel like that effectively owns the device; a second instance
on another stream contends for the same SMs and each spins waiting for blocks that
cannot be scheduled.

So the finding is not "pipelining did not help here". It is that **the codec that
wins on ratio and throughput cannot be overlapped with itself**, and that is what
locks the 76% of codec cost Phase C measured. For a transport path, a compressor
that gives up some ratio for stream concurrency could win outright by hiding
behind the transfer — which is a concrete design question for the compressor, not
for the transport layer.

## What this does not establish

* Whether cuSZp *decode* has the same property, or whether encode-on-0 could
  overlap decode-on-1 across two devices. Only same-device encode/encode was
  tested, because it is the cheapest gate and it failed.
* Whether a different mode (`plain`, `outlier`) behaves differently. `fixed` was
  used throughout.
* Anything about pipelining in general. This is one compressor's kernel structure.
