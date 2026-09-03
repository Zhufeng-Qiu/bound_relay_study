# Gate 2, re-measured — overlap is real, larger than reported, and cannot matter

2×RTX A6000 (sm_86), RunPod `EU-SE-1`, torch 2.8.0+cu128. 56-tensor d00 full cache,
`c = 0.10`, cuSZp `fixed`. Raw: `results/public/gate2/gate2_rerun.json`.
Script: `scripts/gate2_rerun.py`. Session cost **$0.65**.

The closing session reported 30% cross-GPU overlap efficiency and that figure was
then flagged unquotable: its decode wrote into an uninitialised destination, and
nothing checked what came out. This re-measures it.

A40 was out of stock at two GPUs everywhere, so a plain re-run would have confounded
the fix with the hardware. Instead both regimes run back to back in one process on
one host — `as_measured` reproduces the original's uninitialised buffers exactly,
`corrected` zeroes the staging buffer, the receive buffer past `cmpSize`, and the
destination before every decode. The column difference is the fix; the host
difference applies to both columns equally.

## The uninitialised regime was decoding wrong data

Six tensors, both regimes, the reconstruction checked against the source on the
host:

| regime | tensors within ε | worst observed |
|---|---|---|
| `as_measured` (uninitialised) | **3 / 6** | 19.18 against ε = 0.2415, **79× the bound** |
| `corrected` (zeroed) | **6 / 6** | 0.2413 against ε = 0.2415 |

Extended to the whole cache in the corrected regime: **0 bound violations in 56**.
The decoder leaves 0–2144 elements per tensor untouched, which is what makes the
destination's prior contents part of the answer.

## Overlap, over the whole cache

56 tensors, five repetitions each, 30 timed iterations after 5 warm-up, medians:

| issuance | median overlap | IQR | range | reaching 90% |
|---|---|---|---|---|
| one host thread | **1.4%** | [−5.1, 13.8] | [−36.5, 60.9] | 0 / 56 |
| two host threads | **53.7%** | [46.6, 62.1] | [29.7, 88.7] | 0 / 56 |

Single-threaded issuance overlaps nothing, for the reason the original gave and
which survives: cuSZp's compress writes the compressed size back to a host
pointer, so it blocks the caller regardless of what the two devices could do in
parallel. Give it a second host thread and half the work overlaps.

**53.7%, not 30%.** The original figure came from a single unrepeated measurement
of a decode that was not producing correct output. The direction of its conclusion
survives; the number does not.

## It does not matter, and the arithmetic says so without the number

The closing session's corrected end-to-end moves this cache raw in **23.64 ms** and
compressed in **50.48 ms**, of which encode is 19.99 ms and decode 15.01 ms. Encode
and decode live on different GPUs, so overlap can remove at most the smaller of
them:

| | |
|---|---|
| saving needed to match the raw path | **26.84 ms** |
| largest saving perfect overlap can give | **15.01 ms** |
| compressed path at 100% overlap | 35.48 ms — **1.50× raw** |
| compressed path at the measured 53.7% | 42.42 ms — **1.79× raw** |

**Perfect overlap of these two stages is not enough.** But that is a narrower
statement than "pipelining cannot help", and an earlier version of this document
made the wider one. It does not follow.

> **Corrected.** This section originally read "pipelining was the one mechanism
> that could have flipped the decision, and it cannot, at any efficiency." That
> generalises a *two-stage* result to a *seven-stage* pipeline, and the arithmetic
> does not survive the generalisation. See below.

## What a real pipeline would do, which is not what was measured

Gate 2 overlaps encode against decode. A pipelined transport overlaps every stage:
while tensor *i* is encoding on GPU0, *i−1* is crossing to the host, *i−2* is
crossing to GPU1, and *i−3* is decoding. Steady-state cost is then the **busiest
resource**, not the sum of stages and not `total − min(two of them)`.

Grouping the measured stages by the resource each one occupies:

| resource | stages | ms per cache |
|---|---|---|
| GPU0 compute | upcast 1.44 + encode 19.99 | **21.43** |
| GPU1 compute | zero 0.93 + decode 15.01 + downcast 1.34 | 17.28 |
| link | D2H 4.32 ∥ H2D 5.01 | 5.01 |

so a perfect pipeline is **GPU0-bound at 21.43 ms** — against a raw path of
23.64 ms. **It wins.** Not by the margin that matters, and not against a raw path
that is also allowed to pipeline, but the claim as originally written was false.

The comparison has to be like for like, because `raw_iter` is serial too — it
copies each tensor down and then up on the default stream. Pipeline both sides:

| | serial | pipelined |
|---|---|---|
| raw | 23.64 ms | **11.82 ms** |
| compressed | 50.48 ms | **21.43 ms** |
| ratio | 2.14× | **1.81×** |

Pipelining helps the compressed path more than it helps raw (2.4× against 2.0×),
and still does not close a 2.1× gap on this link. The surviving conclusion is the
same one, on a correct footing — but the number that matters changes a great deal:

| | break-even |
|---|---|
| serial, as measured | 3.82 GB/s |
| **fully pipelined** | **10.96 GB/s** |
| measured link | 19.87 GB/s per direction |

**Pipelining nearly triples the link speed below which compression pays.** 10.96
GB/s sits above NVMe (2–7 GB/s), above 25/50 GbE, and around PCIe Gen3 x16 — so
the pipelined verdict on a slower path is the opposite of the verdict here, and
this is now the question worth measuring rather than a settled one.

It also relocates the lever. The pipeline is bound by GPU0, and GPU0 is bound by
an encode that reads **fp32** because cuSZp does not take bf16 — 23.5 GB/s of its
own input, but only 11.75 GB/s of actual cache:

| | pipeline bound | break-even |
|---|---|---|
| as measured | 21.43 ms | 10.96 GB/s |
| bf16-native encode, no upcast | 17.28 ms — now **GPU1**-bound | 13.59 GB/s |
| bf16-native at both ends | 9.99 ms | **23.51 GB/s** |

A codec that took bf16 directly at both ends would put the break-even *above* this
host's own link. That is a concrete, quantified target, and it is not reachable by
scheduling — it is a codec change.

**None of the pipelined numbers above are measured.** They are what the measured
per-stage costs imply if a pipeline achieved perfect overlap with no fill, drain,
or contention. That experiment has not been run.

## A labelling error in the superseded numbers

`session_close_findings.md` tabulated "encode on GPU0 alone 7.46 ms" and "decode on
GPU1 alone 6.01 ms". Those are totals of 20 iterations, not per-call times: 0.373
and 0.301 ms respectively. The per-call figure agrees with the same session's own
end-to-end stage timing (19.99 ms of encode over 56 tensors = 0.357 ms each) and
with this host (0.34–0.44 ms). Overlap efficiency is a ratio, so the 30% was not
affected by the scaling error — but the two absolute numbers were wrong by 20×
and are corrected here.

## What is not established

* One host, one GPU pair, one generation. The A6000 and the A40 share the GA102
  die and `sm_86`, and the per-call codec times agree across them, but a Hopper or
  Blackwell host was not tested.
* Two host threads is the only concurrency mechanism tested. CUDA streams were not,
  because the blocking compress makes them moot from a single thread, and a
  multi-process arrangement was out of scope.
* The 53.7% is the median of a distribution that spans 29.7% to 88.7% across
  tensors of identical shape. What varies between them is not explained here.
