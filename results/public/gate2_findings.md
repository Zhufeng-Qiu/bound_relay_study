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

**Perfect overlap is not enough.** The pipeline question is settled not by the
53.7% but by the ceiling above it: even a hypothetical codec whose encode and
decode overlapped completely would leave this path half again slower than moving
the bytes uncompressed. Pipelining was the one mechanism that could have flipped
the decision on this hardware, and it cannot, at any efficiency.

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
