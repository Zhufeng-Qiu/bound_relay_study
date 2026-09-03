# Pre-registration — the multi-stage pipeline experiment

Written before the harness runs, because the interesting outcomes here are the
ones that disagree with the arithmetic, and a prediction recorded afterwards is
worth nothing.

## What is being tested

`gate2_findings.md` overlaps encode on GPU0 against decode on GPU1 — two stages of
seven. A pipelined transport overlaps all of them, and its steady-state cost is the
busiest resource rather than the sum. That number has never been measured; it has
only been inferred from per-stage costs measured in a serial loop.

The experiment runs the real thing: a seven-stage software pipeline over the
56-tensor cache, with the **raw path pipelined through identical machinery** so the
comparison is like for like.

## Hypotheses, and how they are told apart

**H1 — free overlap.** Every stage overlaps as the resource grouping implies:
GPU0 21.43 ms (upcast + encode), GPU1 17.28 ms, link 5.01 ms. Steady state
**21.43 ms**, GPU0-bound, against a pipelined raw path of 11.82 ms.

**H2 — cuSZp serialises its own device.** `cuSZp_compress_1D_fixed_f32` and its
decompress counterpart each call `cudaMalloc` three times and `cudaFree` three
times per invocation, around the kernel. **`cudaFree` synchronises the device.** If
that is what it does here, no D2H can overlap the encode that precedes it on GPU0,
and no H2D can overlap a decode on GPU1. Steady state becomes

* GPU0: upcast 1.44 + encode 19.99 + D2H 4.32 = **25.75 ms**
* GPU1: H2D 5.01 + zero 0.93 + decode 15.01 + downcast 1.34 = 22.29 ms

→ **25.75 ms, and the pipelined compressed path is then *slower than the serial raw
path* (23.64 ms).** Cross-device overlap still works, because `cudaFree` on device 0
synchronises device 0 only — which is exactly why Gate 2 saw 53.7% and not zero.

**H1 predicts the pipeline beats serial raw; H2 predicts it does not.** They differ
by 4.3 ms, well outside the run-to-run spread of these stages.

The discriminator does not depend on the pipeline at all: time N encodes alone,
N D2Hs alone, and N encodes each with a D2H issued asynchronously before it on a
separate non-blocking stream. Under H1 the third is `max` of the first two; under
H2 it is their sum. That probe runs first.

## Registered predictions

| quantity | H1 | H2 | measured |
|---|---|---|---|
| pipelined compressed, depth ≥ 4 | 21.4 ms | 25.8 ms | |
| pipelined raw, depth ≥ 4 | 11.8 ms | 11.8 ms | |
| ratio compressed / raw | 1.81× | 2.18× | |
| bottleneck resource | GPU0 | GPU0 | |
| break-even implied | 10.96 GB/s | 9.12 GB/s | |

Both are optimistic: neither accounts for fill and drain, for the host ring's extra
copy, or for GPU0 memory-bandwidth contention between an upcast and a concurrent
D2H. **The measured pipeline is expected to be worse than whichever hypothesis
holds**, and the registered numbers are ceilings, not point predictions.

## Falsification conditions, fixed in advance

* **Depth 1 must reproduce the serial measurement** — 50.48 ms compressed, 23.64 ms
  raw, within run-to-run spread. A depth-1 pipeline is a serial loop with extra
  bookkeeping; if it does not agree, the harness is measuring something else and
  no other number from it may be quoted.
* Every reconstruction is checked against its source on the host, against
  `ε + bf16 half-ulp`. Any violation voids the run — the same rule that caught the
  buffer defects in `DEFECTS.md`.
* Depth is swept 1 / 2 / 4 / 8. If the curve does not flatten by 8, the steady state
  has not been reached and the number is a lower bound on cost, reported as such.

## The second half: bracketing the crossing

The break-even the pipeline implies is **10.96 GB/s**, not the 3.82 GB/s of the
serial path, and the host-staged link here runs at 19.87 GB/s per direction — one
side of the crossing only. The pod's `/workspace` is a MooseFS mount
(`mfs#eu-se-1.runpod.net:9421`), so a genuinely slower path is already present, and
**offloading a KV cache to storage is one of the three scenarios this project is
about** rather than a stand-in for one.

So the same harness runs with its transport swapped from the pinned host ring to
files on `/workspace`, and on the container disk. Nothing else changes. The
prediction is that the decision **inverts** on the slower path, and the point of
running it inside one harness is that nothing but the path is different when it does.
