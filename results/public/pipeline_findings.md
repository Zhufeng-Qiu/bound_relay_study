# The multi-stage pipeline, measured — and the stage-sum model is optimistic by 2×

2×A40 (`sm_86`), RunPod `CA-MTL-1`, torch 2.8.0+cu128. 56-tensor d00 full cache,
234.9 MB bf16, `c = 0.10`, cuSZp `fixed`. Raw: `results/public/pipeline/`.
Script: `scripts/pipeline_multistage.py`. Predictions fixed in advance:
`pipeline_prereg.md`.

Gate 2 overlapped encode on GPU0 against decode on GPU1 — two stages of seven — and
the conclusion drawn from it did not follow (correction 20). Grouping the measured
stages by resource said a real pipeline would be GPU0-bound at **21.43 ms**, against
a raw path of 23.64 ms, and would move the break-even from 3.82 to 10.96 GB/s.

That was arithmetic over stage costs measured in a serial loop. This is the pipeline.

## The control, and what it took to make it mean something

Depth 1 was registered as the control on the grounds that a depth-1 pipeline is a
serial loop with bookkeeping. It came out at 119.93 ms against a 50.48 ms
reference. The right response to a failed control is not a wider gate: a depth-1
pipeline hands every tensor between two threads with a host synchronisation on each
side, while the reference runs all 56 encodes and then all 56 transfers. Not the
same computation. So the reference was rebuilt **inside** the harness:

| stage-major serial, this harness | measured | reference, other pod | |
|---|---|---|---|
| compressed | 53.35 ms | 50.48 ms | 1.057× |
| raw | 27.87 ms | 23.64 ms | 1.179× — outside the ±15% gate |

The raw arm still missed, and the first explanation offered here was wrong. It
blamed `raw_iter` for copying every tensor into the same host slot and the same
device buffer — 55 of 56 results overwritten — and claimed that made the baseline
18% cheap. Measured directly:

| variant, same bytes, differing only in where they land | ms |
|---|---|
| shared host + shared device — `raw_iter` as written | 28.18 |
| distinct host slots, shared device | 27.22 |
| distinct host + distinct device — a receiver that keeps its data | **25.94** |

Distinct buffers are **8% faster**. The reuse buys nothing, and the 18% was the
machine: `raw_iter` **re-run on this pod takes 28.18 ms**, against 27.87 ms for the
rebuilt loop — **agreement to 1.1% once both run on the same host.**

**The control passes.** Everything below is measured on one host against a
reference re-measured on that host, and the ratios here are not comparable to
ratios computed from the earlier pod's absolute numbers.

The overwriting is still a defect — a transport benchmark whose receiver never
receives is not measuring a transport — and it is recorded as `DEFECTS.md` D3, with
the timing claim retracted in place.

## Pipelining works, and does not change the answer

Host-staged ring, depth swept, medians of 15 timed passes:

| depth | compressed | raw | ratio |
|---|---|---|---|
| 1 (tensor-major serial) | 119.93 ms | 42.41 ms | 2.83× |
| 2 | 41.98 ms | 22.53 ms | 1.86× |
| 4 | 44.41 ms | 21.84 ms | 2.03× |
| **8** | **43.65 ms** | **22.19 ms** | **1.97×** |
| *stage-major serial, same harness* | *53.35 ms* | *27.87 ms* | *1.91×* |

The curve flattens by depth 2, so the steady state is reached. Every reconstruction
holds its bound at every depth: `max|x − x̂|` = 2.141 against a delivered bound of
3.135.

**Pipelining buys 1.22× on the compressed path and 1.26× on the raw one, and the
ratio between them does not move: 1.91× serial, 1.97× pipelined.** Whatever
pipelining is worth here, it is worth it to both sides, and the decision is the
same one on either.

## The stage-sum model was optimistic by about 2×, on both arms

| | stage-sum model | measured | |
|---|---|---|---|
| pipelined compressed | 21.43 ms | **43.65 ms** | 2.04× the model |
| pipelined raw | 11.82 ms | **22.19 ms** | 1.88× the model |
| pipelining gain, compressed | 2.49× | **1.22×** | |
| pipelining gain, raw | 2.36× | **1.26×** | |

This is the third time in this project that a model built from separately measured
parts has been optimistic about the whole, and the third time in the same
direction. Phase C found a component model — codec throughput from one 20 MB
tensor, link from another session — was 4× optimistic, and located the cause in
per-tensor granularity. This one takes stage costs measured *on the real cache, in
the right order, on the right hardware* and still overshoots by 2×, because a stage
sum contains no per-item cost: the handoffs, the host synchronisations, and the
work the two threads do to each other's devices are simply not in it.

The per-thread occupancy shows there is no idle resource left to recover. At depth
8 the compressed pipeline has the producer busy 37.22 ms and the consumer 38.85 ms,
with 2.11 ms and 0.67 ms of waiting respectively — **both threads are saturated**,
so the 43.65 ms is not a scheduling failure with headroom in it.

## Does a transfer overlap the codec on one device? Mostly

The pre-registration named a mechanism that would have capped the pipeline: cuSZp
calls `cudaMalloc` and `cudaFree` around every kernel, and `cudaFree` synchronises
the device, which would forbid a D2H overlapping the encode before it.

| | |
|---|---|
| encode alone | 0.436 ms |
| D2H alone | 0.230 ms |
| both, issued together on one device | **0.543 ms** |
| serial sum | 0.666 ms |
| the larger alone | 0.436 ms |

**53.5% of the transfer hides under the codec.** Not free overlap and not full
serialisation — the allocations cost something, and less than the whole transfer.

The first version of this probe reported encode alone at 3.945 ms and both at
0.534 ms, which is impossible because `both` calls `enc`. Timing the three variants
back to back let cuSZp's cold start — which B2 measured at 347× the warm median —
land on whichever ran first. The probe now warms twenty rounds, interleaves the
three inside one loop, takes medians, and refuses to report a verdict when `both`
comes out below the larger of its parts. The 0.436 ms it now reports for encode
agrees with the reference's 0.357 ms per tensor; the 3.945 ms never did, and
nothing checked.

## Storage, where the break-even was supposed to be on the other side

The pipelined break-even is far below the host-staged path, so the run swapped its
transport for files on `/workspace` — a MooseFS mount — twice: once with a file per
tensor, once with one file kept open and seeked into. Offloading a cache to storage
is one of the three scenarios this project is about, so this is a missing
measurement rather than a stand-in for a slow link.

Best depth for each, medians of 15 passes:

| transport | compressed | raw | |
|---|---|---|---|
| host-staged ring | 43.65 ms | 22.19 ms | raw, 1.97× |
| MooseFS, 56 files | 1976.83 ms | 1721.69 ms | **raw, 1.15×** |
| MooseFS, one file, seeked | 396.03 ms | 338.45 ms | **raw, 1.17×** |

**Compression loses on every path measured, including the slow ones, and on the
slow ones it loses while moving a third of the bytes.** Opening a file per tensor
costs 4.3× more than seeking within one — 1976.83 against 396.03 on identical bytes
— which is per-object cost, and a break-even argument counts only bytes so it
cannot see that at all. It is the granularity result again, at a third level.

But the pipeline's writes are not durable, and an offload's have to be. Swept
separately on the same mount, with `fsync`:

| write size | MooseFS | container overlay |
|---|---|---|
| 64 KB × 3584 | 0.248 GB/s | 2.99 GB/s |
| 1 MB × 224 | **0.426 GB/s** | 3.08 GB/s |
| 4 MB × 56 | 0.409 GB/s | 2.90 GB/s |
| 235 MB × 1 | 0.298 GB/s | 2.81 GB/s |

At MooseFS's best durable rate the raw cache takes **551 ms** and the compressed one
**182 ms** — compression saves 369 ms, and **wins by 3×**. Reads are excluded from
that comparison: measured back to back with the write they run at 3 GB/s, which is
the client cache rather than the filesystem, and the number is reported as a lower
bound rather than used.

So the crossing is bracketed on both sides, and it took durability to find it:

| path | rate | verdict |
|---|---|---|
| host-staged ring | 10.58 GB/s | compression loses, 1.97× |
| **measured pipelined break-even** | **5.38 GB/s** | |
| container overlay, durable | 3.47 GB/s | compression pays — *implied by the sweep* |
| MooseFS, durable | 0.43 GB/s | compression pays, 3× — *implied by the sweep* |
| MooseFS as the pipeline used it, unsynced | — | compression loses, 1.17× — measured |

The two "implied" rows are arithmetic over a bandwidth sweep, not an offload: they
assume the cost is bytes over bandwidth, which is exactly the assumption the row
below them breaks. `fsync_offload_findings.md` measures the offload itself, on one
mount, with and without durability, and replaces them.

The last two rows are the same filesystem. What separates them is whether the write
is durable and whether it is issued as one object or fifty-six — neither of which
appears anywhere in a break-even expression, and both of which decide the answer.

## The break-even, measured rather than implied

| | link speed below which compression pays |
|---|---|
| serial, as first reported | 3.82 GB/s |
| pipelined, implied by the stage sum | 10.96 GB/s |
| **pipelined, measured** | **5.38 GB/s** |
| the host-staged path itself | 10.58 GB/s |

Pipelining does widen the window where compression pays — from 3.82 to 5.38 GB/s,
a factor of 1.4 — but not by the 2.9× the stage sum promised, and not far enough to
cross the path it runs on.

## What is not established

* **One host.** Every number here is from one 2×A40 pod. The reference numbers from
  the earlier pod differ by 6–19% on identical code, which is the size of several
  effects reported elsewhere in this project, so cross-pod absolute comparisons are
  not safe and none is made.
* **Two host threads.** The consumer runs H2D and decode in sequence, so the
  consumer side is charged more than the resource model allows it. At depth 8 the
  compressed run has the producer busy 37.22 ms and the consumer 38.85 ms with
  under 2.2 ms of waiting on either — both saturated, neither starved — so a third
  thread would redistribute rather than recover. That is an argument, not a
  measurement.
* **The storage reads are cache-warm.** They are reported as lower bounds and the
  bracket is built on the write side only.
* **`fsync` is the only durability model tested.** O_DIRECT, io_uring and a real
  object store are all different paths and none was measured.
* **Depth 8 is the deepest tested.** The curve flattens by depth 2 on every
  transport, so the steady state is reached, but nothing here speaks to a pipeline
  deep enough to hide a multi-second storage stall.
* The `/workspace` mount is shared infrastructure. Its 0.43 GB/s is what this pod
  saw at this hour, not a property of MooseFS.
