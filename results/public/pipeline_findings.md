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

## The control, and what it caught

Depth 1 was registered as the control on the grounds that a depth-1 pipeline is a
serial loop with bookkeeping. It came out at 119.93 ms against a 50.48 ms
reference, and the right response to a failed control is not a wider gate. A
depth-1 pipeline hands every tensor between two threads with a host synchronisation
on each side; the reference runs all 56 encodes, then all 56 transfers. They are
not the same computation, so the reference was rebuilt **inside** the harness:

| stage-major serial, in this harness | measured | reference | |
|---|---|---|---|
| compressed | 53.35 ms | 50.48 ms | 1.057× — **OK** |
| raw | 27.87 ms | 23.64 ms | 1.179× — **fails the ±15% gate** |

The compressed arm reproduces. The raw arm does not, and the reason is a defect in
the baseline rather than in the harness. `raw_iter` copies every tensor into
`host[:nb]` and then into `recv[0]` — **the same two buffers, 56 times.** Fifty-five
of the fifty-six results are overwritten before anything reads them, so the raw
baseline never pays for touching the 470 MB of distinct memory a receiver that
keeps its data must touch.

That is the shape of defect D2, which was found and fixed in the compressed path
and left standing in the path it is compared against. See `DEFECTS.md` D3.

**The raw baseline is therefore optimistic, and every ratio quoted against it is
too large.** Serial, like for like inside one harness: **53.35 / 27.87 = 1.91×**,
not the 2.14× that comes from dividing by the shared-buffer number.

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
