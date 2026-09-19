# B2 — the sign is set by the path, and pairing makes that claim survivable

2×A40 (sm_86, `PXB`), driver 570.195.03, cuSZp `f581dcf3` `fixed`, `c = 0.10`.
Four caches (d00, d01 × 1024, 2048 tokens), three paths, raw against compressed:
**12 paired configurations × 30 pairs = 360 pairs, 720 timed runs**, in three
segments of the session. Raw: `b2/path_trials.jsonl`, `b2/path_summary.json`,
`b2/path_checks.jsonl`.

Every earlier performance number in this project compared a raw loop against a
compressed loop measured at a different moment on a shared machine. This pairs
them: each measurement is a raw run and a compressed run back to back, half the
pairs in each order, shuffled, so machine drift moves both arms together instead of
becoming the result. `R = exp(mean(log(T_compressed / T_raw)))`, `R < 1` means
compression is faster.

## Twelve configurations, twelve determinate answers

| cache | path | raw | compressed | **R** | 95% CI |
|---|---|---|---|---|---|
| d00 L1024 | serial | 16.37 ms | 54.04 ms | 3.471 | [3.312, 3.706] |
| d00 L1024 | pipeline | 19.22 | 42.20 | 2.145 | [1.932, 2.306] |
| d00 L1024 | **fsync** | 269.15 | 208.29 | **0.657** | [0.566, 0.749] |
| d00 L2048 | serial | 29.29 | 60.46 | 2.096 | [2.040, 2.159] |
| d00 L2048 | pipeline | 29.40 | 42.95 | 1.535 | [1.463, 1.667] |
| d00 L2048 | **fsync** | 504.01 | 233.64 | **0.664** | [0.480, 0.969] |
| d01 L1024 | serial | 16.31 | 55.47 | 6.147 | [3.850, 11.123] |
| d01 L1024 | pipeline | 19.20 | 42.91 | 2.326 | [2.236, 2.440] |
| d01 L1024 | **fsync** | 279.33 | 216.75 | **0.743** | [0.686, 0.802] |
| d01 L2048 | serial | 29.60 | 62.64 | 2.165 | [2.109, 2.229] |
| d01 L2048 | pipeline | 29.04 | 43.45 | 1.522 | [1.479, 1.574] |
| d01 L2048 | **fsync** | 501.92 | 248.92 | **0.508** | [0.470, 0.554] |

**No interval crosses 1.** Compression is slower on both GPU-to-GPU paths in all
four inputs and faster on the filesystem write in all four, at a payload of
0.330× raw throughout.

## The segments agree

Pooled by path, per segment:

| path | segment 1 | segment 2 | segment 3 | all |
|---|---|---|---|---|
| serial | 2.774 | 4.216 | 2.640 | 3.137 |
| pipeline | 1.782 | 1.880 | 1.883 | 1.848 |
| fsync | 0.543 | 0.665 | 0.714 | 0.637 |

**All three segments agree with the pooled direction on all three paths.** The
magnitudes do not hold still — serial swings 2.64–4.22 and the fsync advantage
erodes steadily across the session, which is what a shared mount and a shared host
look like. Pairing is why the direction survives that; an unpaired comparison of
segment-2 compressed against segment-1 raw would have said something different.

`d01 L1024 | serial` has the one wide interval, [3.850, 11.123]. It is reported as
measured. Slow samples were not removed.

Bootstrapped **within each segment** as well — 36 intervals, recomputed offline from
`path_trials.jsonl`, no GPU (`b2/path_summary_by_segment.json`):

| path | segment 1 | segment 2 | segment 3 | all four caches on one side of 1 |
|---|---|---|---|---|
| serial | 2.11 – 3.69 | 2.13 – 18.83 | 2.05 – 3.37 | yes, all three segments |
| pipeline | 1.48 – 2.30 | 1.50 – 2.44 | 1.52 – 2.26 | yes, all three segments |
| fsync | 0.46 – 0.72 | 0.48 – 0.83 | 0.57 – 0.80 | yes, all three segments |

Every one of the 36 point estimates falls on the expected side of 1. **Two of the 36
intervals cross it** — `d00 L2048 | fsync` in segments 2 and 3, at 10 pairs each,
where the interval is at its widest. Both still point the same way (R = 0.83 and
0.77); neither reverses. The pooled configuration-level result, 30 pairs per
configuration, is determinate in all twelve.

Segment 2's serial upper end of 18.83 is the `d01 L1024` outlier again, in the
segment where it was worst.

## Pipelining halves the penalty and does not remove it

Pooled R falls from 3.137 serial to 1.848 pipelined. **That is not a 1.7× speedup
of the compressed path**, and an earlier draft read it that way. Each R divides by
its own raw baseline, and the two baselines are not the same — at L1024 the raw path
is *slower* pipelined than serial (16.3 → 19.2 ms), so part of the drop in R is the
denominator moving, not the numerator.

Compared directly, compressed time serial → pipelined:

| cache | compressed | | raw, for reference |
|---|---|---|---|
| d00 L1024 | 54.04 → 42.20 ms | **1.28×** | 16.37 → 19.22 ms (0.85×) |
| d00 L2048 | 60.46 → 42.95 ms | **1.41×** | 29.29 → 29.40 ms (1.00×) |
| d01 L1024 | 55.47 → 42.91 ms | **1.29×** | 16.31 → 19.20 ms (0.85×) |
| d01 L2048 | 62.64 → 43.45 ms | **1.44×** | 29.60 → 29.04 ms (1.02×) |

Pipelining is worth **1.28–1.44×** on the compressed path, not 1.7×, and it still
leaves it 1.5–2.3× slower than moving the bytes. The gap narrows with cache size — at L2048 pipeline sits at
1.52–1.54 against serial's 2.10–2.17 — which is the fixed per-cache cost amortising,
not the per-byte cost improving.

## Delivery, not the last kernel launch

The move endpoint is **all 56 independent bf16 outputs simultaneously live on
GPU1**, not the final launch. Upcast, encode, the zeroing B0 showed is mandatory,
D2H, host scheduling, H2D, decode, downcast and the closing synchronisation are all
inside the clock. The write endpoint is **payload written, `fsync` returned, file
closed**.

**144 verification checks, all passing**: raw delivered byte-identical on both
transport paths and read back byte-identical from the file; compressed within its
fp32 bound on readback and within ε plus the measured bf16 rounding on delivery.

That last criterion is an **aggregate** — the worst error over a cache against the
largest ε in that cache — and it is weaker than it should be, since a tensor with a
small ε can be wrong and still pass under a larger one. **These 1,792 per-tensor comparisons are a separate verification pass, not something
every timed run went through.** Each timed run was bracketed by the block-level
checks described above; the per-tensor comparison was run afterwards, on the same
code, as a stronger check of the same transport. It establishes that the path
delivers exactly — it does not retroactively certify each of the 720 timed runs
individually. It was replaced after these
timings were taken, by a per-tensor byte comparison against fresh single-device
baselines: **1,792 comparisons, all byte-exact**, reported in `b0_findings.md`.
The timings here are unaffected — that change is to the verifier, not to the timed
path — but the correctness claim these runs support is the stronger per-tensor one,
not the aggregate stated above.
Each check runs the arm it verifies — an earlier version checked whichever arm
happened to run last, which read a raw file as a compressed stream and crashed the
decoder.

## What this does not establish

* **These are warm replays of a prepared cache.** Layout, offsets, ε and buffer
  allocation sit outside the clock. The cost of meeting a cache for the first time
  is not measured, so none of this is the latency of a live request.
* **`fsync` is a write, not a restore.** Nothing is decoded on that path. Cold
  restore — reading back after eviction on a client holding nothing — is not
  measured here.
* One host, one mount, one session. The three segments describe drift *within* a
  session; they say nothing about another machine or another day. The fsync mount
  is MooseFS over the data-centre network and its numbers belong to it.
* 30 pairs per configuration supports a direction and a median. It does not support
  a tail claim, and these are not 30 independent documents.
