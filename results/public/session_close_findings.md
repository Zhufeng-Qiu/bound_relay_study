# Closing session — the gaps are closed, and one of them changed the answer

One host, one environment lock, everything below measured on it. 2×A40 (`PXB`,
same NUMA), **driver 570.195.03**, cuSZp `f581dcf3`, Intel Xeon Gold 6342, torch
2.8.0+cu128. Pod terminated; $0.67. Raw: `results/public/session_close/`.

Earlier phases spanned four machines and recorded no driver version. This session
exists because three results rested on measurements that could not be reproduced
from what was committed.

> **Partly superseded.** The E2E numbers here (46.84 ms, break-even 4.06 GB/s) came
> from a run whose staging buffer overwrote itself, so 55 of 56 receivers got the
> wrong payload; and its decode wrote into `torch.empty`. Corrected figures —
> raw 23.64 ms, compressed 50.48 ms, break-even 3.82 GB/s, with a correctness gate
> and zeroing counted as a stage — are in `remeasure_findings.md`.
>
> **Gate 2's serialisation result is suspect** for the same reason: its decode
> destination was uninitialised, so it may have timed a call that wrote nothing.
> Do not quote the 30% overlap figure until it is re-run.
>
> The SZ3/zfp same-host timings and the storage probe are unaffected.

## 1. The 6.29 ms gap was methodology, not overhead

Phase C timed each stage in its own loop and added the medians, leaving 11.4% of
the path unattributed. Timed with CUDA events **inside the same iteration**, with
all 100 samples kept:

| stage | median |
|---|---|
| bf16 → fp32 upcast | 1.429 ms |
| cuSZp encode | 19.576 ms |
| D2H | 4.079 ms |
| H2D | 4.028 ms |
| cuSZp decode | 16.020 ms |
| fp32 → bf16 downcast | 1.286 ms |
| **GPU stage sum** | **46.418 ms** |
| **measured end to end** | **46.840 ms** |
| **host residual** | **0.427 ms — 0.9%** |

The stages account for **99.1%** of the path. The earlier gap was the error of
adding medians drawn from separate distributions, not Python dispatch as
hypothesised. **Host-side dispatch across 56 tensors costs 0.43 ms, not 6.29.**

## 2. Break-even, now closed

`bytes_saved / (E2E − transfer stages)` = **4.06 GB/s**, against a measured link of
**11.24 GB/s**. Bypass, by 2.8×.

Still model-implied in one respect that matters: **only one link speed was ever
exercised**, so no measurement sits on the other side of the crossing. The
decomposition is now closed; the boundary is not yet bracketed.

Also worth recording: **P95 is 173.3 ms against a 46.8 ms median.** The tail on a
shared virtualised GPU is 3.7× the median, which no cost model of this kind
captures.

## 3. Gate 2: cross-GPU overlap exists, and is far too small

Phase E measured same-GPU encode/encode — a question the pipeline never asks — and
inferred device ownership from kernel signatures. Measured properly:

| | |
|---|---|
| encode on GPU0 alone | 7.46 ms |
| decode on GPU1 alone | 6.01 ms |
| issued from one host thread | 17.06 ms |
| **issued from two host threads** | **11.65 ms** |
| serial sum | 13.47 ms |
| perfect overlap would be | 7.46 ms |

**Overlap efficiency 30%**, and only with separate host threads — single-threaded
issuance is *worse* than the serial sum, because the compress call blocks the
caller while it reads the compressed size back to a host pointer.

A pipelined transport needs **95%** overlap efficiency to reach the raw baseline on
this hardware. 30% is not close. **The pipeline question is now settled by
measurement rather than by inference from a header file**, and the answer is that
it cannot flip the decision here.

## 4. SZ3 and zfp on this host, not a laptop

`d14_table_b_findings.md` compared Apple-M5 CPU timings against A40 GPU timings.
Re-measured on this session's Xeon, 12 tensors, single-threaded:

| | c = 0.01 | c = 0.03 | c = 0.10 | throughput @ c = 0.10 |
|---|---|---|---|---|
| SZ3 `LORENZO_REG` | 2.33× | 3.11× | 4.92× | 0.064 GB/s |
| **SZ3 `NOPRED`** | **2.56×** | **3.42×** | **5.52×** | **0.104 GB/s** |
| zfp | 1.30× | 1.48× | 1.76× | 0.077 GB/s |

**`NOPRED` beats the best predictor on both ratio (+12%) and speed (1.6×)**,
replicating the A3 result on a second CPU architecture. Turning SZ3's prediction
off is not a quirk of one machine.

The earlier caveat that Mac numbers were "a floor" was wrong in direction: this
Xeon is *slower* per core than the M5 for SZ3.

## 5. No local NVMe exists here

`/workspace` is `mfs#eu-se-1.runpod.net:9421` — **network storage**. `/` is a 50 GB
overlay. There is no raw local NVMe on these pods, so the cache-offload experiment
cannot be run on this provider without renting a different instance type. Recorded
so it is not attempted again.
