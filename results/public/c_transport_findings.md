# Phase C — the composed cost model was 4× optimistic, and would have decided wrong

One real full KV cache: **56 tensors, 234.9 MB bf16**, from a 2048-token prefill
(document `d00`). 2×A40, host-staged transport, cuSZp `fixed`, `eps_i = c·std_i`.
Everything a receiver waits for is inside one timing region. Raw:
`results/public/c_transport/`.

## The measurement

| | median | verdict |
|---|---|---|
| raw transfer, host-staged | **23.47 ms** (10.01 GB/s) | — |
| cuSZp c = 0.10 `fixed` (3.02×) | **54.94 ms** | bypass, −31.5 ms |
| cuSZp c = 0.03 `fixed` (2.29×) | 57.02 ms | bypass, −33.6 ms |
| cuSZp c = 0.01 `fixed` (1.87×) | 61.79 ms | bypass, −38.3 ms |

Every cell says bypass. Compressing is **2.3× slower than moving the cache raw.**

## Where the time goes

| component | median | share | throughput |
|---|---|---|---|
| bf16 → fp32 upcast | 1.43 ms | 2.9% | 164 GB/s |
| **cuSZp encode** | **19.89 ms** | **40.9%** | 11.8 GB/s |
| transfer (compressed bytes) | 8.99 ms | 18.5% | 8.64 GB/s |
| **cuSZp decode** | **17.07 ms** | **35.1%** | 13.8 GB/s |
| fp32 → bf16 downcast | 1.27 ms | 2.6% | 185 GB/s |
| **total** | **48.65 ms** | | |

Compression does save transfer: 8.99 ms against the raw path's 23.47 ms. It costs
36.96 ms of codec to save 14.48 ms of transfer.

**Measured break-even: 4.83 GB/s.** The link is 10.01 GB/s, twice that.

## Two things this corrects

**The dtype conversion is 6.8% of codec-side time, not 37%.** The withdrawn
estimate had the conversion roughly right — 1.43 ms measured against ~1.35 ms
predicted at this size — and the *codec* wrong by 5×. It is a real cost and worth
removing with a bf16-native path, but it is not where the time is.

**Per-tensor granularity costs 3.4× of codec throughput.** A single 20 MB tensor
benchmarks at ~40 GB/s; the same codec over a real cache of 56 tensors averaging
4.2 MB manages 11.8 GB/s encode, 0.355 ms each. A KV cache is not one array, and
measuring the codec as though it were is what made the composed model optimistic.

## Model validation, which is what this phase existed for

The composed model — codec cost from a single large tensor, link speed from a
separate session, conversion omitted — predicted break-even at **19.6 GB/s** and
a **+0.86 ms benefit** at 8.45 GB/s.

Measured: break-even **4.83 GB/s**, and at 10.01 GB/s a **−25 ms loss**.

**The model was 4× optimistic and its decision was wrong.** Composing
independently-measured components did not survive contact with a real path, and
the error was not in any single component — it was in assuming the components
compose at the granularity they were each measured at.

## Also observed: the peer path is pathological on a third machine

`raw` over direct device-to-device copy took **4.7 s** for the same 234.9 MB —
0.05 GB/s, ~190× slower than host-staging, with `can_device_access_peer` returning
True. This now holds on three separate rented machines across two topologies
(`SYS` cross-NUMA, and `PXB` same-NUMA), so it is a property of this class of
virtualised host rather than of one bad allocation. Host-staged is the only usable
inter-GPU path in every A40 pair measured.

## Limits

* One cache, one document, one GPU pair, one link speed. The break-even is
  measured here, not derived.
* Host-staged only: the same-hardware path isolation this phase wanted needs a
  pair where P2P works, and none of the A40 pairs rented had one.
* A fourth pod was terminated mid-phase with a hardware-faulted GPU 0; no data
  from it is used.
