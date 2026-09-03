# The fsync-acknowledged offload write

1×A40, RunPod `EU-SE-1`, torch 2.8.0+cu128. 56-tensor d00 full cache, 234.9 MB bf16,
`c = 0.10`, cuSZp `fixed`. Raw: `results/public/pipeline/fsync_offload.json`.
Script: `scripts/fsync_offload.py`.

**What is measured is a write.** `fsync` returning means the filesystem has
acknowledged the bytes. It is not a round trip: cold restore — reading the cache
back after eviction, on a client whose cache holds nothing — is **not measured
here**, and the read side of this mount is served warm at around 3 GB/s, so nothing
in this document should be read as a restore cost.

## Why this experiment exists

`pipeline_findings.md` put the cache on a MooseFS mount and found compression
**losing** — 1.17× slower than raw while moving a third of the bytes. Those writes
were never made durable. A separate bandwidth sweep with `fsync` on the same kind
of mount gave 0.43 GB/s, where a third of the bytes ought to be worth a great deal,
and the two projected rows in that document's bracket table said compression should
win 3×. Projected, from bytes over bandwidth — which is precisely the model the
measurement next to them had just broken.

So the offload itself is measured, and both regimes are measured **on one mount, in
one process**, because "durability changes the answer" cannot rest on comparing two
data centres.

## Protocol

Two arms, identical in every respect except the payload:

* **raw** — GPU source, D2H, into a single laid-out cache file, one `fsync`.
* **compressed** — upcast, encode, D2H of the *actual compressed bytes*, into the
  same single-file layout by the same code path, one `fsync`.

Both preallocate the file to their own payload size before the clock starts, both
build the offset table by the same function, both write tensor by tensor at those
offsets, and both call `fsync` exactly once for the whole cache. The clock runs
from **GPU source ready to `fsync` returning**, so the compressed arm is charged its
upcast and its encode.

Two axes are crossed with that:

* **writes per cache** — 56, one per tensor at its offset, against 1 for the whole
  contiguous payload. The difference is per-item cost.
* **durability** — `fsync` against no `fsync`, the latter reproducing the regime
  the pipeline run used.

Before any timing, each arm is written and read back: raw must return the source
byte for byte, compressed must decode inside its error bound. An offload that does
not deliver what it was given is not an offload, and this project has already
shipped one transport benchmark whose receiver never received (`DEFECTS.md` D3).

## Compression wins every configuration, and never by what the bytes say

The cache compresses **3.02×** — 234.9 MB to 77.7 MB. Medians of 7 timed passes
after a verified pass and a warm pass:

| filesystem | writes | `fsync` | raw | compressed | **compression is** |
|---|---|---|---|---|---|
| MooseFS `/workspace` | 56 | yes | 563.1 ms | 354.6 ms | **1.59× faster** |
| MooseFS | 1 | yes | 461.6 ms | 382.3 ms | **1.21× faster** |
| MooseFS | 56 | no | 709.7 ms | 337.7 ms | **2.10× faster** |
| MooseFS | 1 | no | 526.7 ms | 277.8 ms | **1.90× faster** |
| container overlay `/root` | 56 | yes | 130.0 ms | 62.9 ms | **2.07× faster** |
| container overlay | 1 | yes | 131.4 ms | 65.4 ms | **2.01× faster** |
| container overlay | 56 | no | 71.9 ms | 42.1 ms | **1.71× faster** |
| container overlay | 1 | no | 74.5 ms | 48.0 ms | **1.55× faster** |

Raw returned byte for byte in all eight; compressed decoded at 0.9999 × ε in all
eight.

**Compression pays on every one, and on none of them by 3.02×.** A model that
divides the byte reduction by an unchanged bandwidth predicts 3.02× everywhere and
is short by **1.44× to 2.50×**:

| | predicted | measured |
|---|---|---|
| MooseFS, 56 writes, `fsync` | 186.3 ms / 3.02× | 354.6 ms / **1.59×** |
| MooseFS, 1 write, `fsync` | 152.7 ms / 3.02× | 382.3 ms / **1.21×** |
| overlay, 56 writes, `fsync` | 43.0 ms / 3.02× | 62.9 ms / **2.07×** |
| overlay, 1 write, no `fsync` | 24.7 ms / 3.02× | 48.0 ms / **1.55×** |

## Why: a smaller write is given a slower path

The compressed arm never achieves the bandwidth the raw arm does — it gets
**0.40× to 0.70×** of it, on both filesystems, in every configuration. That is not
a property of compressed data. Writing incompressible noise through the same call
pattern, the same layout and the same `fsync`, and changing **only the volume**:

| filesystem | `fsync` | 77.7 MB | 234.9 MB | the small payload gets |
|---|---|---|---|---|
| MooseFS | yes | 0.255 GB/s | 0.468 GB/s | **0.55×** |
| MooseFS | no | 0.416 GB/s | 0.507 GB/s | 0.82× |
| overlay | yes | 1.820 GB/s | 2.097 GB/s | 0.87× |
| overlay | no | 4.092 GB/s | 4.313 GB/s | 0.95× |

**Achieved bandwidth is a function of how much you are writing**, and most strongly
so on the network filesystem under durability — exactly the configuration an offload
runs in. So "bytes ÷ bandwidth" is not merely missing a term; its denominator
depends on its numerator.

The remainder is the codec, and the accounting closes. MooseFS, 56 writes, `fsync`:
a bare 77.7 MB write costs 304.7 ms, the compressed arm costs 354.6 — the 50 ms
difference is the upcast, the encode and the D2H. On the overlay the same
subtraction leaves 20 ms. Both are what those stages cost measured on their own.

## Durability is real, second order, and does not decide the sign

`fsync` changes the ratio in **opposite directions on the two filesystems**: on
MooseFS it shrinks compression's advantage (2.10 → 1.59 and 1.90 → 1.21), on the
container overlay it grows it (1.71 → 2.07 and 1.55 → 2.01). It never flips the
sign. Whatever durability is doing here, it is not the thing that decides whether
compression pays, and a claim that it is would not survive the second filesystem.

## What this does and does not establish

* **It is a write.** `fsync` returning is the filesystem's acknowledgement. **Cold
  restore is not measured** — no eviction, no cold client, and the read side of
  this mount serves warm at about 3 GB/s. Nothing here is a round-trip number.
* **One mount, one hour, one pod.** `/workspace` is shared infrastructure and the
  spread is wide: single passes ranged 243–943 ms where the median was 355. Seven
  reps and medians, but this is not a quiet machine.
* **This does not reconcile with `pipeline_findings.md`, and should not be made
  to.** That run found compression *losing* 1.17× on a MooseFS mount — a different
  data centre, a two-thread ring, and, decisively, a **round trip**: it wrote, read
  back, and decoded on a second GPU. This is a write. The two measure different
  quantities and only one of them is named in this document's title.
* The container overlay is backed by whatever the host gives it; 2–4 GB/s is that
  host's, not a property of overlayfs.
