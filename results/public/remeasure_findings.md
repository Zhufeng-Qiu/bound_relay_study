# Re-measurement — one undocumented API contract invalidated three findings

2×A40, driver 570.195.03, cuSZp `f581dcf3`, Xeon Gold 6342. $1.07.
Raw: `results/public/remeasure/`.

## The root cause: cuSZp requires zeroed buffers

Two requirements, neither stated in its headers, both found by poisoning buffers
and seeing what survived:

* **The decode destination must arrive zeroed.** cuSZp does not write elements it
  expects to be zero — 128 to 2912 per tensor here, and the count *rises with the
  error bound*: 5 of 56 tensors leave something unwritten at c = 0.01, **50 of 56 at
  c = 0.10**.
* **The compressed-data buffer must be zeroed too.** The decoder reads past
  `cmpSize`; leftovers there make it return an all-zero reconstruction silently.

With `torch.empty`, the allocator's previous contents land in the reconstruction.
Nothing raises, nothing warns, and the looser the bound the more of it there is.

## What this invalidated

### 1. cuSZp does not violate its error bound

The corpus reported violations on 16.7% / 22.9% / **32.5%** of tensors, perfectly
deterministic per layer. Re-measured against zeroed buffers:

| mode | c = 0.01 | c = 0.03 | c = 0.10 | worst error / ε |
|---|---|---|---|---|
| plain / outlier / fixed | **0/56** | **0/56** | **0/56** | **1.000×** |

cuSZp meets its bound exactly and saturates it. The earlier violations were the
allocator's leftovers in positions the decoder correctly declined to write.

### 2. The K/V interaction does not exist

The asymmetric grid reported a super-additive interaction — configurations that were
individually harmless collapsing to perplexity 6191 in combination — and a contrast
of +5.77. Re-measured:

| | 2×2 contrast | 95% CI |
|---|---|---|
| c_K = 0.005, c_V = 0.10 | **+0.0012** | [−0.0030, +0.0052] |
| c_K = 0.10, c_V = 0.10 | **−0.0027** | [−0.0081, +0.0021] |

Both cross zero. **There is no interaction.** The collapse was garbage in unwritten
elements, and it appeared at loose c_V precisely because that is where the decoder
leaves the most unwritten.

### 3. Compression is far gentler than reported

| c_K | c_V | payload | vs raw | ΔNLL | 95% CI |
|---|---|---|---|---|---|
| 0.10 | None | 78.0 MB | 0.66× | **+0.0228** | [+0.0109, +0.0361] * |
| None | 0.10 | 78.3 MB | 0.67× | −0.0323 | [−0.0382, −0.0258] * |
| 0.01 | 0.03 | 56.9 MB | 0.48× | −0.0019 | [−0.0062, +0.0021] |
| 0.01 | 0.10 | 50.7 MB | 0.43× | −0.0341 | [−0.0394, −0.0280] * |
| **0.10** | **0.10** | **38.8 MB** | **0.33×** | −0.0122 | [−0.0255, +0.0013] |

Nothing is catastrophic. **Compressing both kinds at c = 0.10 ships a third of the
bytes with no detectable degradation.**

The K/V asymmetry survives in direction and collapses in magnitude: K-only
compression is the one configuration that significantly *degrades* quality, and by
0.023 NLL — about 0.15% of perplexity, not the 138% previously reported.

Several rows show small but significant **improvements**. Mild quantisation noise
acting as regularisation is a known effect, but it is reported here as an
observation, not a claim: an improvement from lossy compression deserves more
scrutiny than this experiment gives it.

## The corrected end-to-end

With every buffer zeroed, per-tensor staging offsets, and a correctness gate that
verifies `max |x − x̂_bf16| ≤ ε + bf16 half-ulp` on a full iteration **before** timing:

| | median | P95 |
|---|---|---|
| raw | 23.64 ms | 25.32 ms |
| compressed | **50.48 ms** | 159.45 ms |

| stage | median |
|---|---|
| bf16 → fp32 upcast | 1.443 ms |
| cuSZp encode | 19.985 ms |
| D2H | 4.315 ms |
| H2D | 5.013 ms |
| **zero destination** | **0.928 ms** |
| cuSZp decode | 15.008 ms |
| fp32 → bf16 downcast | 1.344 ms |
| GPU sum | 48.036 ms |
| host residual | 2.448 ms (4.8%) |

Zeroing the destination is now a stage: it is work the receiver must do because the
codec requires it, and hiding it outside the timed region would understate the path.

**Break-even 3.82 GB/s against a 9.94 GB/s link — bypass**, and the contract
delivered is `ε + bf16 half-ulp`, not ε: the measured 2.14062 against ε = 2.13547
overshoots by 0.24% for exactly that reason.

## Method note

The correctness gate first failed with `inf`, which turned out to be the gate
itself: it compared a GPU0 tensor against a GPU1 reconstruction across the peer
path, and that path is pathological on this hardware. **A correctness check routed
through a broken transport measures the transport.** It compares on the host now.

## What still stands

* The corpus: 24 documents, 1152 observations, statistics, shapes, hashes.
* Compression ratios everywhere — they come from `cmpSize`, which compress writes
  directly, and they differ correctly across modes.
* B1 predictability, the SZ3 configuration scan, and the NOPRED result, none of
  which use a cuSZp reconstruction.
* The transport conclusion: compression loses on this link, by a wider margin than
  before (3.82 GB/s against 9.94).

## What is now open

Gate 2 still reports serialisation (encode 0.365 ms, decode 0.278 ms, together
0.653 ms against a 0.644 ms serial sum) — but its decode ran into a `torch.empty`
destination, so it may have been timing a call that wrote nothing. **That number
needs re-running before it is quoted.**
