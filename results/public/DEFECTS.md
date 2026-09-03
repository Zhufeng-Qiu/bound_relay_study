# Two defects found by audit, and what they invalidate

Both are in this project's measurement code, not in the systems being measured.
Recorded before the re-run so the record shows what was believed and why it was
wrong.

## D1 — reconstruction errors were stale, so no cuSZp error-bound claim stands

`b0_corpus/manifest.json` reports `within_eps_fp32 = False` on 16.7% / 22.9% /
32.5% of tensors at c = 0.01 / 0.03 / 0.10, with some `max_error_fp32` exceeding ε
by more than 100×, and the failures perfectly deterministic per layer — every K
tensor at layers 5 and 10, none elsewhere.

That looked like cuSZp violating its bound. It is not, and the data says so:

| | across `fixed`, `plain`, `outlier` |
|---|---|
| compressed bytes identical | **0 of 1200** |
| max error identical | **1200 of 1200** |

Three modes produced **different compressed data and byte-identical reconstruction
errors.** That is impossible if each error came from its own reconstruction.

Cause: `dec = torch.empty(...)`. PyTorch's caching allocator returns the same block
across the nine calls made per tensor, so a decompress that wrote nothing left the
previous call's reconstruction in place, and the error measured belonged to that
earlier call.

**Consequences.** Every `max_error_fp32`, `max_error_bf16` and `within_eps_fp32` in
the frozen corpus is unreliable. No claim about cuSZp meeting or violating its
error bound can be made from it, in either direction, and the Table B and B0
comparisons cannot be called matched-error until the corpus is re-measured.

Compression **ratios are unaffected** — they come from `cmpSize`, which the compress
call writes directly, and they differ correctly across modes.

**Fix.** The destination is filled with NaN before decoding and checked afterwards;
a partial or absent decode now returns `None` for the error fields rather than a
number belonging to something else.

## D2 — the closed E2E did not transfer the right bytes

`session_close.py` staged all 56 tensors through `host[:s]`:

```python
for i, s in enumerate(sizes): host[:s].copy_(scr[i][:s])   # each overwrites the last
for i, s in enumerate(sizes): recv[i][:s].copy_(host[:s])  # all read the survivor
```

By the time the H2D loop ran the buffer held only the final tensor, so 55 of 56
receivers were given the wrong payload. The downcast result was discarded and the
reconstruction never checked, so nothing caught it.

**Consequences.** The 46.84 ms figure measures comparable work — the same kernels
over the same byte counts — but it is not a correct compressed transfer, and the
stage decomposition and the 4.06 GB/s boundary derived from it must be re-run
before being quoted.

**Fix.** Each tensor gets its own offset in the staging buffer, the final bf16
tensors are kept, and a correctness gate verifies `max |x − x̂_bf16| ≤ ε` on a full
iteration **before** any timing begins.

## What is not affected

* The corpus itself: 24 documents, 1152 observations, statistics, shapes, hashes.
* Compression ratios throughout, and everything built on them — B1 predictability,
  the SZ3 configuration scan, the NOPRED finding.
* Gate 2's overlap measurement, which uses neither the error fields nor the staging
  buffer.
* The quality results, which go through a separate round-trip path
  (`roundtrip_` in `quality_d.py`) that allocates per call and feeds the
  reconstruction straight back into the model.
