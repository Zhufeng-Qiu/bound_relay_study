# Three defects found by audit, and what they invalidate

All three are in this project's measurement code, not in the systems being
measured. Recorded before the re-run so the record shows what was believed and why
it was wrong.

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

## D3 — the raw baseline discards 55 of its 56 results, and it costs nothing

`raw_iter` in `session_close.py` reads, in full:

```python
for x in cache:
    nb = x.numel()*2
    host[:nb].copy_(x.reshape(-1).view(torch.uint8)); recv[0][:nb].copy_(host[:nb])
```

`host[:nb]` and `recv[0]` — the same host slot and the same device buffer, 56 times.
Every tensor overwrites the last, so 55 of the 56 arrive nowhere. That is the shape
of D2, fixed in the compressed path and left standing in the path it is compared
against, and nothing caught it because nothing checks the raw path's output: there
is nothing to decompress, so no correctness gate ever ran on it.

**It is a correctness defect and not a timing one, and the first version of this
entry got that wrong.** This entry originally claimed the shared buffers made the
baseline 18% cheaper and that every ratio quoted against it was inflated. Measured
directly, on one host, same bytes, differing only in where they land:

| variant | ms | GB/s |
|---|---|---|
| shared host + shared device — `raw_iter` as written | 28.18 | 8.34 |
| distinct host slots, shared device | 27.22 | 8.63 |
| **distinct host + distinct device — a receiver that keeps its data** | **25.94** | **9.05** |

Writing to distinct buffers is **8% faster**, not slower. The reuse is not buying
the baseline anything; if anything the repeated write to one hot destination
serialises against itself.

So where did the 18% come from? The host. `raw_iter` re-run on *this* pod takes
**28.18 ms** against the 23.64 ms recorded on the pod it was first measured on, and
this harness's rebuilt stage-major loop takes 27.87 ms — **agreeing with the
original code to 1.1% when both run on the same machine.** The control passes. The
gap was a machine, and this entry blamed a buffer for it.

What stands: `raw_iter` does not deliver 55 of the 56 tensors it claims to move, and
a transport benchmark whose receiver never receives is not measuring a transport.
Its *timing* is representative, so no ratio changes. It should still be fixed, and
the pipeline harness's raw arm — which does deliver every tensor and is checked
against the source — is the one to quote from here.

## What is not affected

* The corpus itself: 24 documents, 1152 observations, statistics, shapes, hashes.
* Compression ratios throughout, and everything built on them — B1 predictability,
  the SZ3 configuration scan, the NOPRED finding.
* Gate 2's overlap measurement, which uses neither the error fields nor the staging
  buffer.
* The quality results, which go through a separate round-trip path
  (`roundtrip_` in `quality_d.py`) that allocates per call and feeds the
  reconstruction straight back into the model.
