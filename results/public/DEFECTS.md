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

## D3 — the raw baseline discards 55 of its 56 results

D2 was found and fixed in the compressed path. The path it is compared against
still has it. `raw_iter` in `session_close.py` reads, in full:

```python
for x in cache:
    nb = x.numel()*2
    host[:nb].copy_(x.reshape(-1).view(torch.uint8)); recv[0][:nb].copy_(host[:nb])
```

`host[:nb]` and `recv[0]` — the same host slot and the same device buffer, 56 times.
Every tensor overwrites the last, so 55 of the 56 reconstructions never exist, and
the baseline never pays for touching the 470 MB of distinct memory a receiver that
keeps its data has to touch. The compressed path was given per-tensor offsets when
D2 was fixed; this one was not, because nothing checks the raw path's output — there
is nothing to decompress and so no correctness gate ever ran on it.

Rebuilt with distinct buffers on both sides, the same 56 tensors take **27.87 ms
against the baseline's 23.64 ms**, and the compressed arm of the same rebuild
reproduces its reference to 1.057×. So the gap is in the raw arm, and it is 18%.

**Every ratio quoted against the raw baseline is too large by that factor.** The
end-to-end comparison is **1.91×, not 2.14×**; the serial break-even and everything
derived from it move with it. The direction of every conclusion is unchanged — the
compressed path is still about twice the raw one — but the number was flattering
the wrong side, which is the direction that matters least for this project's
conclusions and most for its credibility.

Found by building a control for a different experiment. The pipeline harness
rebuilt the serial reference inside itself so a depth-1 pipeline could be checked
against it, and the compressed arm agreed while the raw arm did not.

## What is not affected

* The corpus itself: 24 documents, 1152 observations, statistics, shapes, hashes.
* Compression ratios throughout, and everything built on them — B1 predictability,
  the SZ3 configuration scan, the NOPRED finding.
* Gate 2's overlap measurement, which uses neither the error fields nor the staging
  buffer.
* The quality results, which go through a separate round-trip path
  (`roundtrip_` in `quality_d.py`) that allocates per call and feeds the
  reconstruction straight back into the model.
