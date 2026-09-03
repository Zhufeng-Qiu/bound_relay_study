# A3 — SZ3 configuration scan: prediction costs ratio on KV tensors

24 captured Qwen3-1.7B KV tensors, `eps_i = c · std_i`, ratios against bf16 bytes.
`pysz` on CPU, **$0**. Raw: `results/public/superseded/d14_table_b/sz3_config_scan.json`.

> **Superseded by `../a3_sz3_full_findings.md`.** Two things this scan stated as its
> own limits turned out to carry most of the result. It scanned three `cmprAlgo`
> values out of the space `Config::loadcfg` reaches, and it ran on the *superseded*
> corpus — an 81-token greedy-decode cache from an unpinned revision, not the Gate
> B0 prefill capture. Against the whole reachable grid on Gate B0 data the margin
> below falls from **14.4% to 2.9%**, and it stops being a property of KV cache:
> no-prediction beats every predictor on 28 of 28 **value** tensors and loses on 22
> of 28 **key** tensors. The numbers below reproduce exactly; the conclusion drawn
> from them was too broad.

SZ3 was authored by one of the groups this work is being sent to, so reporting a
number produced by whatever `pysz` defaults to would *under*-report their system —
worse than over-reporting it. This scan exists to remove that liability, and it
found something more interesting than a tuning delta.

## Tuning beats the default, as expected

Held-out documents (`p2`, `p3`); configuration chosen on tuning documents
(`p0`, `p1`) only, one global choice per `c`, frozen across K/V, layer and length.

| algorithm | c = 0.01 | c = 0.03 | c = 0.10 |
|---|---|---|---|
| LORENZO_REG | **2.08×** | **2.86×** | **4.58×** |
| INTERP_LORENZO *(pysz default)* | 2.01× | 2.70× | 4.18× |
| INTERP | 1.95× | 2.61× | 3.91× |

`LORENZO_REG` beats the default by **+3.6% / +5.8% / +9.5%**. Reporting the
default would have understated SZ3 by up to a tenth. Per-tensor best (2.10× /
2.88× / 4.62×) is barely above the frozen global choice, so nothing is lost by
refusing to pick per tensor — and picking per tensor would not be deployable
anyway, so it is reported as an oracle and never used as the coordinate.

## Turning prediction off beats every predictor

`NOPRED` was included as a sanity baseline, on the assumption it would lose. It
does not:

| layout | LORENZO_REG | INTERP_LORENZO | INTERP | NOPRED |
|---|---|---|---|---|
| native `[H,S,D]` (contiguous byte order) | 4.41× | 4.14× | 4.14× | **5.05×** |
| token-major `[T,D]` | 4.62× | 4.21× | 3.91× | **5.11×** |

*(c = 0.10, all 24 tensors)*

**No-prediction quantisation plus entropy coding compresses these tensors 14.4%
better than the best predictor**, in the layout that matches their actual memory
order — and 10.4% better in the permuted view Table B had been using. The result
is not a layout artefact, which was the first thing checked: Table B had been
feeding SZ3 a `permute(0,2,1,3)` view rather than the contiguous `[B,H,S,D]`
order, and a mismatched layout is exactly how prediction would be made to look
useless whether or not it is.

So SZ3's advantage over a plain blockwise quantiser on this data comes from its
quantisation and entropy coding, **not from the prediction stage that is the
centre of the SZ design**. On KV cache the predictor is not merely neutral; it
costs ratio.

## What this does and does not establish

Established, on this corpus: prediction reduces compression ratio at matched
error, in both layouts tested, at every `c`, on held-out documents.

Not established: **why**. Candidate explanations — the values are close to
independent once SZ3's own blocking is applied; the correlation that does exist
lies along an axis neither layout exposes; the error bound is loose enough that
prediction residuals quantise to the same codes as the raw values. Distinguishing
these needs the statistical characterisation in Phase B, and none of them should
be asserted before it.

This also partially rehabilitates an intuition retracted in D14 — that scientific
compressors carry an inductive bias these tensors do not reward. D14 retracted it
because SZ3 beat this project's codec outright. That retraction stands: SZ3 does
win. But the win is now located in the entropy stage rather than the predictor,
which is a narrower and testable version of the original claim.

## Limits

* Four documents. Phase A validates the script, that each configuration takes
  effect, and the runtime budget; the **final** configuration choice waits for the
  Gate B0 corpus, because four documents cannot select a configuration that holds.
* `pysz` Tier-1 algorithms only. Interpolation-algorithm, predictor-flag and
  block-size settings sit behind `loadcfg(INI)` and are not scanned here.
* Ratio only. SZ3 is CPU here and contributes no throughput number by design.
