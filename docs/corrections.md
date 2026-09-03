# Corrections

Every claim this project has withdrawn, why, and what replaced it. Retractions
are marked in place in the original documents rather than edited away — the
record of what was believed and why it was wrong is part of the evidence.

| # | Claim | Withdrawn because | Where |
|---|---|---|---|
| 1 | Generic error-bounded compressors underperform on LLM state because their inductive bias assumes one exploitable structure | SZ3 compresses the same tensors better than this project's codec. But the reason was misattributed: A3 later found its *prediction* stage costs ratio here (NOPRED beats every predictor by 14.4%), so the advantage lies in quantisation and entropy coding | `d8_kv_findings.md`, retracted by D14 |
| 2 | The advantage is implementation, not algorithm — a simple codec that maps onto a GPU beats a better one that does not | cuSZp is a better algorithm *and* maps onto a GPU: 2–3× the ratio at ~9× the speed | `d14_table_b_findings.md`, retracted by D14b |
| 3 | Cross-tensor mean compression ratios at a fixed absolute ε | Per-tensor std spans 115× across this corpus, so a shared absolute ε is a different bound on every tensor; the means are not comparisons | `d13`, `d14`, `d14b`, `d20`, corrected in Phase A |
| 4 | Break-even at 19.6 GB/s | Stated unconditionally. It excludes bf16↔fp32 conversion and is unvalidated below 4 MB, the smallest payload the link was measured at | `d9_codec_findings.md`, conditioned in Phase A |
| 5 | Dtype conversion adds 37% | 37% is the surcharge on encode+decode; as a share of codec-side time it is 27%. Both come from memory-bandwidth arithmetic — **no GPU timing exists** | withdrawn in Phase A, measurement moved to Phase C |
| 6 | An online controller bypasses in 69% of cases | The controller reads measured compressed bytes, so it is an offline oracle; 69% is the share of equal-weighted sweep cells, not of requests | `d21_policy_findings.md`, relabelled in Phase A |
| 7 | SZ3's ratio (at the `pysz` default configuration) | The default under-reports SZ3 by up to 9.5% against a documented scan | `a3_sz3_config_findings.md` |
| 8b | Break-even 4.83 GB/s, and "measured" | Computed as `raw_bytes / total_time`, which is not a break-even. The quantity is `bytes_saved / codec_cost` → 3.42–4.25 GB/s depending on what counts as codec cost — and it is model-implied, not measured: only one link speed was exercised | `c_transport_findings.md` |
| 8c | "180× different quality cost" between K and V | The denominator's CI crosses zero, so the ratio is arbitrary. K-only degrades significantly; V-only shows no detectable degradation | `d_quality_findings.md` |
| 8d | "The one mechanism able to flip the decision is unavailable" | The gate measured same-GPU encode/encode concurrency, not the cross-GPU pipeline it stood in for | `e_pipeline_findings.md` |
| 9 | cuSZp violates its error bound on up to 32.5% of tensors | It does not. cuSZp requires zeroed buffers — it skips elements it expects to be zero and reads past `cmpSize` — and `torch.empty` put the allocator's leftovers in the reconstruction. Measured correctly: 0 violations, worst error exactly 1.000× ε | `remeasure_findings.md` |
| 10 | K and V interact super-additively; contrast +5.77 | Same cause. The collapse appeared at loose `c_V` because that is where the decoder leaves the most unwritten. Corrected contrasts +0.0012 and −0.0027, both crossing zero: **no interaction** | `asymmetric_kv_findings.md`, withdrawn in full |
| 11 | K-only +138% vs V-only −0.76% perplexity | Direction survives, magnitude does not: K-only costs +0.0228 NLL, ~0.15% of perplexity, and both kinds at c = 0.10 ship 0.33× the bytes with no detectable degradation | `d_quality_findings.md` |
| 12 | Closed E2E 46.84 ms, break-even 4.06 GB/s | The staging buffer overwrote itself, so 55 of 56 receivers got the wrong payload. Corrected: raw 23.64 ms, compressed 50.48 ms, break-even 3.82 GB/s | `session_close_findings.md` |
| 13 | Cross-GPU overlap efficiency 30%, encode 7.46 ms, decode 6.01 ms | Re-measured over 56 tensors with zeroed buffers and a correctness gate: **53.7%** two-threaded (IQR 46.6–62.1), 1.4% single-threaded. The two absolute times were totals of 20 iterations — per call they are 0.373 and 0.301 ms. The conclusion survives on a stronger footing: overlap can save at most 15.01 ms where 26.84 ms is needed, so **no efficiency reaches the raw baseline** | `session_close_findings.md`, superseded by `gate2_findings.md` |
| 14 | The B0 manifest's per-observation cuSZp error columns | Measured against `torch.empty` before the zeroed-buffer contract was known. 32.5% carry a false `within_eps_fp32`; 14.6% report an identical error across all nine (c, mode) cells, which no real measurement does. Re-measured properly: 0 violations. **`cmp_bytes` and every ratio derived from it are unaffected** | `b0_corpus_findings.md` |
| 8 | cuSZp `fixed` mode has 3–6× cross-process time variance | Measured cold start, not the mode. All modes share a ~0.88 ms warm median; cold start is 347× that, and the tails belong to a shared virtualised GPU | `b0_corpus_findings.md`, withdrawn by B2 |

Correction 3 is the one that reaches furthest: it invalidates aggregates in four
documents at once. Correction 5 is the one that was caught before it reached
anything published.
