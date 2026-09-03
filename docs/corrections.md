# Corrections

Every claim this project has withdrawn, why, and what replaced it. Retractions
are marked in place in the original documents rather than edited away — the
record of what was believed and why it was wrong is part of the evidence.

| # | Claim | Withdrawn because | Where |
|---|---|---|---|
| 1 | Generic error-bounded compressors underperform on LLM state because their inductive bias assumes one exploitable structure | SZ3 compresses the same tensors 5× better than this project's codec; its prediction works fine on KV cache | `d8_kv_findings.md`, retracted by D14 |
| 2 | The advantage is implementation, not algorithm — a simple codec that maps onto a GPU beats a better one that does not | cuSZp is a better algorithm *and* maps onto a GPU: 2–3× the ratio at ~9× the speed | `d14_table_b_findings.md`, retracted by D14b |
| 3 | Cross-tensor mean compression ratios at a fixed absolute ε | Per-tensor std spans 115× across this corpus, so a shared absolute ε is a different bound on every tensor; the means are not comparisons | `d13`, `d14`, `d14b`, `d20`, corrected in Phase A |
| 4 | Break-even at 19.6 GB/s | Stated unconditionally. It excludes bf16↔fp32 conversion and is unvalidated below 4 MB, the smallest payload the link was measured at | `d9_codec_findings.md`, conditioned in Phase A |
| 5 | Dtype conversion adds 37% | 37% is the surcharge on encode+decode; as a share of codec-side time it is 27%. Both come from memory-bandwidth arithmetic — **no GPU timing exists** | withdrawn in Phase A, measurement moved to Phase C |
| 6 | An online controller bypasses in 69% of cases | The controller reads measured compressed bytes, so it is an offline oracle; 69% is the share of equal-weighted sweep cells, not of requests | `d21_policy_findings.md`, relabelled in Phase A |
| 7 | SZ3's ratio (at the `pysz` default configuration) | The default under-reports SZ3 by up to 9.5% against a documented scan | `a3_sz3_config_findings.md` |
| 8 | cuSZp `fixed` mode has 3–6× cross-process time variance | Measured cold start, not the mode. All modes share a ~0.88 ms warm median; cold start is 347× that, and the tails belong to a shared virtualised GPU | `b0_corpus_findings.md`, withdrawn by B2 |

Correction 3 is the one that reaches furthest: it invalidates aggregates in four
documents at once. Correction 5 is the one that was caught before it reached
anything published.
