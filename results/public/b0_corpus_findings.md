# B0 / B2 — frozen corpus, and a variance claim that did not survive its own protocol

Qwen3-1.7B pinned at `b9352fbb`, transformers 5.16.1, 1×A40. **24 independent
WikiText-2 articles**, prefixes at 64 / 256 / 1024 / 2048 tokens, layers
0/5/10/14/20/27, K and V separate → **1152 tensor observations**, plus three
28-layer full caches (56 tensors each) for the transport experiment.
Raw: `results/public/b0_corpus/`.

Documents are real articles cut on the corpus's own `= Title =` markers.
Concatenating WikiText and slicing it into equal lengths would manufacture
"documents" that share content and defeat the grouping that every split depends
on. **The independent unit is the document, not the tensor** — six layers × two
kinds × four lengths all come from one article.

Capture is a single `model(input_ids, use_cache=True)` prefill. The earlier corpus
used `generate(max_new_tokens=64)` from short prompts, which produces a
decode-time cache rather than the prefill cache a disaggregated transfer moves.

## cuSZp on the real corpus is remarkably flat

Ratio against bf16 bytes, mean over 24 documents, `eps_i = c · std_i`:

| kind | layer | c = 0.01 | c = 0.03 | c = 0.10 |
|---|---|---|---|---|
| K | 0 | 2.01× | 2.48× | **3.33×** |
| K | 14 | 1.87× | 2.29× | 3.01× |
| K | 27 | 1.83× | 2.23× | 2.94× |
| V | 0 | 1.87× | 2.28× | 3.02× |
| V | 14 | 1.93× | 2.38× | 3.18× |
| V | 27 | 1.88× | 2.30× | 3.04× |

*(`fixed` mode; `plain` and `outlier` shown in the raw manifest)*

Across every layer, both kinds and all four sequence lengths the spread is about
**10%**. Layer-0 keys compress best and key ratios fall slightly with depth; V is
flatter with a small bump at layer 14. This is much less structure than the
earlier 24-tensor allocation study suggested — but that study asked a different
question (where *this project's* codec should spend its error budget), and this
one measures what cuSZp achieves outright.

`plain` and `outlier` produced **identical ratios to two decimals** on every cell.
Either they coincide on this data or the difference is below reporting precision;
either way they are not two independent options here. `fixed` is consistently
2–6% better.

> **The error columns in this manifest are not usable; the byte counts are.**
> Every `max_error_fp32`, `max_error_bf16` and `within_eps_fp32` field in
> `b0_corpus/manifest.json` was measured against a `torch.empty` destination,
> before cuSZp's zeroed-buffer requirement was known. 374 of 1152 observations
> (32.5%) carry a `within_eps_fp32: false`, and 168 of them (14.6%) report the
> *same* error across all nine (c, mode) cells — impossible for a real
> measurement, and the signature of a reconstruction left over from the previous
> call. The remaining 984 scale correctly (median `err(0.10)/err(0.01)` = 10.00).
> `error_bound_audit.py` re-measured the bound against zeroed buffers and found
> **0 violations in 56 tensors at every mode and every c**.
>
> `cmp_bytes` is an encoder output and never passed through the poisoned buffer:
> it differs between c = 0.01 and c = 0.10 on all 1152 observations. **Every ratio
> in the table above stands.** See `remeasure_findings.md`.

## The "3–6× fixed-mode variance" was an artefact

D14b reported cuSZp's `fixed` mode showing 3–6× encode-time spread across
processes at tight ε. Under a protocol built to distinguish the candidate
explanations — 10 fresh processes, 20 warmup + 100 measured iterations,
pre-allocated buffers, mode and ε order randomised within each process, first
call after context creation recorded separately — **that claim does not hold.**

| | |
|---|---|
| Cold start (first call after CUDA context creation) | **305 ms** |
| Warm median, over all 24 cells | **0.88 ms** |
| Ratio | **347×** |

Warm medians are flat: every cell lands between 0.83 and 0.96 ms regardless of
tensor, ε or mode. What varies is the **tail**, and it varies for both modes and
all tensors — between-process CV ranges 0.07–1.59 with spreads up to 15×, and
within-process CV reaches 1.2.

So the earlier number was measuring context creation plus a heavy-tailed
measurement environment, not a property of `fixed` mode. Within-process CV above
1 points at interference on a shared, virtualised GPU rather than at codec
behaviour. **`fixed` and `plain` cost the same at the median**; the mode choice
should be made on ratio, where `fixed` leads by 2–6%.

The one number that survives and matters for the transport model: **cold start is
347× the warm cost**, so any deployment measurement that does not amortise context
creation will be dominated by it.

## Limits

* One model, one corpus, one GPU type. Ratios are what cuSZp achieves here, not a
  property of KV cache in general.
* Latency is measured on a shared cloud GPU. The medians are stable; the tails
  are not attributable to the codec, and no tail claim is made from them.
* Compression ratio is measured on every observation; timing only on the six
  stratified tensors of this protocol, to avoid a hidden full cartesian product.
