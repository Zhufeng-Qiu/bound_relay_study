# BoundRelay

**When does error-bounded compression pay for LLM state movement?**

A measurement study of disaggregated prefill–decode KV transfer, KV migration and
cache offloading. The codec is a controlled variable, not the contribution.

> **Status: measurements in hand, evidence closure incomplete.** Phases A–E are
> run and committed. Three things must land before this is a finished artifact:
> the end-to-end decomposition does not sum to the measured path (11.4%
> unaccounted), the break-even is model-implied rather than measured, and the
> asymmetric-bound experiment the quality result points at has not been run.
> Everything below traces to a manifest; nothing is a placeholder.

## The question

Independently benchmarked codec and transport components predict one thing; a real
KV cache moving end to end does another. This project measures the gap and what
causes it.

## What is measured

| | |
|---|---|
| Model | Qwen3-1.7B, revision `b9352fbb`, bf16, 28 layers, 8 KV heads |
| Corpus | 24 independent WikiText-2 articles → **1152 tensor observations**, nested in documents |
| Bounds | `eps_i = c · std_i`, `c ∈ {0.01, 0.03, 0.10}` |
| Compressors | cuSZp (3 modes) · SZ3 (pilot configuration scan) · CUDA zfp · int8 baselines |
| Transport | one real 56-tensor, 234.9 MB full cache, host-staged, 2×A40 |
| Quality | 16 documents, 1024→128 tokens, real cuSZp round trip ending in bf16 |
| Spend | **$1.90** |

## Findings

**Characterization.** cuSZp is flat on this corpus — about 10% spread across every
layer, both kinds and four sequence lengths. Request metadata alone (layer, K/V,
sequence length, ε) predicts the compressed size to **within 1%**; a pass over the
tensor buys a further 11%.
→ `results/public/b0_corpus_findings.md`, `b1_predictability_findings.md`

**A component model predicted the wrong decision.** Codec cost measured on a
single 20 MB tensor and a link measured in a separate session predicted a benefit.
The real 56-tensor cache moves raw in 23.47 ms and compressed in 54.94 ms —
**2.3× slower**, every cell bypass. Per-tensor granularity costs **3.4×** of codec
throughput (40 GB/s on one 20 MB tensor, 11.8 GB/s over the real cache).
→ `results/public/c_transport_findings.md`

**Keys and values are not equally safe to compress.** At payloads within 0.4% of
each other, K-only compression degrades perplexity significantly (ΔNLL +0.869,
CI [+0.692, +1.041]); V-only shows no detectable degradation (ΔNLL −0.008, CI
[−0.023, +0.008]). Compressing both gently beats compressing one hard: uniform at
c = 0.01 ships **0.53×** the bytes for **+0.09%** perplexity.
→ `results/public/d_quality_findings.md`

**SZ3's prediction stage costs ratio here.** Turning prediction off beats every
predictor by **14.4%** in the contiguous layout — checked in both layouts, so not
an artefact. The SZ family's advantage on this data is in quantisation and entropy
coding. *Pilot scan: four documents, block size not swept.*
→ `results/public/a3_sz3_config_findings.md`

## What is not established

* One model, one corpus, one GPU generation, one link speed.
* The break-even (3.42–4.25 GB/s depending on what counts as codec cost) is
  **implied by measured costs, not measured** — no experiment sits on the other
  side of the crossing.
* Phase E measured same-GPU encode/encode concurrency, not the cross-GPU pipeline
  it stood in for. No conclusion about pipelined transport follows.
* Asymmetric bounds (`c_K ≪ c_V`), which the quality result points at, are untested.
* Payload corruption, bit flips and retransmission are out of scope.

## Corrections

Eleven claims have been withdrawn or conditioned, each marked in place in the
document that made it. `docs/corrections.md` is the register.

## Reproduction

| level | what | needs a GPU? |
|---|---|---|
| **L0** | `uv run pytest` — codec property tests, both suites | no |
| **L1** | `scripts/predictability.py`, `scripts/sz3_config_scan.py` from the committed manifests | no |
| **L2** | `scripts/capture_corpus.py`, `transport_e2e.py`, `quality_d.py` per `docs/environment.lock.md` | yes |

```bash
uv sync --extra dev && uv run pytest
```

## Layout

```
boundrelay/codec/     contract, reference, allocation, packing, Triton kernel, cuSZp bridge
boundrelay/policy/    break-even cost model, decision rule
boundrelay/bench/     Table A, Table B, quality
scripts/              capture, benchmarks, transport, analysis
docs/                 corrections, environment lock, budget ledger, codec contract
results/public/       findings and sanitised JSON
```
