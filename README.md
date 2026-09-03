# BoundRelay

**When does error-bounded compression pay for LLM state movement?**

A measurement study of disaggregated prefill–decode KV transfer, KV migration and
cache offloading. The codec is a controlled variable, not the contribution.

> **Status: closed.** Every claim below is measured, and every claim this project
> has withdrawn is marked in place in the document that made it and listed in
> `docs/corrections.md`. Superseded work is under `results/public/superseded/`,
> not deleted. Nothing here is a placeholder.

## The question

Components benchmarked independently predict one thing; a real KV cache moving
end to end does another. This project measures the gap and what causes it — and
the largest single cause turned out not to be the codec.

## What is measured

| | |
|---|---|
| Model | Qwen3-1.7B, revision `b9352fbb`, bf16, 28 layers, 8 KV heads |
| Corpus | 24 independent WikiText-2 articles → **1152 tensor observations**, nested in documents |
| Bounds | `eps_i = c · std_i`, `c ∈ {0.01, 0.03, 0.10}` |
| Compressors | cuSZp (3 modes) · SZ3 (28-configuration scan) · CUDA zfp · int8 baselines |
| Transport | one real 56-tensor, 234.9 MB full cache, host-staged, 2×A40 and 2×A6000 |
| Quality | 16 documents, 1024→128 tokens, real cuSZp round trip ending in bf16 |
| Spend | **≈ $2** of GPU rental |

## Findings

**Peer-to-peer GPU copies report success and transfer nothing.**
`can_device_access_peer` returns `True`, `cudaMemcpyPeer` returns `cudaSuccess`,
and the destination is left entirely zero — both directions, 4 B through 32 MB, ten
of ten attempts, fully synchronised. Through host memory it is exact. **Three of
three rented pods**: two GPU models, two data centres, and both interconnect
topologies a rented pair comes in (PXB and SYS). A KV transfer built the obvious
way moves zeros between GPUs and reports that it worked.
→ `results/public/peer_copy_findings.md`

**Models built from separately measured parts are optimistic, twice over.**
A component model — codec throughput from one 20 MB tensor, link from another
session — was **4× optimistic**; per-tensor granularity was the cause. A stage-sum
model built from the *right* stages, on the real cache, on the right hardware was
still **2× optimistic** about a real pipeline: predicted 21.43 ms, measured 43.65,
because a stage sum contains no per-item cost. Measured on one host, compressed is
**1.91×** raw serial and **1.97×** pipelined — pipelining moves both arms and
neither past the other.
→ `results/public/c_transport_findings.md`, `pipeline_findings.md`

**Overlapping encode against decode cannot rescue it; a full pipeline is not
tested.** Measured cross-GPU overlap is **53.7%** with two host threads (IQR
46.6–62.1) and **1.4%** with one, because cuSZp's compress blocks the caller
reading `cmpSize` back to a host pointer. Two-stage overlap tops out at 1.50× raw.
A seven-stage pipeline is bound by the busiest resource instead — GPU0, at 21.43 ms
against a raw path of 23.64 ms — which moves the break-even from **3.82 to 10.96
GB/s**. Implied by measured stage costs, **not measured**; it is the open question.
→ `results/public/gate2_findings.md`

**Keys and values are not equally safe to compress.** At payloads within 0.4% of each
other, K-only compression degrades perplexity significantly (ΔNLL +0.0228, 2 of 16
documents improving); V-only *improves* it — ΔNLL −0.0323, CI [−0.0382, −0.0258],
**16 of 16 documents**, growing with the bound. Reported as observed, not explained.
Compressing both at `c = 0.10` ships **0.33×** the bytes with no detectable cost.
→ `results/public/d_quality_findings.md`

**SZ3's predictor costs ratio on values and pays on keys.** Given every configuration
`pysz` can reach — 28 of them, chosen per tensor as an oracle — no-prediction still
beats the best predictor on **28 / 28 value tensors** at every `c`, and loses on 22 of
28 key tensors. The mechanism is measured: adjacent-difference std over tensor std is
√2 (white) along channels and heads for both kinds, and **0.49 for K against 1.12 for
V along the token axis**. The cache carries correlation on one axis, mostly for keys.
→ `results/public/a3_sz3_full_findings.md`

**Characterization.** cuSZp is flat on this corpus — about 10% spread across every
layer, both kinds and four sequence lengths. Request metadata alone (layer, K/V,
sequence length, ε) predicts the compressed size to **within 1%**; a pass over the
tensor buys a further 11%.
→ `results/public/b0_corpus_findings.md`, `b1_predictability_findings.md`

**An undocumented API contract invalidated three findings.** cuSZp requires zeroed
buffers on both sides — it reads past `cmpSize` and does not write elements it
expects to be zero — and `torch.empty` put the allocator's leftovers into the
reconstruction. Re-measured: **0 bound violations**, worst error exactly 1.000× ε.
→ `results/public/remeasure_findings.md`, `DEFECTS.md`

## What is not established

* One model, one corpus, one GPU generation (`sm_86`), one link speed.
* The break-even (3.82 GB/s) is **implied by measured costs, not measured** — no
  experiment sits on the other side of the crossing.
* The peer-copy failure is characterised, not diagnosed: cause and prevalence both
  need host-level access a rented pod does not have.
* The SZ3 mechanism result rests on one document's full 28-layer cache; the depth
  axis is complete, the document axis is not.
* Payload corruption, bit flips and retransmission are out of scope.

## Corrections

Twenty-one claims have been withdrawn or conditioned, each marked in place in the
document that made it. `docs/corrections.md` is the register.

## Reproduction

| level | what | needs a GPU? |
|---|---|---|
| **L0** | `uv run pytest` — codec property tests, both suites | no |
| **L1** | `scripts/predictability.py`, `scripts/sz3_full_scan.py` from the committed manifests | no |
| **L2** | `scripts/capture_corpus.py`, `transport_e2e.py`, `quality_d.py` per `docs/environment.lock.md` | yes |
| **L3** | `scripts/gate2_rerun.py`, `scripts/peer_copy_audit.py` | two GPUs |

```bash
uv sync --extra dev && uv run pytest
```

## Layout

```
boundrelay/codec/          contract, reference, allocation, packing, Triton kernel, cuSZp bridge
boundrelay/policy/         break-even cost model, decision rule
boundrelay/bench/          Table A, Table B, quality
scripts/                   capture, benchmarks, transport, analysis
docs/                      corrections, environment lock, budget ledger, codec contract
results/public/            findings and sanitised JSON — current claims
results/public/superseded/ replaced, withdrawn and early-phase work, markers intact
```
