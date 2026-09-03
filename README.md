# BoundRelay

**When does error-bounded compression pay for LLM state movement?**

A measurement study of disaggregated prefill–decode KV transfer, KV migration and
cache offloading. The codec is a controlled variable, not the contribution.

> **Status: scaffold.** No measurements yet. Every `[X]` below is a placeholder.
> Nothing enters this README, the CV, or any email until it traces back to a
> committed manifest, a commit SHA, and a raw JSON file in `results/public/`.

## The research question

Scientific data compressors earn their ratio from spatial smoothness. LLM
intermediate states are not smooth along the token axis. So:

> For non-smooth LLM intermediate states, what codec design and what
> payload × bandwidth regime let error-bounded compression pay for its own
> overhead?

The answer is a boundary, not a speedup. Whether moving compressed state beats
moving raw state depends on the payload, the link, and what the codec costs on
the hardware in hand — and across the operating points measured so far, most sit
outside the profitable regime.

## What is out of scope

- Not a new codec. cuSZp is the strongest tested baseline and beats this
  project's reference implementation on both ratio and speed; the reference codec
  exists because the harness was developed against it.
- Not an online controller. The decision study is offline; a predictor that uses
  only pre-compression information is evaluated separately.
- Not a serving-runtime integration, and not multi-node.
- Not a quality guarantee. An element-wise error bound bounds tensors, not output.
- Not a resilience study. Payload corruption, bit flips and retransmission are
  explicitly out of scope.

## Results

| | Result | Where |
|---|---|---|
| Codec | `[X]` GB/s encode / `[Y]` GB/s decode at `[Z]`× on `[GPU]` | `results/public/table_a.json` |
| Coordinate | benchmarked against cuSZp, SZ3, CUDA zfp under matched error settings | `results/public/table_b.json` |
| Boundary | compression pays below `[B]` GB/s effective bandwidth | Figure 1 |
| Quality | perplexity delta `[d]` at eps `[e]` | Figure 3 |

## Error contract

See [`docs/codec_contract.md`](docs/codec_contract.md). In short: `max|x - x̂| ≤ ε`
on finite blocks; NaN/Inf blocks travel bit-exact, are counted, and are outside
the guarantee; every fallback is `bf16_passthrough` with a recorded reason.

## Reproduction

| Level | What | Needs a model? |
|---|---|---|
| **L0** | `pytest` — codec property tests, both suites | no |
| **L1** | download the sanitised tensor sample, redraw one break-even curve | no |
| **L2** | rebuild baseline + one adaptive cell per `docs/environment.lock.md` | yes, 2×H100 |

```bash
uv sync --extra dev
uv run pytest
```

## What did not work

`results/public/` carries the failed and excluded configurations alongside the
successful ones. A sweep that only publishes its best cell is not a measurement.

## Layout

```
boundrelay/codec/        contract, reference, allocation, packing, Triton kernel
boundrelay/trace/        edge capture + replay manifest
boundrelay/policy/       break-even cost model + decision log
boundrelay/integration/  SGLang-Omni transform hook (stretch)
boundrelay/bench/        Table A, Table B, atlas, quality
tests/                   two separate suites: finite bound, non-finite bypass
docs/                    contract, environment lock, budget ledger, edge budget
results/public/          sanitised JSON, committed
results/private_raw/     never committed
```
