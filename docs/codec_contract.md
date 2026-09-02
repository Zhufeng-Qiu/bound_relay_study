# Codec contract

Frozen at D3, before any GPU spend. Enforced by two separate test suites.

## Guarantee

For a block in which **every value is finite**:

```
max |x - x_hat| <= eps + FP_TOLERANCE      (FP_TOLERANCE = 1e-6)
```

Block metadata (`min`, `scale`) is stored in **fp32**. In fp16 the metadata's own
representation error can exceed the bound the codec advertises.

## Explicit exclusions

| Case | Behaviour | Why |
|---|---|---|
| Block contains NaN or ±Inf | Transmitted **bit-exact**, counted in `nonfinite_blocks`, outside the eps guarantee | `\|x - x_hat\|` is undefined; folding these into the bound test would hollow it out |
| Unsupported dtype | `bf16_passthrough`, `BypassReason.UNSUPPORTED_DTYPE` | Fail closed |
| Non-contiguous layout | `bf16_passthrough`, `BypassReason.NON_CONTIGUOUS` | Fail closed |
| Tensor below block size | `bf16_passthrough`, `BypassReason.TENSOR_TOO_SMALL` | Partial-block arithmetic is where bounds quietly break |
| Post-encode bound check fails | `bf16_passthrough`, `BypassReason.BOUND_CHECK_FAILED` | Never ship data that violates the advertised contract |

## What is NOT guaranteed

**Output quality.** An element-wise error bound says nothing about WER,
perplexity, or speaker similarity. Those are measured separately against a gate
frozen before any performance sweep. Any claim of the form "error-bounded
therefore output-safe" is out of contract.

## Naming

The bypass path is `bf16_passthrough`. It is not "fp16": payloads are bf16, and
bf16 carries fp32's exponent range. Narrowing to fp16 overflows on outlier
channels — a documented property of LLM activations, not a hypothetical.

## Reporting

Compression ratio is always against **bf16 bytes (2 B/element)**. Baselines that
read fp32 (cuSZp, SZ3) consume a lossless bf16→fp32 upcast; reporting their ratio
against those fp32 bytes would hand each of them a free 2×.
