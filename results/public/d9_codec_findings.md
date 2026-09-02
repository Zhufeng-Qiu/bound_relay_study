# D9–D15 — GPU codec verified and measured (1×A40, CA-MTL-1)

torch 2.8.0+cu128, Triton 3.4.0, A40 sm_86. Session cost $0.12 (955 s).
Raw: `results/public/d9_codec/cost_table.json`.

## Verification before measurement

`tests/test_gpu_equivalence.py`: **16/16 pass**. Local suite: 108 pass.

The GPU and reference paths agree on every structural decision — identical
widths, identical raw-group sets, identical payload sizes — and either decoder
reads the other's payload inside the error bound.

They are **not** byte-identical, and that is recorded rather than tuned away.
With bit-identical float32 inputs, `range / 255` differs between CPU and CUDA by
one float32 ulp on 302 of 512 groups; the Triton reductions themselves matched
`scatter_reduce` exactly, so the divergence is the division. One ulp on a scale
flips values sitting on a rounding boundary, moving codes by ±1 — well inside
`eps`. Forcing agreement would mean computing every scale on the host and
shipping it to the device: a synchronisation added to satisfy a test rather than
a requirement. **The encoder is not bit-reproducible across devices**; that
belongs in the note's reproducibility caveat.

A second lesson came from the decision rule. Expressed as
`ceil(log2(range / (2·budget) + 1))`, a one-ulp wobble flips the `ceil` and the
two paths classified borderline groups differently. Asking the question directly
— *is half a quantisation step inside the budget?* — is one division, one
comparison, clearer, and stable enough that both paths agree.

## Codec cost (kernel-only)

Stats + quantise + pack, and unpack + dequantise. The research serialiser's
host-side assembly is excluded and reported separately: it is a Python loop over
groups, runs ~100× the kernel time, and blaming the codec for it would have put
encode two orders of magnitude below the link and made compression look
unconditionally worthless.

| bits | payload | ratio | encode | decode | codec total | break-even |
|---|---|---|---|---|---|---|
| 6 | 0.26 MB | 2.55× | 0.364 ms | 0.046 ms | 0.411 ms | 0.4 GB/s |
| 6 | 2.10 MB | 2.55× | 0.456 ms | 0.038 ms | 0.494 ms | 2.6 GB/s |
| 6 | 10.5 MB | 2.55× | 0.503 ms | 0.058 ms | 0.561 ms | 11.3 GB/s |
| 6 | 21.0 MB | 2.55× | 0.536 ms | 0.113 ms | 0.648 ms | **19.6 GB/s** |
| 8 | 21.0 MB | 1.93× | 0.812 ms | 0.130 ms | 0.943 ms | 10.7 GB/s |
| 4 | any | 0.98× | — | — | — | 0 |

At ε = 0.15, **4 bits cannot meet the bound at all** on these activations: every
group falls to the bit-exact path and the payload grows slightly (metadata). Six
bits beats eight — it meets the same bound with fewer bits, so the extra
precision of 8-bit is pure cost.

## The headline: both regimes measured, neither manufactured

At 6 bits on a 21 MB payload (a video-encoder embedding), against the two link
speeds measured in D6 on the same hardware:

| link | measured | net benefit |
|---|---|---|
| host-staged inter-GPU | 8.45 GB/s | **+0.859 ms — compression pays** |
| one-way PCIe | 23.0 GB/s | **−0.095 ms — compression loses** |

**Break-even: 19.6 GB/s**, bracketed by two measured points on real hardware. No
artificial bandwidth cap was needed to produce a slow-path regime, and no NVLink
was needed to produce a fast-path one.

The same boundary read along the payload axis, at the measured 8.45 GB/s link:
0.26 MB and 2.1 MB payloads lose (−0.39 ms, −0.34 ms); 10.5 MB and 21 MB win
(+0.19 ms, +0.86 ms). That axis is the one that matters here, because the two
real stage edges sit on opposite sides of it: Thinker→Talker carries ~4 KB per
token and must bypass, while an encoder→Thinker embedding carries 3–20 MB and
straddles the crossing.

## What is not yet established

* **One GPU generation.** Ampere only; Hopper codec cost is unmeasured, so the
  break-even is specific to this hardware pair.
* **No end-to-end run.** These are component measurements composed through the
  cost model, not a compress → transfer → decompress timing. The model's
  prediction has not yet been checked against a real round trip.
* **Synthetic tensors.** Gaussian activations with injected channel outliers, not
  captured Qwen3-Omni states. Ratio depends on distribution, so 2.55× is not yet
  a claim about the real model.
* **The serialiser is not production-shaped**, and its cost is excluded above.
