# B0 — buffer reuse is safe, and two things about the codec that were not known

2×A40 (sm_86), driver 570.195.03, torch 2.8.0+cu128, **cuSZp `f581dcf3`**
(libcuSZp.so sha256 `d0c0f4a4…`). Four caches: d00 and d01 at 1024 and 2048 tokens,
28 layers × K/V each. Raw: `b0/b0_summary.json`, `b0/b0_tensor_checks.jsonl`.
Runtime 32 s. **PASSED.**

This is the gate. Everything else this round depends on a reused buffer delivering
the right bytes, so it ran first and B1/B2 were not allowed to start until it
passed.

## Reuse

| | |
|---|---|
| fresh baselines (4 caches × 56 tensors × `c ∈ {0.01, 0.10}`) | **448**, 0 fp32 failures |
| reuse round trips through **one** pool | **2240** |
| bitwise mismatches against the fresh reference | **0** |
| non-finite values | **0** |
| pool buffer addresses unchanged throughout | yes |

Twenty full-cache passes per `c`, each of the four caches appearing exactly five
times, in an order that forces large→small→large and puts two same-length
different-content caches next to each other. Every third pass the buffers were
deliberately filled with a non-zero byte and NaN first, so the production zeroing
had to survive something rather than merely a fresh allocation.

The criterion is **bit-identical output**, not a matching error summary. A
reconstruction that matches bit for bit has the same error by construction; one
that matches on max-abs error can still be a different tensor.

So: with the conservative discipline — zero the scratch before encoding, zero the
receive slot before receiving, zero the decode destination before decoding — a slot
can carry a different tensor, a different length and different content indefinitely
without contaminating the next one. That discipline is not free, and B2 pays for it
inside its timed region.

## The validator rejects what it should

Six injected faults, all detected:

| fault | caught by |
|---|---|
| NaN in the reconstruction | finiteness, checked **before** the bound |
| +Inf, −Inf | same |
| a value pushed past ε | the fp32 bound |
| **a single flipped bit in the delivered bf16** | the bitwise comparison — no error bound can see this |
| half the payload bytes removed | the fp32 bound, at **245 × ε** |

The bit flip is the one that matters. It changes a value by one ulp, no error bound
will ever notice, and the only thing that catches it is comparing the bytes.

## cuSZp ignores the `cmpSize` argument entirely

The truncation test originally declared half the compressed size while leaving the
bytes in place, and reported MISSED. The validator was right; the test was wrong,
and finding out which produced the better result:

| declared `cmpSize` | true size | reconstruction |
|---|---|---|
| 628,536 (true) | 628,536 | correct |
| 314,268 (half) | 628,536 | **bit-identical to the honest decode** |
| **64** | 628,536 | **bit-identical to the honest decode** |
| 628,536, but bytes past the halfway point zeroed | — | wrong, **245 × ε** |

`cuSZp_decompress_1D_fixed_f32` never reads the parameter. It works from `nbEle`
and consumes the buffer until it is done.

This project previously *inferred* a zeroed-buffer contract from the observation
that the decoder reads past `cmpSize`. The mechanism is now specific, and it has a
consequence for anyone building a transport on this library: **a receiver that
sizes a buffer, bounds a copy, or validates a payload from the declared compressed
length is trusting a number the library never consults.** It is also why bytes left
past a short payload are live input rather than harmless tail.

## The fp32 tolerance declared in the protocol has the wrong shape

The protocol fixed `E32 ≤ ε + τ` with `τ = 1e-6` **absolute**, before any data
existed, and said plainly that τ is not a knob to widen after a failure. B0 passed
it — 0 of 448. B1 then failed it at ε = 1.465, and B0's own distribution explains
why:

| | `c = 0.01` | `c = 0.10` |
|---|---|---|
| max `E32/ε` | 0.999999526 | **1.000000578** |
| max absolute excess `E32 − ε` | −2.33e-09 | **+1.79e-07** |
| ε range | 0.00134 – 0.2215 | 0.0134 – 2.215 |
| combinations over `ε + 1e-6` | 0 | 0 |

**The codec does exceed its declared bound**, by a few float32 ulps *of ε* — which
is what a quantiser working from a float32 `1/ε` will do. The excess is
**relative**. B0's largest ε was 2.2, so its largest absolute excess was 1.79e-07
and everything passed; at a larger ε the same relative excess crosses 1e-6.

τ was not widened. What the evidence supports is a *relative* criterion of about
`E32 ≤ ε · (1 + 1e-6)` — three orders of magnitude tighter than the `ε × 1.001`
this round replaced, and the right shape. That is a change to propose with the
data, not one to make while collecting it, so this round reports the margin it
measured and does not certify the absolute form.

## What this does not establish

* Four caches from two documents, one model, one codec build, one GPU pair.
* Reuse is shown safe **under conservative zeroing**. Nothing here says which of
  the three zeroings could be dropped, and the `cmpSize` result argues against
  dropping the receive-slot one.
* The bf16 error is recorded but not bounded. `Ebf` regularly exceeds ε because the
  downcast adds up to half a bf16 ulp on top; that is a measured fact about the
  delivered value, not a contract, and B1 is where its consequences are asked about.
