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
| **fp32 reconstruction** mismatches against the fresh reference | **0** |
| **delivered bf16** mismatches against the fresh reference | **0** |
| non-finite values | **0** |
| pool buffer addresses unchanged throughout | yes |

Twenty full-cache passes per `c`, each of the four caches appearing exactly five
times, in an order that forces large→small→large and puts two same-length
different-content caches next to each other. Every third pass the buffers were
deliberately filled with a non-zero byte and NaN first, so the production zeroing
had to survive something rather than merely a fresh allocation.

The criterion is **exact equality against the fresh reference**, not a matching
error summary — two tensors can share a max-abs error and be different tensors.

Both representations are compared, and the numbers above are from the re-run that
does so. **The first pass compared only the delivered bf16**, on the reasoning that
a bit-identical reconstruction has the same error by construction. That reasoning
is sound for the bf16 value and does not carry to the fp32 one, because two fp32
reconstructions differing by less than half a bf16 ulp round to the same bf16 — so a
bf16 match can sit on top of an fp32 difference, and the fp32 reconstruction is what
the codec actually produces. See *The acceptance itself was too weak* below.

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

τ was not widened, and no replacement threshold is proposed here either.

An earlier draft suggested `E32 ≤ ε · (1 + 1e-6)`. That is arithmetically wrong as a
pass line: `1 + 1e-6 = 1.0000010000`, and B1 went on to measure
**1.0000014646 × ε** — the proposal fails on the very data meant to motivate it. A
threshold cannot be set by eyeballing one round's worst case and rounding it down.

What the evidence supports is narrower: **the excess observed here scales with ε
rather than sitting under a fixed absolute bound**, so the *shape* of an absolute
tolerance is wrong for it. Where a relative line should sit is not determined by
this data. Fixing one would need the margin characterised across ε, mode and tensor
shape, and it would have to be fixed before the round it governs — which is the
whole point.

So this round reports the margin it measured, certifies neither form, and leaves
the criterion open.

## The acceptance itself was too weak, twice, and is redone here

A review of this document found two places where the verdict was reached on less
evidence than the protocol asks for. Both are re-run; **both still pass**, but they
passed the first time for a weaker reason than the write-up implied.

**Reuse compared only the delivered bf16.** The justification given was that a
bit-identical reconstruction has the same error by construction. That is true of the
bf16 value and says nothing about the fp32 one: two fp32 reconstructions differing
by less than half a bf16 ulp round to the same bf16, so a bf16 match can hide an
fp32 difference — and the fp32 reconstruction is what the codec actually produces.
Re-run comparing **both**, by exact equality against a device-resident reference:

| | |
|---|---|
| reuse round trips | **2240** |
| **fp32** reconstruction mismatches | **0** |
| **bf16** output mismatches | **0** |
| non-finite | 0 |

The full check now costs no host transfer at all, which was the reason the fp32
audit was dropped in the first place.

**The two-device check judged a whole cache against its largest ε.** That is an
aggregate: a tensor whose own ε is small can be badly wrong and still sit under the
largest ε in the cache. Re-run per tensor, byte for byte, against a reconstruction
made on one device through buffers zeroed for it alone — at **both** depths, both
paths, both arms:

| | |
|---|---|
| per-tensor comparisons | **1792** (2 depths × 2 paths × 2 arms × 4 caches × 56) |
| raw delivered byte-exact to source | **896 / 896** |
| compressed delivered byte-exact to the fresh baseline | **896 / 896** |
| non-finite | 0 |

**The transport is exact.** Not "within a bound" — the two-device path delivers the
identical bytes a single GPU produces, at depth 1 and depth 8, on the serial path
and through the ring.

The residual `err/own ε` of up to **1.581** is therefore not transport error. It is
the bf16 downcast on top of the codec's fp32 error, and 852 of 896 delivered tensors
exceed their own ε for that reason alone. The clearest evidence it is not the
transport: for each cache the serial and pipeline paths report the *same* worst
value to nine digits.

## The slot lifecycle holds at depth 1 as well as depth 8

The protocol asks for the two-device stress at **both** depths, and the B2 matrix
fixes depth at 8. Depth 1 is the harsher case: one slot is reused immediately by the
next tensor, so a slot returned before the asynchronous read of it completes has no
grace period at all — it is where the race B2's harness originally had would show
first. Run separately on d00 L2048, 2 segments, both arms verified at the start and
end of every block:

| depth | pipeline raw | pipeline compressed | R | delivery checks |
|---|---|---|---|---|
| **1** | 33.39 ms | 90.90 ms | 3.224 | **24 / 24 pass** |
| **8** | 21.13 ms | 40.66 ms | 2.207 | **24 / 24 pass** |

Raw delivered byte-identical and compressed within ε plus measured rounding at both
depths. **The slot discipline is correct at depth 1**, which is the case that would
break first.

Depth 1 is also slower than the serial path it nominally resembles — 90.90 ms
against serial's 57.88 ms on the same cache — because a one-slot ring pays the
thread hand-off, the per-slot zeroing and the completion event while overlapping
nothing. That is a property of the harness at an unusable setting, not a result
about pipelining; it is reported because the run happened, not because it means
anything for the transport.

## What this does not establish

* Four caches from two documents, one model, one codec build, one GPU pair.
* Reuse is shown safe **under conservative zeroing**. Nothing here says which of
  the three zeroings could be dropped, and the `cmpSize` result argues against
  dropping the receive-slot one.
* The bf16 error is recorded but not bounded. `Ebf` regularly exceeds ε because the
  downcast adds up to half a bf16 ulp on top; that is a measured fact about the
  delivered value, not a contract, and B1 is where its consequences are asked about.
