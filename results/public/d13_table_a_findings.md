# D13 — Table A, native transport benchmark on captured tensors

24 KV tensors captured from Qwen3-1.7B (D8), flattened to `[tokens × heads, head_dim]`,
bf16 throughout. Reference codec. Raw: `results/public/d13_table_a/table_a_kv.json`.

> **Corrected in Phase A.** Every cross-tensor mean below was computed at a fixed
> *absolute* ε. Across this corpus the per-tensor std spans 115× and the range
> spans 255×, so ε = 0.15 is 0.009 std on one tensor and 1.03 std on another —
> the averages are therefore not comparisons and are withdrawn. Re-derived under
> `eps_i = c · std_i` in `table_a_kv_rel.json`: blockwise 1.00/1.35/2.18×,
> per-channel 1.35/2.01/2.87×, per-token 1.02/1.44/2.25× at c = 0.01/0.03/0.10.
> Normalisation *widens* per-channel's lead (1.49× over blockwise at c = 0.03,
> against 1.25× under the old coordinates). The matched-error int8 comparison is
> unaffected — it is matched per tensor, not aggregated at a shared ε.

## At matched error

Each int8 baseline is measured first, its *achieved* maximum absolute error read
off, and the error-bounded codec then asked for that same error. This is the
comparison a reader actually wants: given the same reconstruction quality, which
representation moves fewer bytes?

| representation | ratio | max error | bits/element |
|---|---|---|---|
| bf16 passthrough | 1.00× | 0 | 16.00 |
| int8, per-tensor scale | 2.00× | 0.5013 | 8.00 |
| int8, per-channel scale | 1.99× | 0.5002 | 8.05 |
| **error-bounded, matched to int8's error** | **2.55×** | 0.5013 | **6.27** |

**1.28× fewer bytes at identical error.** Mean over the 24 tensors, each matched
to its own int8 error rather than to a single shared ε.

At a fixed ε = 0.5, where int8 also lands, per-channel allocation reaches 3.12×
against int8's 2.00× — **1.56×**.

## Per-channel int8 buys almost nothing here, and that is the point

`int8_per_channel` scores 1.99× against per-tensor's 2.00× — marginally *worse*,
once its per-channel fp32 scales are counted — and its error barely moves
(0.5002 vs 0.5013). This is the row that answers the standard objection, and it
answers it in the opposite direction to the one usually assumed.

The mechanism: int8 fixes the width at 8 bits and varies the *scale*. Per-channel
scaling shrinks the error of narrow channels, but the maximum error is set by the
widest channel, and that channel still gets `range / 254` regardless of how the
scales are grouped. So per-channel scaling improves the average and leaves the
bound where it was.

The error-bounded codec varies the *width* instead: narrow groups take few bits,
wide groups take more or fall to the bit-exact path. Bits go where the error
budget actually binds. That is why targeting the error beats targeting the width,
and it is a statement about the objective, not about tuning.

## Ratio by allocation, all 24 tensors

| allocation | ε = 0.05 | ε = 0.15 | ε = 0.5 |
|---|---|---|---|
| blockwise | 1.44× | 1.85× | 2.42× |
| **per-channel** | **1.82×** | **2.31×** | **3.12×** |
| per-token | 1.46× | 1.86× | 2.43× |

Per-channel leads at every ε, and blockwise and per-token are indistinguishable
from each other — consistent with D8, where the per-channel advantage was
concentrated in the key tensors and reversed for values at depth. Averaged over
keys and values together, the key-side advantage dominates.

## Limits

* One model (Qwen3-1.7B), one tensor family (KV cache), 24 tensors from 4 prompts.
* Reference codec only. Per-channel allocation still has no GPU throughput number:
  the GPU path is blockwise-only by scope, and the channel-major permute it would
  need is an unmeasured cost that belongs in the cost model, not hidden in a kernel.
* Ratios are reported against bf16 bytes (2 B/element) throughout, including for
  the int8 rows, whose fp32 scales are counted in their payload.
