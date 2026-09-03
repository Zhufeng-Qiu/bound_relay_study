# D8 — real KV-cache corpus, and the allocation hypothesis tested on it

Qwen3-1.7B (28 layers, 8 KV heads, head_dim 128), bf16, greedy decode, 4 prompts
× 64 new tokens. Layers 0 / 14 / 27 sampled, keys and values separately.
1×A40. Raw: `results/public/superseded/d8_kv/allocation_study_kv.json`.

Every ratio below is the mean over the four prompts; they agreed closely, so the
structure is a property of the tensors rather than of one input.

## Result: there is no single right allocation axis

Ratio at ε = 0.15:

| kind | layer | blockwise | per-channel | per-token |
|---|---|---|---|---|
| keys | 0 | **0.98×** | **2.44×** | **0.99×** |
| keys | 14 | 1.95× | **2.53×** | 1.97× |
| keys | 27 | 1.25× | **2.56×** | 1.49× |
| values | 0 | 3.74× | **3.89×** | 3.51× |
| values | 14 | 2.19× | **1.48×** | **2.23×** |
| values | 27 | 0.99× | 0.99× | 0.99× |

**Keys carry per-channel structure; values do not.** At layer 0 the effect is
categorical rather than marginal: blockwise and per-token allocation compress key
tensors *not at all* (0.98×, 0.99× — every group falls to the bit-exact path),
while per-channel reaches 2.44×, and 3.48× at ε = 0.5. A blockwise group spans
many channels, so one wide-range channel sets the range for the whole group and
no supported width meets the bound. Per-channel isolates it.

Values invert the ranking at depth: at layer 14 per-channel is the *worst* of the
three (1.48× against 2.23× for per-token). At layer 27 nothing compresses at
ε = 0.15 — all three sit at 0.99×.

## Why this matters more than the synthetic version

The earlier study used Gaussian activations with channel outliers injected by
hand, chosen because that is what the quantisation literature describes. It
confirmed the hypothesis, but only in the sense that a distribution built to have
per-channel structure turned out to have it.

Measured on real cache the finding is sharper and partly contradicts the
prediction: **the right axis is a property of the tensor and its depth, not of
the model or the codec.**

> **Corrected by D14.** This section originally went on to claim that the result
> explains why a general error-bounded compressor underperforms on these tensors.
> It does not. Benchmarked directly, SZ3 compresses them 5× better than this
> codec at ε = 0.5 — its prediction works fine on KV cache, and the assumption
> that LLM intermediate state is too unsmooth to predict was wrong. What survives
> is the narrower statement above, about which axis *this* codec should spend its
> budget along. See `d14_table_b_findings.md`.

It also gives the adaptive controller something real to adapt to: the choice is
not merely *whether* to compress, but *along which axis to spend the budget*.

## Limits

* One model, one size (1.7B), one decoding regime (greedy, short outputs).
* Three sampled layers out of 28, chosen as early/middle/late.
* Ratios come from the reference codec; per-channel allocation has no GPU
  throughput number, because the GPU path is blockwise-only by scope and the
  channel-major permute it would need is an unmeasured cost.
* Says nothing yet about Qwen3-Omni's own hidden states — that is D7.
