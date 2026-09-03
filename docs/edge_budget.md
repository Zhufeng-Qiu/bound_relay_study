# Edge budget

> **Superseded.** This document was written when the target was an inter-stage
> relay edge in SGLang-Omni. The corpus is KV cache, which is not a short-lived
> stage-edge payload, so the scenario is now disaggregated prefill–decode
> transfer, KV migration and cache offloading. Kept as the record of how the
> target edge was originally chosen; the payload arithmetic below no longer
> describes what is being measured.


Paper arithmetic done before renting anything. Confirm each row against a real
trace; the point is to know which edges are worth capturing, not to guess.

## Payload per edge (Qwen3-Omni-30B-A3B, from `config.json`)

`thinker_config.text_config.hidden_size = 2048` · audio `output_dim = 2048` ·
vision `out_hidden_size = 2048` · `talker_config.text_config.hidden_size = 1024`

| Edge | Payload | Order |
|---|---|---|
| Thinker → Talker | 2048 × 2 B = **4 KB / token** | 10⁰ KB |
| Audio encoder → Thinker | ~30 s audio ≈ **3 MB / request** | 10³ KB |
| Vision encoder → Thinker | ~10 s video @2 fps ≈ **20 MB / request** | 10⁴ KB |

> The per-frame token count for video must be confirmed from the processor
> config; it is the one estimate here that is not read off `config.json`.

## Why both edges get captured in the same session

The span between them is three to four orders of magnitude. Capturing only
Thinker → Talker leaves Figure 1 with a single point on its x-axis. Capturing
both costs nothing extra and is what turns the atlas into a curve.

## Known upstream measurement

Public SGLang-Omni profiling reports inter-stage IPC across all seven pipeline
edges at roughly **12 ms/request, ~0.4% of wall time**, with same-node multi-GPU
relay running as a zero-copy CUDA IPC data plane.

This is not a discouraging result to be worked around — it *is* the finding. It
means the canonical single-host path is a correct-bypass case, and the research
content sits in the break-even boundary: at what payload and what effective
bandwidth does error-bounded compression start to pay? Claiming a canonical
runtime speedup would contradict published measurement.
