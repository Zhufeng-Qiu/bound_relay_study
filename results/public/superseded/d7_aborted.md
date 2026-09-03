# D7 — aborted under the kill rule, with two findings worth keeping

Attempted on the same 1×A40 as D8. Stopped rather than debugged in-pod, which is
what the rule is for.

## What happened

`pip install sglang-omni==0.1.4` pulls a very large dependency tree — its own
torch, triton, NCCL, CUTLASS, flash-attn-4, tilelang, a CUDA toolkit — and filled
the 40 GB container disk before finishing. Two things had already gone wrong by
then:

* **Triton was replaced**, 3.4.0 → 3.7.1, before the install ran out of space.
* The 29 GB INT4 checkpoint landed in `/root/.cache/huggingface` on the container
  disk rather than the 120 GB `/workspace` volume: the pod's `HF_HOME` env var is
  set in the container environment but an `ssh` session does not inherit it.

## Finding 1 — the Omni lane and the codec lane cannot share a pod

This is the substantive one. Installing SGLang-Omni **replaces the torch/Triton
the codec was measured against**. Every number in `d9_codec_findings.md` was
taken on Triton 3.4.0; a pod that has installed SGLang-Omni is no longer the
machine that produced them.

So D7 needs its own pod and its own environment lock, and the two sets of results
must be reported against separate software stacks rather than being quietly
pooled. Retrying D7 in place would have produced a cost table and a capture that
looked comparable and were not.

Requirements for the retry: **≥150 GB container disk**, `HF_HOME` exported inside
the shell (not only set on the pod), and no expectation of reusing the codec
environment.

## Finding 2 — the network volume is unnecessary

The 29 GB checkpoint downloaded in about 23 seconds, roughly 1 GB/s. The Runbook
budgeted a 150 GB network volume at ~$11/month specifically so an expensive
session would not spend its first half hour downloading weights. At this speed
that concern does not exist: a checkpoint pull costs seconds, not a fraction of a
session.

**Drop the volume line from the budget.** It also removes the data-centre pinning
problem found earlier — A40's data centres offer no network volumes at all, which
had looked like a real constraint on how the lanes could be arranged.

## Status

D7 remains open. It is the 15% stretch item, and nothing else depends on it: the
codec, the cost table, the break-even boundary and the allocation study all rest
on measurements that are already in hand. What D7 would add is Qwen3-Omni's own
hidden states, which would turn the Omni-specific ratio claims from "expected" to
"measured".
