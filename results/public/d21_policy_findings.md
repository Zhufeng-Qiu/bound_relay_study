# D21 — the decision rule, evaluated on measured costs

Candidates are the measured 4/6/8-bit options from the A40 cost table (D9–D15);
the bandwidth axis spans 0.5–450 GB/s and includes the two points measured on the
same hardware in D6. Raw: `results/public/d21_policy/policy_eval.json`.

## How this is scored, and why not as "regret vs oracle"

A controller scored against an oracle that shares its cost model has zero regret
by construction. Reporting that number would be theatre, so it is not the
headline. The one policy knob actually kept is the safety margin, and that is
what gets measured: what the margin costs in latency, and how many chunks it
turns from compress into bypass.

| margin | compress | bypass | bypass rate | flipped vs oracle | mean regret |
|---|---|---|---|---|---|
| 0 ms | 11 | 25 | 69.4% | 0 | 0 |
| 0.10 ms | 11 | 25 | 69.4% | 0 | 0 |
| 0.25 ms | 9 | 27 | 75.0% | 2 | **0.0093 ms** |

A 0.25 ms margin converts two of 36 decisions to a conservative bypass and gives
up 0.0093 ms of mean latency. Cheap insurance against prediction error, and now a
measured quantity rather than an assumption.

## The decision map

| payload | compresses up to | at 8.45 GB/s (measured) |
|---|---|---|
| 0.26 MB | never | bypass |
| 2.10 MB | 2 GB/s | bypass |
| 10.49 MB | 8.45 GB/s | **compress** |
| 20.97 MB | 12.5 GB/s | **compress** |

The 10.49 MB row flips exactly at the measured link speed: the real hardware sits
on the boundary rather than comfortably inside either regime, which is the
condition under which an online decision is worth making at all.

Where it compresses, the controller always picks **6-bit** — consistent with the
codec measurement, where 6 bits met ε = 0.15 with fewer bits than 8 and the extra
precision of 8-bit was pure cost.

## The headline is the colour balance

**The controller bypasses in 69% of the grid.** Compression is the exception. On
the small edge — Thinker → Talker at roughly 4 KB per token, three orders of
magnitude below the smallest payload here — it is never close: the codec's fixed
cost dwarfs any transfer saving at every bandwidth tested.

That is the result the project set out to be able to state honestly, and it is
the opposite of the claim a compression project is tempted to make. The value is
not that compression helps; it is knowing the narrow band where it does.

## Limits

* Costs are A40-measured; the boundary moves with codec speed, so it is specific
  to this hardware.
* Decisions are computed from the cost model, not observed in a running transport.
  No chunk has yet been compressed, sent, and decompressed end to end.
* One ε (0.15) and one allocation (blockwise, the GPU path's scope). Per-channel
  allocation compresses better (D13) but has no GPU cost number, so the controller
  cannot yet choose it.
