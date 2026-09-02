# Budget ledger

Smooth path **$71** · plan **$105** · **hard stop $150**.

No pod starts without a row here stating its purpose and expected outputs. A
session with no expected outputs must not be started.

| # | Date | Pod ID | GPUs | Purpose | Expected output | Planned | Actual | Cum. |
|---|---|---|---|---|---|---|---|---|
| 1 | | | 2×A40 0.5h | Topology smoke | P2P verdict, `topo -m` | $0.50 | | |
| 2 | | | CPU/1×A40 1.5h | Volume pre-fill | int4 ckpt on volume | $1 | | |
| 3 | | | 1–2×A40 3h | INT4 capture | 2-edge dev corpus | $2–5 | | |
| 4 | | | 1×A40 2h | KV capture | KV corpus | $1 | | |
| 5 | | | 1×A40 30h | Codec + tables | Triton kernel, Table A/B | $13 | | |
| 6 | | | 2×A40 10h | Atlas validation | 2–3 measured link points | $9 | | |
| 7 | | | 1×A40 6h | Quality sweep | eps → perplexity | $3 | | |
| 8 | | | 1×A40 8h | Integration rehearsal | hook passes on TTS model | $4 | | |
| 9 | | | 2×H100 4h | Final session | Hopper codec bench, bf16 corpus, E2E | $27 | | |
| — | | | — | Network volume 150 GB × 1 mo | — | $11 | | |

## Stop rules

- **90 min** with no smoke output on a bring-up → terminate, fix offline, use the
  recovery window. Do not debug dependencies inside a pod.
- **Gate E**: integration rehearsal fails → do not book the H100. Saves $27; the
  project is already deliverable.
- **$150 cumulative** → all paid work stops. Finish the note with what exists.
