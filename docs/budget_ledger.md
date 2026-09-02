# Budget ledger

Smooth path **$71** · plan **$105** · **hard stop $150**.

No pod starts without a row here stating its purpose and expected outputs. A
session with no expected outputs must not be started.

| # | Date | Pod ID | GPUs | Purpose | Expected output | Planned | Actual | Cum. |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-02 | `67r3culelsiiqr` | 2×A40 | Topology smoke (D6) | P2P verdict, measured link BW | $0.50 | ~$0.20 | ~$0.20 |
| 2 | | | CPU/1×A40 1.5h | Volume pre-fill | int4 ckpt on volume | $1 | | |
| 3 | | | 1–2×A40 3h | INT4 capture | 2-edge dev corpus | $2–5 | | |
| 4 | 2026-09-02 | `fvajialo63pzmk` | 1×A40 ~40min | KV capture (D8) + D7 attempt | real KV corpus, Fig 2; D7 aborted | $1 | ~$0.29 | **~$0.62** |
| 5 | 2026-09-02 | `y3ei7x3rrnhyy9` | 1×A40 16min | Codec verify + cost table (D9–D15) | 16/16 equivalence, cost_table.json | $13 | **$0.12** | **$0.33** |
| 6 | | | 2×A40 10h | Atlas validation | 2–3 measured link points | $9 | | |
| 7 | | | 1×A40 6h | Quality sweep (D20) | eps → perplexity | $3 | not started | |
| 8 | | | 1×A40 8h | Integration rehearsal | hook passes on TTS model | $4 | | |
| 9 | | | 2×H100 4h | Final session | Hopper codec bench, bf16 corpus, E2E | $27 | | |
| — | | | — | ~~Network volume 150 GB × 1 mo~~ **dropped** — checkpoint pull measured at ~1 GB/s, so pre-filling saves nothing | — | ~~$11~~ **$0** | | |

## Stop rules

- **90 min** with no smoke output on a bring-up → terminate, fix offline, use the
  recovery window. Do not debug dependencies inside a pod.
- **Gate E**: integration rehearsal fails → do not book the H100. Saves $27; the
  project is already deliverable.
- **$150 cumulative** → all paid work stops. Finish the note with what exists.
