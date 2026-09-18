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
| 10 | 2026-09-03 | `09xqdxdm9y28dd` | 2×RTX A6000 | **Gate 2 re-run** — the 30% overlap figure was measured with uninitialised buffers and is flagged unquotable. A40 is out of stock at 2 GPUs, so both the buggy and the fixed variant run here back to back and the hardware change cancels. | `results/public/gate2/gate2_rerun.json`: encode/decode/1-thread/2-thread on the same host under both buffer regimes, with a reconstruction check the original never did | $1.00 | **$0.76** (43 min × $1.06/h) | |
| 11 | 2026-09-03 | — | 0 | **Peer-copy audit** — found while debugging Gate 2's correctness gate on the same pod, no extra pod time | `results/public/gate2/peer_copy_audit.json`: every direct peer path returns `cudaSuccess` and transfers zeros; host-staged is exact | $0 | **$0** | |
| 12 | 2026-09-03 | `bl467xxd83g4eq` | 2×A40 | **Multi-stage pipeline** — Gate 2 measured two stages of seven and the conclusion drawn from it did not follow. A40 chosen over the cheaper A6000 because the depth-1 control has to reproduce a serial baseline measured on A40. Plus the `/workspace` MooseFS transport, to put a point on the far side of the 10.96 GB/s pipelined break-even | `pipeline_multistage.json`: contention probe, depth sweep 1/2/4/8, compressed and raw through one harness, both transports | $2.10 | | |
| 13 | 2026-09-03 | `o79bqs1onvr988` | 1×A40 | **fsync-acknowledged offload write** — the pipeline run put the cache on MooseFS without ever making the write durable, and compression lost there while moving a third of the bytes. Both arms, one mount, one process, with and without `fsync`, so the durability claim does not rest on comparing two data centres | `fsync_offload.json`: raw vs compressed, per-tensor and single write, fsync on and off, each read back and verified before timing; plus the payload-size bandwidth control | $0.40 | **$0.23** (31 min × $0.44/h) | |
| 14 | 2026-09-18 | `pxc2rbsu0jvm08` | 2×A40 @ $0.98/h | **B0/B1/B2 under the 2026-09-17 protocol** — buffer-reuse correctness gate, held-out quality on 32 frozen articles, and paired three-path performance. One pod for all three so the cuSZp build, model download and corpus capture are paid once | `b0_summary.json` + `b0_tensor_checks.jsonl`; `quality_documents.jsonl` + `quality_summary.json`; `path_trials.jsonl` + `path_summary.json` | $10 | **$1.57** (1.6 h × $0.98/h) | |
| 15 | 2026-09-18 | `0rcxc7fznp6s3z` (dead), `xgiap63o63f6el` | 2×A40 @ $0.98/h | **depth=1 slot lifecycle** — protocol §4.3 asks for the dual-GPU stress at depth 1 *and* 8, and the B2 matrix only ran 8. Depth 1 is the harsher case: one slot is reused immediately by the next tensor, so a slot freed before the asynchronous read of it completes has no grace period. The first pod reported RUNNING for 9 minutes without ever producing a runtime and was terminated under the bring-up stop rule | depth 1 vs 8 delivery verification on the real two-device path | $0.40 | | |
| 16 | 2026-09-18 | `x9ij9ieddzv0q8` | 2×A40 @ $0.98/h | **Acceptance gaps found by protocol review** — reuse compared only the delivered bf16 and never the fp32 reconstruction; the two-device check judged a whole cache against its largest ε instead of each tensor against a fresh baseline; the alignment investigation existed as prose with no record behind it. No timing: B2's 720 measurements stand, since nothing in the synchronisation, zeroing or output-copy paths changed | `b0v2/`, `twodevice_pertensor_depth{1,8}.jsonl`, `alignment/` | $0.60 | | |
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
