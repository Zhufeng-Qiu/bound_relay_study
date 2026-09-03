# Superseded results

Nothing here has been deleted or edited to agree with what replaced it. Each
document still carries the retraction marker it was given when its claim fell, and
each is reachable from `docs/corrections.md`. They are out of the front directory
for one reason: a reader landing on `results/public/` should see one set of claims
this project currently defends, not a stratigraphy of what it believed on the way.

Three different things put a document in here, and they are not the same thing.

## Replaced by a later measurement of the same question

| document | replaced by | what changed |
|---|---|---|
| `a3_sz3_config_findings.md` | `../a3_sz3_full_findings.md` | Scanned three `cmprAlgo` values on the superseded 81-token decode corpus. The replacement scans every configuration `pysz` can reach, on the Gate B0 prefill capture, and the effect it reports is a third the size |
| `d14_table_b_findings.md` | `../a3_sz3_full_findings.md` | Fed SZ3 a permuted view rather than the stored order, and timed CPU codecs on a laptop against GPU codecs on an A40 |
| `d14b_cuszp_findings.md` | `../b0_corpus_findings.md` | Its "3–6× fixed-mode variance" was cold start, not the mode |
| `d20_quality_findings.md` | `../d_quality_findings.md` | Qwen3-0.6B on a laptop at a fixed absolute ε, on a corpus since replaced |
| `e_pipeline_findings.md` | `../session_close_findings.md` (Gate 2) | Measured same-GPU encode against encode, which is not the question a cross-GPU pipeline asks |
| `d13_table_a_findings.md` | `../b0_corpus_findings.md` | Cross-tensor means at a fixed absolute ε, over 24 tensors from four prompts |

## Withdrawn outright

| document | why |
|---|---|
| `asymmetric_kv_findings.md` | Every number in it came from reconstructions cuSZp had not fully written. Re-measured with zeroed buffers, the K/V interaction contrasts cross zero. See `../remeasure_findings.md` |

## Early-phase work, correct but subsumed

`d6_topology_findings.md`, `d7_aborted.md`, `d8_kv_findings.md`,
`d9_codec_findings.md`, `d21_policy_findings.md`.

These are not wrong in the way the two groups above are wrong. They are bring-up
and pilot records: a topology smoke, a corpus capture on four prompts, a codec
verification, a decision rule evaluated on the cost table of the day, and one phase
stopped under its own kill rule. Every claim they carry that later work touched is
marked in place. They stay because a reader asking "what did the pilot look like
before the frozen corpus existed" should be able to answer it.
