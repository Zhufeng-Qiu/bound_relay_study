# Environment lock

What every committed measurement was produced on. The earlier version of this file
described an SGLang-Omni/H100 plan that was never executed and carried eleven
`TODO`s; it is replaced by the record of what actually ran.

## Known gap

**The GPU results span four rented machines and two driver versions.** Driver
version was read from the terminal but is **not recorded in any committed
manifest**, so a given number cannot be traced to the driver that produced it.
Every future session must write `nvidia-smi --query-gpu=driver_version`, the
topology matrix and the pod id into its manifest, and the remaining timings that
enter a comparison should be re-measured together on one pinned host.

## Software

| item | value |
|---|---|
| Container image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| CUDA (container toolkit) | 12.8 |
| PyTorch | 2.8.0+cu128 |
| Triton | 3.4.0 |
| transformers | 5.16.1 |
| Python | 3.12 |
| cuSZp | built from `github.com/szcompressor/cuSZp` @ depth-1 clone, `CMAKE_CUDA_ARCHITECTURES=86`, `BUILD_SHARED_LIBS=ON`. **Commit not pinned — a gap.** Bound through ctypes by mangled symbol: its headers carry no `extern "C"`. |
| SZ3 | via `pysz`; version not pinned — a gap |
| zfp | via `zfpy` |

## Model

| role | model | revision | dtype |
|---|---|---|---|
| Corpus, transport, quality | `Qwen/Qwen3-1.7B` | `b9352fbb8ce704292730cf54b3b1dceb2a808738` | bf16 |

28 layers, 8 KV heads, head_dim 128. Full cache at 2048 tokens: 224 MiB.
Capture is one `model(input_ids, use_cache=True)` prefill.

## Hardware, per phase

| phase | GPUs | data centre | topology | driver | spend |
|---|---|---|---|---|---|
| D6 topology smoke | 2×A40 | CA-MTL-1 | `SYS`, cross-NUMA | 580.159.04 | $0.19 |
| D9 codec + cost table | 1×A40 | CA-MTL-1 | — | not recorded | $0.12 |
| D8/D14b KV capture, cuSZp | 1×A40 | CA-MTL-1 | — | not recorded | $0.29 |
| B0 corpus + B2 variance | 1×A40 | EU-SE-1 | — | not recorded | $0.26 |
| C/D/E transport, quality, pipeline | 2×A40 | EU-SE-1 | `SYS`, cross-NUMA | not recorded | $0.82 |
| *(terminated: GPU 0 hardware fault)* | 2×A40 | CA-MTL-1 | `PXB`, same-NUMA | 570.195.03 | $0.23, unused |

**Total $1.90.**

## Measured environment properties worth carrying forward

* **Direct peer copy is pathological on every A40 pair rented** — 0.05 GB/s against
  10.01 GB/s host-staged, while `can_device_access_peer` returns True. Observed on
  three machines across both `SYS` and `PXB` topologies.
* **CUDA context creation costs 305 ms**, 347× the warm cost of a compression call.
  Any measurement that does not amortise it measures context creation.
* Local work (property tests, model fitting, plotting, SZ3/zfp ratios) ran on
  Apple M5. Ratios are deterministic and unaffected; **the SZ3/zfp *timings* in
  `d14_table_b_findings.md` come from that host and must not be compared against
  GPU timings** — they are a floor, single-threaded, and were used in one
  cross-environment comparison that is flagged there.
