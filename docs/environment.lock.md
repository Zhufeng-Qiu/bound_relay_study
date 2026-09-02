# Environment lock

Fill before the first pod (Gate A). Pin SHAs and digests, never tags or `main`.

## Software

| Item | Value |
|---|---|
| SGLang-Omni commit SHA | `TODO` |
| Docker image (linux/amd64) tag | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Docker image digest | `TODO` |
| CUDA | 12.8 (container toolkit; host driver reports 13.0) |
| PyTorch | 2.8.0+cu128 |
| Triton | 3.4.0 |
| Python | 3.12 |

> The Mac is arm64 and cannot build or test the amd64 image locally without slow
> emulation. Use a stock RunPod PyTorch image plus a pinned `setup.sh`, or build
> on an amd64 CI runner. Do not hand-build the image on the laptop.

## Models

| Role | Model | Revision | Weight dtype | Notes |
|---|---|---|---|---|
| Dev corpus | `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` | `TODO` | int4 AutoRound (W4A16) | ≈26.3 GB. Activations stay bf16, so edge shapes/dtypes match — but the value *distribution* is not guaranteed to match bf16, and both ratio and downstream sensitivity depend on distribution. Dev use only. |
| Reference corpus | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `TODO` | bf16 | Headline numbers and quality conclusions come from here. |
| KV corpus | `TODO` (1–3B open model) | `TODO` | bf16 | Second tensor family. |
| Integration rehearsal | `TODO` (1–2B TTS) | `TODO` | bf16 | Cheap stand-in for hook wiring. |

## Hardware

| Session | GPU | P2P enabled | `nvidia-smi topo -m` |
|---|---|---|---|
| Topology smoke | 2×A40, CA-MTL-1, sm_86, driver 580.159.04 | driver says OK, **but path is pathological** | `SYS` (cross-socket PCIe, different NUMA nodes, no NVLink) |
| Capture | `TODO` | — | — |
| Final | 2×H100 SXM | `TODO` | `TODO` |

> GPU SKU is not evidence of topology. Record `p2pBandwidthLatencyTest`,
> `nvidia-smi topo -m`, the relay backend in use, and the measured effective
> bandwidth — those are.
