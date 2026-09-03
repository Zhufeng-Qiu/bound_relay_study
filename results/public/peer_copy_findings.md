# Peer-to-peer device copies that report success and move nothing

2×RTX A6000 (sm_86), RunPod `EU-SE-1`, torch 2.8.0+cu128, driver CUDA 12.8.
Raw: `results/public/gate2/peer_copy_audit.json`. Script: `scripts/peer_copy_audit.py`.
**$0** — found while debugging a correctness gate, on a pod already running.

This began as a bug hunt and is the sharpest measurement in the project.

## What happens

`torch.cuda.can_device_access_peer(0, 1)` returns `True`, in both directions.
`cudaDeviceEnablePeerAccess` returns `cudaSuccess`. `cudaMemcpyPeer` returns
`cudaSuccess`. And the destination buffer is left exactly as it was found.

Every path was fully synchronised on both devices before the copy was issued, so
none of this is a missing `synchronize`:

| path | elements exactly transferred | destination left zero |
|---|---|---|
| `dst.copy_(src)`, current device 0 | **0.000** | 1.000 |
| `dst.copy_(src)`, current device 1 | 0.016 | 0.984 |
| `cudaMemcpyPeer`, returned `cudaSuccess` | **0.000** | 1.000 |
| `cudaMemcpy(cudaMemcpyDefault)`, returned `cudaSuccess` | 0.016 | 0.984 |
| device 1 → device 0, the reverse direction | **0.000** | 1.000 |
| **routed through host memory** | **1.000** | 0.000 |

Not a size threshold: 4 B, 16 KB, 1 MB, 4 MB and 32 MB all transfer nothing. Not
intermittent: ten consecutive attempts, exact fraction 0.000 every time. The
topology is `PXB` — both cards behind one PCIe switch, no NVLink.

The source pattern is `(arange % 977) + 1`, which contains no zeros, so "the
destination is zero" and "the destination is correct" cannot be confused.

## Why it matters more than a broken host

Nothing raises. Nothing warns. The CUDA runtime returns success, PyTorch returns a
tensor of the right shape, dtype and device, and every element in it is zero.

A disaggregated prefill–decode transfer is, in its most natural form, exactly this
copy: compress the KV cache on the prefill GPU, move the bytes to the decode GPU,
decompress. `can_device_access_peer` is the API that exists to tell a system
whether it may take that path, and here it says yes and the path is silently
lossy. The receiver decompresses zeros into a plausible tensor and the model keeps
generating.

**This is the second rented multi-GPU host on which the peer path has misbehaved.**
It first appeared on 2×A40 in `CA-MTL-1` as an `inf` in a correctness gate, which
was worked around by moving the comparison to the host — recorded at the time as a
harness quirk. It is not a harness quirk. Two hosts, two GPU models, two data
centres, two of two.

## What this changes about the rest of this project

The host-staged transport measured in Phase C and in the closing session had
looked like a limitation of the harness: bytes go GPU → host → GPU when a peer
copy should have been available. On this infrastructure the host staging is not a
limitation, it is the only path that returns the data that was sent. Every
transport number in this project is therefore measured on the only correct route,
and the peer-copy route it was implicitly conceding throughput to does not work at
all.

It also explains a defect this project spent real time on. The closing session's
correctness gate reported `inf`, and the first hypothesis was the codec. The codec
was fine; the *reference* had been fetched across the peer path.

## What is not established

* **Cause.** Whether this is the hypervisor's IOMMU configuration, a PCIe ACS
  setting that forces peer traffic through a path that silently drops it, or
  something in the driver, is not determined here. Diagnosing it needs host-level
  access a rented pod does not have.
* **Prevalence.** Two hosts is two hosts. This says the failure recurs across
  machines and GPU models on one provider; it does not say what fraction of
  multi-GPU instances are affected, and it says nothing about other providers or
  about bare metal.
* **Bandwidth.** No throughput number is reported for the peer path, because it
  did not transfer anything to time.

The actionable form of this, for anyone building GPU-to-GPU state movement: do not
trust `can_device_access_peer`. Send a known pattern and read it back before
committing a transport to the peer route.
