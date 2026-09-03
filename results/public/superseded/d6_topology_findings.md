# D6 — topology smoke, 2×A40 (CA-MTL-1)

Pod `67r3culelsiiqr`, 2026-09-02. Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`,
torch 2.8.0+cu128, triton 3.4.0, driver 580.159.04, A40 sm_86, 46 GB each.

## Topology

`nvidia-smi topo -m` reports **SYS** between GPU0 and GPU1: the path traverses PCIe *and*
the inter-socket link. The GPUs sit on different NUMA nodes (GPU0 → node 0, cores
0-23/48-71; GPU1 → node 1, cores 24-47/72-95). No NVLink on this host.

`nvidia-smi topo -p2p r` and `-p2p w` both report **OK**, and
`torch.cuda.can_device_access_peer(0,1)` returns **True**.

## Measured

| payload | direct D2D | host-staged | host-staged BW | one-way PCIe |
|---|---|---|---|---|
| 4 MB | 95.3 ms | 0.51 ms | 8.16 GB/s | ~23 GB/s |
| 64 MB | 98.5 ms | 7.89 ms | 8.50 GB/s | ~23 GB/s |
| 256 MB | 98.5 ms | 31.75 ms | 8.45 GB/s | ~23 GB/s |

Kernel-launch floor measured at 0.0074 ms, so the D2D cost is not launch overhead.
Controlled for: dual-device synchronisation, blocking vs non-blocking copy, 5-iteration
warmup, three payload sizes. Non-blocking made no difference (98.54 vs 98.55 ms at 256 MB).

## Finding

**The direct peer path costs a constant ~98 ms independent of payload size**, so it is not
a bandwidth at all — it is a fixed per-copy penalty. For a 4 MB payload, staging through
pinned host memory is **≈190× faster** than the "direct" GPU-to-GPU copy.

The driver advertises P2P as available and PyTorch agrees, yet the path that advertisement
describes is the slowest one on the machine. This is a concrete instance of the
project's own methodological rule: **transport cost cannot be inferred from a topology API
or a GPU SKU; it has to be measured.** Recorded here rather than worked around.

Likely cause is a virtualised cross-socket host where peer DMA falls back to a
synchronising staged path. Not investigated further — it is a host pathology, not the
research object, and diagnosing it further would spend budget on someone else's bug.

## Consequences for the study

1. **No fast path exists on this hardware.** The fastest usable inter-GPU route is
   host-staged at ~8.4 GB/s. An NVLink arm can only be modelled here, or measured later
   on an H100 SXM session if one becomes available.
2. **~8.4 GB/s is a genuine, non-artificial slow-path regime.** The preliminary cost model
   put break-even near 17 GB/s for a 20 MB payload at ~0.7 ms codec cost, so this real
   on-node path sits *below* break-even. The slow-path regime does not have to be
   manufactured with a bandwidth cap — the hardware supplies one.
3. Atlas measured points available from this session: **8.45 GB/s** (host-staged) and
   **23 GB/s** (one-way PCIe ceiling). The pathological peer path is reported but excluded
   from the bandwidth axis, since it is a fixed cost rather than a rate.

Raw: `results/public/superseded/d6_topology/d6_topology_v2.json`
