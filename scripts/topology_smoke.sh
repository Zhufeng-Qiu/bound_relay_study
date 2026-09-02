#!/usr/bin/env bash
# D6, ~$0.50, 2xA40, 30 minutes.
#
# Answers one question before it can cost anything: is P2P actually enabled on
# this rented host? Cloud PCIe instances often ship with it off (ACS left on in
# the VM). If it is off, the atlas validation set becomes {host-staged, capped x2}
# instead of {P2P, host-staged, capped} -- a design change worth knowing on day 6
# rather than on day 18.
set -euo pipefail
OUT="${1:-results/private_raw/topology_smoke}"
mkdir -p "$OUT"

nvidia-smi                     > "$OUT/nvidia-smi.txt"
nvidia-smi topo -m             > "$OUT/topo.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpus.csv"

# Ships with the CUDA samples; build it if the image lacks the binary.
if command -v p2pBandwidthLatencyTest >/dev/null; then
  p2pBandwidthLatencyTest      > "$OUT/p2p.txt"
else
  echo "p2pBandwidthLatencyTest not on PATH -- build from cuda-samples" | tee "$OUT/p2p.txt"
fi

echo "--- P2P verdict ---"
grep -iE "peer|p2p" "$OUT/topo.txt" || true
echo "Record the verdict in docs/environment.lock.md before terminating the pod."
