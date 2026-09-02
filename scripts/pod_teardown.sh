#!/usr/bin/env bash
# Sync everything worth keeping to the network volume, then stop.
#
# Runs at the end of every session, and is the last thing a kill rule triggers.
# The volume is for checkpoints and cross-session reuse -- not an unlimited cache.
set -euo pipefail
VOL="${VOLUME_PATH:-/workspace/volume}"
RUN_ID="${1:?usage: pod_teardown.sh <run-id>}"

DEST="$VOL/runs/$RUN_ID"
mkdir -p "$DEST"
rsync -a --info=progress2 results/ "$DEST/results/" 2>/dev/null || true
rsync -a --info=progress2 logs/    "$DEST/logs/"    2>/dev/null || true

echo "synced -> $DEST"
echo "Now record actual cost in docs/budget_ledger.md, then terminate the pod."
