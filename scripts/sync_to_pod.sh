#!/usr/bin/env bash
# Push the source tree to a pod. No results, no venv, no tensors.
#
# tar-over-ssh rather than rsync: macOS ships openrsync, which does not accept
# several GNU rsync flags, and the transfer is small enough that incremental
# sync buys nothing. One less thing to debug while a pod is billing.
#
# The repo is not on GitHub yet, so this is the transport. Once it is public this
# becomes a git clone and the pod stops needing anything from the laptop.
set -euo pipefail
HOST="${1:?usage: sync_to_pod.sh <user@host> [ssh-port]}"
PORT="${2:-22}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST=/workspace/boundrelay

COPYFILE_DISABLE=1 tar czf - -C "$HERE" \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude 'results' --exclude 'artifacts' --exclude '.git' \
  --exclude '.ruff_cache' --exclude '*.egg-info' \
  . | ssh -p "$PORT" -o BatchMode=yes "$HOST" \
      "mkdir -p $DEST && tar xzf - --no-same-owner -C $DEST && cd $DEST && pip install -q -e . 2>&1 | tail -3; echo SYNCED"
