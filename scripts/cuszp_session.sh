#!/usr/bin/env bash
# D14b — build cuSZp on the pod and benchmark it against the same tensors.
#
# cuSZp ships no Python binding, so a small C++ driver is written against
# whatever API the cloned headers actually expose; the header is inspected first
# rather than guessed, because a wrong signature costs a build cycle at $0.44/h.
set -euo pipefail
cd /workspace
if [ ! -d cuSZp ]; then
  git clone --depth 1 https://github.com/szcompressor/cuSZp.git
fi
cd cuSZp
echo "=== repo layout ==="; ls
echo "=== public headers ==="; find . -name '*.h' -o -name '*.hpp' | grep -iv test | head -20
echo "=== entry points ==="
grep -rhoE '^[a-zA-Z_][a-zA-Z0-9_ *]*cuSZp[a-zA-Z0-9_]*\([^)]*\)' --include='*.h' . | sort -u | head -20
