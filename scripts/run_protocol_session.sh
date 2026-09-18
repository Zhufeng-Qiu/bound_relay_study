#!/usr/bin/env bash
# Capture the four caches, then B0, then B1 and B2 -- in that order, because B0
# gates the other two. If the correctness gate fails, this stops: a quality number
# or a latency number taken through buffers that do not deliver the right bytes is
# not a slightly worse result, it is a meaningless one.
set -uo pipefail
cd /workspace
BR=/workspace/boundrelay
CORP=/workspace/corpus_protocol
OUT=/workspace/out
mkdir -p "$OUT"

CACHES="$CORP/fullcache_d00_L1024 $CORP/fullcache_d00_L2048 \
$CORP/fullcache_d01_L1024 $CORP/fullcache_d01_L2048"

step () { echo; echo "=============== $* ==============="; date -u +"%Y-%m-%dT%H:%M:%SZ"; }

step "environment"
python "$BR/scripts/bringup_protocol_session.sh" 2>/dev/null || \
  bash "$BR/scripts/bringup_protocol_session.sh" || exit 1

if [ ! -d "$CORP/fullcache_d01_L2048" ]; then
  step "capture: 2 documents x {1024, 2048} tokens, 28 layers each"
  python "$BR/scripts/capture_corpus.py" --docs 2 --full-cache-docs 2 \
      --lengths 1024 2048 --full-cache-lengths 1024 2048 --out "$CORP" \
      2>&1 | tail -20 || exit 1
fi
for d in $CACHES; do
  printf "%s  %s tensors\n" "$(basename "$d")" "$(ls "$d"/l*.pt 2>/dev/null | wc -l)"
done

step "B0 -- buffer reuse and error acceptance (gates everything below)"
python "$BR/scripts/validate_reuse.py" --caches $CACHES --passes 20 \
    --out "$OUT/b0" 2>&1 | tail -30
B0=${PIPESTATUS[0]}
if [ "$B0" -ne 0 ]; then
  echo "B0 FAILED -- stopping. B1 and B2 would not mean anything."
  exit 1
fi

step "B1 -- held-out quality, 32 frozen articles x 4 arms"
python "$BR/scripts/quality_holdout.py" --out "$OUT/b1" 2>&1 | tail -45 || \
  echo "B1 FAILED -- continuing to B2, which does not depend on it"

step "B2 -- paired three-path performance"
python "$BR/scripts/benchmark_paths.py" --caches $CACHES \
    --mount /workspace/b2mount --pairs 30 --segments 3 --warmup 10 \
    --out "$OUT/b2" 2>&1 | tail -30 || echo "B2 FAILED"

step "done"
ls -la "$OUT"/b0 "$OUT"/b1 "$OUT"/b2 2>/dev/null
echo SESSION_COMPLETE
