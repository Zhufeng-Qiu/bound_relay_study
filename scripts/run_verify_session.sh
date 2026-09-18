#!/usr/bin/env bash
# The three gaps the protocol review found, in one session.
#
#  1. reuse compared per round against BOTH the fp32 reconstruction and the bf16
#     output, not the bf16 alone
#  2. the two-device path judged per tensor against fresh baselines, at depth 1 and
#     depth 8, instead of against an aggregate epsilon over the whole cache
#  3. the scoring-alignment investigation saved as a record rather than prose
#
# No timing here. B2's 720 measurements stand: nothing in the timed path changed.
set -uo pipefail
cd /workspace
BR=/workspace/boundrelay
CORP=/workspace/corpus_protocol
OUT=/workspace/out
mkdir -p "$OUT"
CACHES="$CORP/fullcache_d00_L1024 $CORP/fullcache_d00_L2048 \
$CORP/fullcache_d01_L1024 $CORP/fullcache_d01_L2048"

step () { echo; echo "=============== $* ==============="; date -u +"%H:%M:%SZ"; }

step "environment"
bash "$BR/scripts/bringup_protocol_session.sh" 2>&1 | tail -3
pip install -q --break-system-packages transformers==5.16.1 datasets==5.0.1 2>&1 | tail -1
bash "$BR/scripts/bringup_protocol_session.sh" 2>&1 | tail -3

if [ ! -d "$CORP/fullcache_d01_L2048" ]; then
  step "capture"
  python "$BR/scripts/capture_corpus.py" --docs 2 --full-cache-docs 2 \
      --lengths 1024 2048 --full-cache-lengths 1024 2048 --out "$CORP" 2>&1 | tail -6
fi

step "1. reuse -- fp32 AND bf16 compared every round"
python "$BR/scripts/validate_reuse.py" --caches $CACHES --passes 20 \
    --out "$OUT/b0v2" 2>&1 | tail -22

for D in 1 8; do
  step "2. two-device per-tensor against fresh, depth=$D"
  python "$BR/scripts/benchmark_paths.py" --caches $CACHES \
      --mount /workspace/b2mount --depth $D --verify-only \
      --out "$OUT/twodevice" 2>&1 | tail -22
done

step "3. scoring alignment diagnostic"
python "$BR/scripts/scoring_alignment_diag.py" --out "$OUT/alignment" 2>&1 | tail -12

step "done"
echo VERIFY_SESSION_COMPLETE
