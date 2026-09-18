#!/usr/bin/env bash
# Bring a pod to the point where B0, B1 and B2 can run, and record what it is.
#
# Every version that a result could depend on is captured into environment.json
# *from the running machine*, not from a lock file written on a laptop. The cuSZp
# commit in particular was never pinned before, so a number from this round and a
# number from an earlier one cannot be assumed to come from the same codec.
set -euo pipefail
cd /workspace

if [ ! -d cuSZp ]; then
  git clone --depth 1 https://github.com/szcompressor/cuSZp.git
fi
cd cuSZp
CUSZP_COMMIT=$(git rev-parse HEAD)
export PATH=$PATH:/usr/local/cuda/bin
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 >/dev/null 2>&1
cmake --build build -j "$(nproc)" >/dev/null 2>&1
LIB=/workspace/cuSZp/build/libcuSZp.so
test -f "$LIB"
cd /workspace

python - "$CUSZP_COMMIT" "$LIB" <<'PY'
import hashlib, json, subprocess, sys, torch, platform
commit, lib = sys.argv[1], sys.argv[2]
sha = hashlib.sha256(open(lib, "rb").read()).hexdigest()
def sh(*c):
    try: return subprocess.run(c, capture_output=True, text=True).stdout.strip()
    except Exception: return None
import transformers, datasets, numpy
env = {
  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
  "capability": [list(torch.cuda.get_device_capability(i))
                 for i in range(torch.cuda.device_count())],
  "torch": torch.__version__, "cuda": torch.version.cuda,
  "transformers": transformers.__version__, "datasets": datasets.__version__,
  "numpy": numpy.__version__, "python": platform.python_version(),
  "driver": sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
  "topology": sh("nvidia-smi", "topo", "-m"),
  "cuszp": {"commit": commit, "so_sha256": sha,
            "build": "Release, BUILD_SHARED_LIBS=ON, CMAKE_CUDA_ARCHITECTURES=86"},
  "workspace_mount": sh("bash", "-c", "df -h /workspace | tail -1"),
  "peer_0_1": bool(torch.cuda.can_device_access_peer(0, 1))
               if torch.cuda.device_count() > 1 else None,
}
import os; os.makedirs("/workspace/out", exist_ok=True)
json.dump(env, open("/workspace/out/environment.json", "w"), indent=2)
print(json.dumps({k: env[k] for k in ("gpus", "torch", "cuda", "driver")}, indent=1))
print("cuSZp", commit[:12], "so sha256", sha[:16])
PY
echo BRINGUP_OK
