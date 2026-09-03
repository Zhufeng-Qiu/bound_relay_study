"""Peer-to-peer device copies that report success and move nothing.

This started as a debugging detour and became the sharpest measurement in the
project, so it gets its own script.

`torch.cuda.can_device_access_peer(0, 1)` returns True on this host.
`cudaDeviceEnablePeerAccess` returns `cudaErrorPeerAccessAlreadyEnabled`, so it is
not merely available, it is on. `cudaMemcpyPeer` returns `cudaSuccess`. And the
destination buffer is left exactly as it was: every element zero.

Nothing raises. Nothing warns. A KV-transfer path built on the peer copy — which
is the obvious way to build one, and what `can_device_access_peer` exists to tell
you — moves zeros between GPUs and reports that it worked. The receiver decodes
them into a plausible-looking tensor of the right shape and dtype.

This is the second rented multi-GPU host where the peer path has misbehaved: it
first showed up as an `inf` in a correctness gate on 2xA40 in CA-MTL-1 and was
worked around by comparing on the host. Two hosts, two GPU models, two data
centres. It is why the transport measured in this project is host-staged, a choice
that had looked like a limitation of the harness.

The script separates the two explanations a single bad result cannot distinguish:

* **An ordering fault** — the copy is issued before the producing kernel on the
  other device has finished. Synchronising both devices before the copy removes it.
* **A mapping fault** — the write never lands. Synchronising changes nothing.

It also sweeps size, direction, which device is current when the copy is issued,
and four different APIs, because "peer copies are broken" is only worth reporting
if it is not one narrow path in one direction at one size.
"""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import torch

N = 1 << 20


def pattern(dev: int, n: int = N) -> torch.Tensor:
    """A value per element that no memset would produce, so a zeroed destination
    and a stale one are distinguishable from a correct one."""
    torch.cuda.set_device(dev)
    x = (torch.arange(n, dtype=torch.float32, device=f"cuda:{dev}") % 977) + 1.0
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    return x


def score(dst: torch.Tensor, src: torch.Tensor) -> dict:
    a, b = src.cpu(), dst.cpu()
    return {"exact_fraction": float((a == b).float().mean()),
            "zero_fraction": float((b == 0).float().mean())}


def sync() -> None:
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)


def main() -> int:
    if torch.cuda.device_count() < 2:
        raise SystemExit("needs two devices")
    rt = ctypes.CDLL("libcudart.so")
    rt.cudaMemcpyPeer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_size_t]
    rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_int]
    rt.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]

    out: dict = {
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "can_access_peer_0_1": torch.cuda.can_device_access_peer(0, 1),
        "can_access_peer_1_0": torch.cuda.can_device_access_peer(1, 0),
        "paths": {}, "size_sweep": {}, "repeats": {},
    }
    torch.cuda.set_device(0)
    out["enable_peer_access_rc"] = int(rt.cudaDeviceEnablePeerAccess(1, 0))  # 704 = already on

    print(f"peer 0->1 {out['can_access_peer_0_1']}   peer 1->0 {out['can_access_peer_1_0']}   "
          f"cudaDeviceEnablePeerAccess rc {out['enable_peer_access_rc']}")
    print(f"{'path':<56} {'exact':>7} {'zeros':>7}")

    def record(label: str, dst, src) -> None:
        s = score(dst, src)
        out["paths"][label] = s
        print(f"{label:<56} {s['exact_fraction']:7.3f} {s['zero_fraction']:7.3f}")

    # every path is fully synchronised first: an ordering fault cannot survive this
    for cur in (0, 1):
        src = pattern(0)
        torch.cuda.set_device(cur)
        dst = torch.zeros(N, dtype=torch.float32, device="cuda:1")
        sync()
        dst.copy_(src)
        sync()
        record(f"torch copy_ 0->1, synchronised, current device {cur}", dst, src)

    src = pattern(0)
    torch.cuda.set_device(0)
    sync()
    d = src.cpu().to("cuda:1")
    sync()
    record("torch, routed through host memory", d, src)

    src = pattern(0)
    dst = torch.zeros(N, dtype=torch.float32, device="cuda:1")
    sync()
    rc = rt.cudaMemcpyPeer(ctypes.c_void_p(dst.data_ptr()), 1,
                           ctypes.c_void_p(src.data_ptr()), 0, N * 4)
    sync()
    out["cudaMemcpyPeer_rc"] = int(rc)
    record(f"cudaMemcpyPeer, runtime returned {rc}", dst, src)

    src = pattern(0)
    dst = torch.zeros(N, dtype=torch.float32, device="cuda:1")
    sync()
    rc = rt.cudaMemcpy(ctypes.c_void_p(dst.data_ptr()),
                       ctypes.c_void_p(src.data_ptr()), N * 4, 4)  # cudaMemcpyDefault
    sync()
    out["cudaMemcpyDefault_rc"] = int(rc)
    record(f"cudaMemcpy(cudaMemcpyDefault), runtime returned {rc}", dst, src)

    src1 = pattern(1)
    torch.cuda.set_device(1)
    dst0 = torch.zeros(N, dtype=torch.float32, device="cuda:0")
    sync()
    dst0.copy_(src1)
    sync()
    record("torch copy_ 1->0, the reverse direction", dst0, src1)

    print(f"\n{'elements':>12} {'bytes':>12} {'exact':>7} {'zeros':>7}")
    for n in (1, 4096, 262_144, 1_048_576, 8_388_608):
        s_ = pattern(0, n)
        torch.cuda.set_device(0)
        d_ = torch.zeros(n, dtype=torch.float32, device="cuda:1")
        sync()
        d_.copy_(s_)
        sync()
        r = score(d_, s_)
        out["size_sweep"][str(n)] = r
        print(f"{n:12d} {n*4:12d} {r['exact_fraction']:7.3f} {r['zero_fraction']:7.3f}")

    # intermittent or deterministic? a flaky peer link and a dead one are
    # different operational problems
    ex = []
    for _ in range(10):
        s_ = pattern(0)
        torch.cuda.set_device(0)
        d_ = torch.zeros(N, dtype=torch.float32, device="cuda:1")
        sync()
        d_.copy_(s_)
        sync()
        ex.append(score(d_, s_)["exact_fraction"])
    out["repeats"] = {"n": len(ex), "min": min(ex), "max": max(ex)}
    print(f"\nrepeated 10x: exact fraction min {min(ex):.3f} max {max(ex):.3f}")

    dst = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/out/peer_copy_audit.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
