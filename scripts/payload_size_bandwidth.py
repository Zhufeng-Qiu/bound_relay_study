"""Control for `fsync_offload.py`: is the compressed arm penalised for being
compressed, or for being small?

The compressed payload achieved 0.40-0.70x the bandwidth the raw one did, on
both filesystems. Is that a property of the payload, or of its size?

Same call pattern, same file layout, same fsync -- only the total volume changes,
and the bytes are incompressible noise in both cases so nothing about the content
differs. If 77.7 MB is slower per byte than 234.9 MB here too, the compressed arm
was not penalised for being compressed; it was penalised for being small.
"""
import os, statistics as st, sys, time
from pathlib import Path

RAW, COMP = 234_881_024, 77_693_156
CHUNK = 4_194_304


def run(root: Path, total: int, do_fsync: bool, reps: int = 7) -> float:
    root.mkdir(parents=True, exist_ok=True)
    p = root / "sz.bin"
    buf = os.urandom(CHUNK)
    n_full, tail = divmod(total, CHUNK)
    s = []
    for _ in range(reps + 1):
        with open(p, "wb") as f:
            f.truncate(total)
        t = time.perf_counter()
        fd = os.open(p, os.O_WRONLY)
        off = 0
        for _ in range(n_full):
            os.pwrite(fd, buf, off); off += CHUNK
        if tail:
            os.pwrite(fd, buf[:tail], off)
        if do_fsync:
            os.fsync(fd)
        os.close(fd)
        s.append(time.perf_counter() - t)
    p.unlink(missing_ok=True)
    return st.median(s[1:])


for root in ("/workspace/szbw", "/root/szbw"):
    print(f"\n=== {root} ===")
    for do_fsync in (True, False):
        rows = {}
        for label, total in (("compressed-sized 77.7 MB", COMP), ("raw-sized 234.9 MB", RAW)):
            t = run(Path(root), total, do_fsync)
            rows[label] = (t, total / 1e9 / t)
            print(f"  fsync={str(do_fsync):<5} {label:<26} {t*1e3:7.1f} ms  "
                  f"{total/1e9/t:6.3f} GB/s")
        small = rows["compressed-sized 77.7 MB"][1]
        big = rows["raw-sized 234.9 MB"][1]
        print(f"  fsync={str(do_fsync):<5} {'small payload gets':<26} {small/big:6.2f}x "
              f"the bandwidth of the large one")
