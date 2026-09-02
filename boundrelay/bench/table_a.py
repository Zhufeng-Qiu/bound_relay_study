"""Table A — native transport benchmark, on captured tensors.

Everything stays in its native bf16 layout: this is the table about bytes that
actually move.

The comparison is run at **matched error**, not at matched settings. Reporting
"int8 gets 2.0x, ours gets 2.5x" invites the obvious objection that the two are
not doing the same job. So each int8 baseline is measured first, its *achieved*
maximum absolute error is read off, and the error-bounded codec is then asked for
that same error. The question becomes the one a reader actually has: given the
same reconstruction quality, which representation moves fewer bytes?

``int8_per_channel`` earns its row because it is the first thing any inference-
systems reader asks about — it is the standard answer to activation outliers, and
a table without it invites "why not just per-channel int8?" as the opening
question of every conversation.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from boundrelay.codec import Allocation, CodecConfig, reference, serialize

EPS_GRID = [0.05, 0.15, 0.5]


def raw_pct(buf: bytes) -> float:
    h, body = serialize.read(buf)
    if h.n_groups == 0:
        return 100.0
    w = torch.frombuffer(bytearray(body[: h.n_groups]), dtype=torch.uint8)
    return 100.0 * float((w == 0).float().mean())


def int8_baseline(x: torch.Tensor, per_channel: bool) -> dict:
    """Symmetric int8, and the error it actually achieves.

    Scales are fp32: one per tensor, or one per channel (the last axis). Payload
    is the int8 codes plus those scales — counted, because a per-channel scheme
    that ignores its own metadata reports a ratio the transport never sees.
    """
    xf = x.float()
    amax = xf.abs().amax(dim=0, keepdim=True) if per_channel else xf.abs().amax()
    scale = (amax / 127.0).clamp(min=1e-30)
    q = torch.round(xf / scale).clamp(-127, 127)
    err = float((xf - q * scale).abs().max())
    nbytes = x.numel() + 4 * (x.shape[-1] if per_channel else 1)
    return {"ratio": (x.numel() * 2) / nbytes, "max_error": err,
            "bytes": nbytes, "bits_per_element": 8 * nbytes / x.numel()}


def codec_row(x: torch.Tensor, eps: float, alloc: Allocation) -> dict:
    buf, st = reference.encode(x, CodecConfig(eps=eps, allocation=alloc))
    err = reference.max_abs_error(x, reference.decode(buf))
    assert err <= eps + 1e-6, (eps, alloc, err)
    return {"ratio": st.ratio, "max_error": err, "eps": eps,
            "raw_pct": raw_pct(buf),
            "bits_per_element": 8 * st.compressed_bytes / x.numel()}


def load_corpus(root: Path) -> dict[str, torch.Tensor]:
    """Captured KV tensors, flattened to [tokens x heads, head_dim].

    The channel axis is head_dim: that is the axis per-channel allocation and
    per-channel int8 both key on, so both see the same notion of "channel".
    """
    out = {}
    for p in sorted(root.glob("*.pt")):
        x = torch.load(p)
        out[p.stem] = x.permute(0, 2, 1, 3).reshape(-1, x.shape[-1]).contiguous()
    return out


def run(corpus: Path, out: Path) -> dict:
    tensors = load_corpus(corpus)
    result = {"n_tensors": len(tensors), "eps_grid": EPS_GRID, "per_tensor": {}}

    agg: dict[str, list[float]] = {}
    for name, x in tensors.items():
        row: dict = {"shape": list(x.shape)}
        for pc in (False, True):
            tag = "int8_per_channel" if pc else "int8_per_tensor"
            b = int8_baseline(x, pc)
            row[tag] = b
            agg.setdefault(f"{tag}.ratio", []).append(b["ratio"])
            agg.setdefault(f"{tag}.max_error", []).append(b["max_error"])
            # matched-error question: same reconstruction quality, fewer bytes?
            m = codec_row(x, b["max_error"], Allocation.PER_CHANNEL)
            row[f"matched_{tag}"] = m
            agg.setdefault(f"matched_{tag}.ratio", []).append(m["ratio"])

        for eps in EPS_GRID:
            for a in Allocation:
                c = codec_row(x, eps, a)
                row[f"eps{eps:g}_{a.value}"] = c
                agg.setdefault(f"eps{eps:g}_{a.value}.ratio", []).append(c["ratio"])
        result["per_tensor"][name] = row

    result["mean"] = {k: sum(v) / len(v) for k, v in agg.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result
