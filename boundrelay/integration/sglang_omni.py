"""Payload transform hook for one SGLang-Omni relay backend.

Scope lock: **one edge, one backend.** If the framework exposes a common
send/recv payload boundary the hook goes there; if not, only the backend the
canonical Qwen3-Omni path actually uses is touched. Backend generality is not
claimed.

Three requirements the integration must meet before an H100 is booked:

1. ``off`` is byte-identical to the unmodified relay path.
2. Any exception, unsupported dtype, non-contiguous layout, or unprofitable
   prediction falls back to ``bf16_passthrough``. A research feature must never
   stop a request from completing correctly.
3. Every decision is written to the structured log.

Rehearsed on a 1-2B TTS model on an A40 first (D22-D23). Gate E: if the
rehearsal does not pass, the H100 session is not booked and the $27 is saved —
the project is already complete without it.

STATUS: D22-D23.
"""

from __future__ import annotations

from enum import Enum


class HookMode(str, Enum):
    OFF = "off"          # must be byte-identical to upstream
    FIXED = "fixed"      # one eps tier, no policy
    ADAPTIVE = "adaptive"


def install(mode: HookMode) -> None:
    """Install the transform hook at the relay payload boundary."""
    raise NotImplementedError("D22")


def uninstall() -> None:
    """Restore the unmodified path. Used to prove ``off`` equivalence."""
    raise NotImplementedError("D22")
