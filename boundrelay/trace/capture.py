"""Tensor capture at a stage edge.

Two corpora, captured with the same code path:

``hidden_state`` / ``encoder_embedding``
    From a **speech** Qwen3-Omni pipeline. Text-only mode never produces the
    Thinker -> Talker edge, so it cannot substitute (Runbook D7).

``kv_cache``
    From a 1-3B open model on the A40 (D8). Second tensor family, different
    statistics, and the bridge to disaggregated-serving KV transport.

Both edges are captured in the *same* session: the encoder -> Thinker embedding
carries three to four orders of magnitude more payload than Thinker -> Talker,
and that span is what gives Figure 1 an x-axis instead of a single point.

STATUS: D5 (dry-run against a synthetic stand-in) then D7-D8.
"""

from __future__ import annotations

from pathlib import Path

import torch

from boundrelay.trace.manifest import Manifest


def capture(tensor: torch.Tensor, name: str, out_dir: Path, manifest: Manifest) -> None:
    """Persist one tensor and append it to the manifest. Idempotent per name."""
    raise NotImplementedError("D5")


def sanitize(manifest: Manifest) -> Manifest:
    """Strip anything that must not reach a public repo.

    Tokens, absolute cache paths, environment dumps, and any profiler payload
    that could carry prompt content. Only derived statistics and the reproduction
    scripts are published.
    """
    raise NotImplementedError("D8")
