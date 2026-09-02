"""Replay manifest — frozen at D4, before any GPU spend.

The manifest is what lets the expensive corpus be captured once and reused for
every offline sweep. It is also what a reviewer downloads at reproduction level
L1 to redraw a break-even curve without touching a 60 GB checkpoint.

Layout on the network volume::

    <corpus>/
        manifest.json          # this schema
        tensors/<id>.pt        # never committed to Git
        capture.log

Writes are **incremental**: each tensor lands and the manifest is appended
before the next is captured. A session that dies at minute 50 keeps everything
up to minute 49 — the H100 session has exactly one job and it must survive a
crash.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from boundrelay.codec.contract import TensorSpec

SCHEMA_VERSION = 1


@dataclass
class CaptureEnv:
    """Provenance. Every number in the paper traces back to one of these."""

    gpu: str                      # "A40" | "L40S" | "H100 SXM"
    gpu_count: int
    driver: str
    cuda: str
    torch: str
    model_id: str
    model_revision: str           # pinned commit/revision, never "main"
    weight_dtype: str             # "bf16" | "int4_autoround" — changes distribution
    sglang_omni_sha: str
    image_digest: str
    notes: str = ""


@dataclass
class Manifest:
    schema_version: int = SCHEMA_VERSION
    corpus_id: str = ""
    env: CaptureEnv | None = None
    tensors: list[TensorSpec] = field(default_factory=list)

    def append(self, path: Path, spec: TensorSpec) -> None:
        """Add one tensor and flush. Crash-safe by construction."""
        self.tensors.append(spec)
        write(path, self)


def write(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, default=str))


def read(path: Path) -> Manifest:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema {raw.get('schema_version')} != {SCHEMA_VERSION}")
    env = CaptureEnv(**raw["env"]) if raw.get("env") else None
    return Manifest(
        schema_version=raw["schema_version"],
        corpus_id=raw.get("corpus_id", ""),
        env=env,
        tensors=[TensorSpec(**t) for t in raw.get("tensors", [])],
    )
