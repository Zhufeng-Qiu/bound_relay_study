"""Wire format: metadata + payload + checksum.

Every payload is self-describing so a decode failure is loud rather than a
silently wrong tensor. Header fields:

    codec_version, eps, allocation, bit_width,
    original_dtype, original_shape, compressed_bytes, crc32

The receiver validates the checksum before reconstructing. A mismatch raises;
the integration layer catches it and falls back to ``bf16_passthrough``.

STATUS: D2.
"""

from __future__ import annotations

from dataclasses import dataclass

CODEC_VERSION = 1
MAGIC = b"BRLY"


@dataclass
class Header:
    codec_version: int
    eps: float
    allocation: str
    bit_width: int
    original_dtype: str
    original_shape: tuple[int, ...]
    compressed_bytes: int
    crc32: int


def write(header: Header, payload: bytes) -> bytes:
    """Serialise header + payload into one buffer."""
    raise NotImplementedError("D2")


def read(buf: bytes) -> tuple[Header, bytes]:
    """Parse and validate. Raises on magic/version/checksum mismatch."""
    raise NotImplementedError("D2")
