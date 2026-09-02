"""Wire format: magic, JSON header, payload, checksum.

Every payload is self-describing, so a decode failure is loud rather than a
silently wrong tensor. The receiver validates magic, version and CRC before
reconstructing anything; a mismatch raises, and the integration layer turns that
into a ``bf16_passthrough`` fallback rather than shipping corrupt state.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import asdict, dataclass

CODEC_VERSION = 1
MAGIC = b"BRLY"


@dataclass
class Header:
    codec_version: int
    eps: float
    allocation: str
    bit_width: int          # representative width; -1 when groups differ
    original_dtype: str
    original_shape: tuple[int, ...]
    block_size: int
    n_groups: int
    compressed_bytes: int
    crc32: int


def write(header: Header, payload: bytes) -> bytes:
    """Serialise header + payload into one buffer."""
    header.crc32 = zlib.crc32(payload) & 0xFFFFFFFF
    header.compressed_bytes = len(payload)
    blob = json.dumps(asdict(header)).encode("utf-8")
    return MAGIC + struct.pack("<I", len(blob)) + blob + payload


def read(buf: bytes) -> tuple[Header, bytes]:
    """Parse and validate. Raises on magic, version, or checksum mismatch."""
    if buf[:4] != MAGIC:
        raise ValueError("not a BoundRelay payload")
    (blob_len,) = struct.unpack("<I", buf[4:8])
    raw = json.loads(buf[8 : 8 + blob_len].decode("utf-8"))
    raw["original_shape"] = tuple(raw["original_shape"])
    header = Header(**raw)

    if header.codec_version != CODEC_VERSION:
        raise ValueError(f"codec version {header.codec_version} != {CODEC_VERSION}")

    payload = buf[8 + blob_len :]
    if (zlib.crc32(payload) & 0xFFFFFFFF) != header.crc32:
        raise ValueError("payload checksum mismatch")
    return header, payload
