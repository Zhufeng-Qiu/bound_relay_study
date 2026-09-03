"""The codec's public contract.

This module is *declarative*. It states what BoundRelay guarantees and — just as
importantly — what it explicitly does not. Nothing here computes; the guarantees
are enforced by ``tests/test_error_bound_finite.py`` and
``tests/test_nonfinite_bypass.py``, which are deliberately two separate suites.

Guarantee
---------
For a block in which **every value is finite**::

    max |x - x_hat| <= eps + FP_TOLERANCE

Explicit exclusions
-------------------
* Blocks containing NaN or +/-Inf are transmitted **bit-exact** and are outside
  the eps guarantee. |x - x_hat| is undefined for them, so folding them into the
  error-bound test would silently hollow out the contract.
* Output quality is *not* guaranteed. An element-wise error bound says nothing
  about WER, perplexity, or speaker similarity; those are measured separately
  against a gate frozen before any performance sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

#: Slack for float round-off in the bound check. The reconstruction arithmetic
#: itself introduces error below the quantisation step, so the assertion is
#: ``<= eps + FP_TOLERANCE`` rather than ``<= eps``.
FP_TOLERANCE: float = 1e-6


class Mode(str, Enum):
    """Transport mode selected per chunk by the policy.

    ``BF16_PASSTHROUGH`` is the bypass path and the correctness oracle. It is
    *not* called "fp16": the payloads are bf16, and bf16 carries fp32's exponent
    range. Narrowing to fp16 would overflow on outlier channels, which is a real
    phenomenon in LLM activations, not a hypothetical.
    """

    BF16_PASSTHROUGH = "bf16_passthrough"
    INT8_PER_TENSOR = "int8_per_tensor"
    INT8_PER_CHANNEL = "int8_per_channel"
    ERROR_BOUNDED = "error_bounded"


class Allocation(str, Enum):
    """How the error budget is distributed across a tensor.

    The Figure 2 independent variable. Scientific compressors assume spatial
    smoothness; LLM intermediate states are not smooth along the token axis, but
    do carry known structure along the channel axis (activation outliers). Which
    axis the budget should follow is the question, not an implementation detail.
    """

    BLOCKWISE = "blockwise"
    PER_CHANNEL = "per_channel"
    PER_TOKEN = "per_token"


class BypassReason(str, Enum):
    """Why a chunk fell back to ``BF16_PASSTHROUGH``.

    Every bypass is logged with one of these. "No reason recorded" is a bug.
    """

    NOT_PROFITABLE = "not_profitable"          # cost model says compression loses
    QUALITY_TIER_DISALLOWED = "quality_tier_disallowed"
    UNSUPPORTED_DTYPE = "unsupported_dtype"
    NON_CONTIGUOUS = "non_contiguous"
    TENSOR_TOO_SMALL = "tensor_too_small"
    BOUND_CHECK_FAILED = "bound_check_failed"  # fail closed, never ship bad data
    ENCODE_ERROR = "encode_error"


@dataclass(frozen=True)
class CodecConfig:
    """One codec configuration. Hashable, so it can key a cost-model lookup."""

    eps: float
    allocation: Allocation = Allocation.BLOCKWISE
    block_size: int = 256
    #: None => derive the smallest width satisfying eps; else force this width.
    bit_width: int | None = None

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.bit_width is not None and self.bit_width not in (4, 6, 8):
            raise ValueError("bit_width must be one of 4, 6, 8")

    @classmethod
    def relative(cls, x, c: float, **kw) -> "CodecConfig":
        """Build a config whose bound is ``c`` times this tensor's own std.

        A single absolute eps means different things to different tensors. Across
        the captured KV corpus the per-tensor std spans 115x and the range spans
        255x, so eps = 0.15 is 0.009 std on one tensor and 1.03 std on another --
        larger than the signal. Any average taken across tensors at a fixed
        absolute eps is therefore not a comparison, and the ones already
        published were withdrawn for this reason.

        Within a single tensor an absolute bound is still the right contract; it
        is only cross-tensor aggregation that requires normalising first.
        """
        import torch  # local: keep the contract module import-light
        sd = float(x.float().std()) if torch.is_tensor(x) else float(x)
        if sd <= 0:
            raise ValueError("cannot normalise against zero std")
        return cls(eps=c * sd, **kw)


@dataclass
class EncodeStats:
    """What one encode call actually did. Serialised into every result JSON."""

    original_bytes: int
    compressed_bytes: int
    bit_width: int
    max_abs_error: float
    #: Blocks sent bit-exact because they held NaN/Inf. Expected to be 0 on a
    #: healthy model; a nonzero value is a finding, not noise to be smoothed over.
    nonfinite_blocks: int = 0
    total_blocks: int = 0
    bypass_reason: BypassReason | None = None

    @property
    def ratio(self) -> float:
        """Compression ratio against the **bf16** original (2 bytes/element).

        Never against an fp32 upcast. The scientific-compressor coordinate table
        upcasts bf16 to fp32 losslessly so cuSZp/SZ3 can read it; reporting a
        ratio against those fp32 bytes would hand every such baseline a free 2x.
        """
        return self.original_bytes / max(self.compressed_bytes, 1)

    @property
    def nonfinite_block_rate(self) -> float:
        return self.nonfinite_blocks / max(self.total_blocks, 1)


@dataclass
class TensorSpec:
    """Identity of a captured tensor. Written into the replay manifest."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    contiguous: bool
    producer: str          # e.g. "thinker"
    consumer: str          # e.g. "talker"
    family: str            # "hidden_state" | "encoder_embedding" | "kv_cache"
    bytes_per_request: int
    extras: dict = field(default_factory=dict)
