"""Encode/decode must be deterministic.

Non-determinism would make paired A/B comparison meaningless: two arms would see
different reconstructions of the same input and the quality delta would measure
the codec's jitter rather than its error bound.
"""

from __future__ import annotations

import torch

from boundrelay.codec import CodecConfig
from boundrelay.codec import reference
from tests.conftest import hidden_state


def test_encode_is_deterministic(gen):
    x = hidden_state(gen=gen)
    cfg = CodecConfig(eps=1e-2)
    a, _ = reference.encode(x, cfg)
    b, _ = reference.encode(x, cfg)
    assert a == b


def test_decode_is_deterministic(gen):
    x = hidden_state(gen=gen)
    payload, _ = reference.encode(x, CodecConfig(eps=1e-2))
    assert torch.equal(reference.decode(payload), reference.decode(payload))
