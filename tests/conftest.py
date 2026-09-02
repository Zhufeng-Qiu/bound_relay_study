"""Shared fixtures. Tensor shapes mirror the real capture targets.

Hidden size 2048 is Qwen3-Omni's thinker (``thinker_config.text_config.hidden_size``);
the audio and vision encoders both emit 2048 as well, and the talker is 1024.
Testing on the real widths from day one means the property suite exercises the
same layouts the A40 corpus will carry.
"""

from __future__ import annotations

import pytest
import torch

HIDDEN = 2048
TALKER_HIDDEN = 1024

EPS_VALUES = [1e-3, 1e-2, 1e-1]
BIT_WIDTHS = [4, 6, 8]


@pytest.fixture(params=EPS_VALUES, ids=lambda e: f"eps{e:g}")
def eps(request) -> float:
    return request.param


@pytest.fixture
def gen() -> torch.Generator:
    g = torch.Generator().manual_seed(20260902)
    return g


def hidden_state(tokens: int = 64, gen: torch.Generator | None = None) -> torch.Tensor:
    """Gaussian stand-in for a thinker hidden-state chunk, bf16."""
    return torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)


def with_channel_outliers(x: torch.Tensor, n: int = 8, scale: float = 40.0) -> torch.Tensor:
    """Amplify a few channels.

    Not decoration: activation outliers concentrated in a handful of channels are
    the documented behaviour of LLM activations, and they are precisely what
    per-channel budget allocation is hypothesised to exploit. A codec that only
    ever sees clean Gaussians is not being tested on its target distribution.
    """
    y = x.clone().float()
    y[:, :n] *= scale
    return y.to(x.dtype)
