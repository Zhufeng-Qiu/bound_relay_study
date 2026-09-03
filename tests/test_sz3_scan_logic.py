"""Tests for the parts of the SZ3 scan that carry a claim.

Two of them do. `grid` decides how many configurations the prediction oracle gets
to search, and the whole comparison is "an oracle over every reachable
configuration still loses to one frozen default" -- if the grid silently contained
duplicates, the oracle would be weaker than advertised and the claim would be
overstated in exactly the direction that flatters it.

`axis_predictability` is the instrument behind the mechanism result: "1.414 means
white, 0.49 for keys along tokens means predictable". An instrument that reports
1.414 for a smooth field would make the mechanism story up.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "sz3_full_scan", Path(__file__).resolve().parents[1] / "scripts" / "sz3_full_scan.py")
sz3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sz3)


def test_grid_has_no_duplicate_configurations():
    """A duplicate would pad the oracle's search without widening it."""
    for ndim in (2, 3):
        tags = [sz3.tag(a, s) for a, s in sz3.grid(ndim)]
        assert len(tags) == len(set(tags)), f"ndim={ndim} repeats a configuration"


def test_grid_offers_one_direction_per_axis_permutation():
    """InterpolationDirection indexes a permutation of the axes, so a 2-D view has
    two and a 3-D view six. Handing a 2-D tensor six would count four duplicates."""
    n3 = sum(1 for a, s in sz3.grid(3) if "InterpolationDirection" in s)
    n2 = sum(1 for a, s in sz3.grid(2) if "InterpolationDirection" in s)
    assert n3 == 2 * 2 * 6      # two algorithms x two kernels x six directions
    assert n2 == 2 * 2 * 2


def test_grid_contains_nopred_exactly_once():
    """NOPRED is the frozen side of the comparison; it must not be in the oracle's
    search, and it must be present to be compared against."""
    for ndim in (2, 3):
        assert sum(1 for a, _ in sz3.grid(ndim) if a == "NOPRED") == 1


def test_axis_predictability_reports_root_two_for_white_noise():
    """The calibration the mechanism claim is read against."""
    g = torch.randn(1, 8, 512, 128, generator=torch.Generator().manual_seed(0))
    r = sz3.axis_predictability(g)
    for axis in ("adj_over_std_head_dim", "adj_over_std_tokens", "adj_over_std_heads"):
        assert r[axis] == pytest.approx(np.sqrt(2.0), rel=0.02), axis


def test_axis_predictability_finds_structure_on_the_axis_that_has_it():
    """A field smooth along tokens and white along the others must be reported that
    way, and only that way -- otherwise the K/V split could be an artefact of the
    statistic rather than of the data."""
    # x[h, s, d] = a[h, d] * sin(s): each (head, channel) carries an independent
    # amplitude, so stepping along either of those axes lands on an unrelated value
    # -- white. Stepping along tokens moves along one slow sine -- predictable.
    #
    # A field that is merely *constant* along an axis would not test this: constant
    # is the most predictable an axis can be, and the statistic correctly reports it
    # near zero rather than near sqrt(2). The first draft of this test made that
    # mistake and asserted the wrong number.
    a = torch.randn(8, 1, 128, generator=torch.Generator().manual_seed(1))
    sn = torch.sin(torch.linspace(0, 6.0, 512)).reshape(1, 512, 1)
    x = (a * sn).unsqueeze(0)
    r = sz3.axis_predictability(x)
    assert r["adj_over_std_tokens"] < 0.2, r
    assert r["adj_over_std_head_dim"] == pytest.approx(np.sqrt(2.0), rel=0.05), r
    assert r["adj_over_std_heads"] == pytest.approx(np.sqrt(2.0), rel=0.05), r


def test_axis_predictability_is_scale_free():
    """It normalises by the tensor's own std, so a 100x rescale must not move it --
    per-tensor std spans 115x across this corpus."""
    x = torch.randn(1, 4, 128, 64, generator=torch.Generator().manual_seed(2))
    a = sz3.axis_predictability(x)
    b = sz3.axis_predictability(x * 100.0)
    for k in ("adj_over_std_head_dim", "adj_over_std_tokens", "adj_over_std_heads"):
        assert a[k] == pytest.approx(b[k], rel=1e-4)
