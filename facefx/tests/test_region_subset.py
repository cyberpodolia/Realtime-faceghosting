"""Region subset helper tests for triangle filtering."""

from __future__ import annotations

import numpy as np

from facefx.main import (
    _active_region_names,
    _expand_region_indices,
    _filter_simplices_by_indices,
)


def test_active_region_names():
    assert _active_region_names("all") == ("forehead", "eyes", "mouth")
    assert _active_region_names("eyes") == ("eyes",)


def test_expand_region_indices_adds_one_hop_neighbors():
    simplices = np.array(
        [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
        ],
        dtype=np.int32,
    )
    expanded = _expand_region_indices({1}, simplices, hops=1)
    assert expanded == {0, 1, 2}

    expanded2 = _expand_region_indices({1}, simplices, hops=2)
    assert expanded2 == {0, 1, 2, 3, 4}


def test_filter_simplices_by_indices_keeps_related_triangles():
    simplices = np.array(
        [
            [0, 1, 2],
            [2, 3, 4],
            [7, 8, 9],
        ],
        dtype=np.int32,
    )
    out = _filter_simplices_by_indices(simplices, {3})
    assert out.shape == (0, 3)

    out2 = _filter_simplices_by_indices(simplices, {2, 3, 4})
    assert out2.shape == (1, 3)
    assert np.array_equal(out2[0], np.array([2, 3, 4], dtype=np.int32))
