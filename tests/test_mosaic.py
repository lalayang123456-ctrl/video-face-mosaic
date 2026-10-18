import av
import numpy as np
import pytest

from facemosaic.mosaic import mosaic_frame, plane_views


def _frame(fmt, w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), np.uint8)
    return av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format=fmt)


@pytest.mark.parametrize("fmt", ["yuv420p", "yuv420p10le", "nv12", "yuv444p", "rgb24", "bgr0", "gray", "p010le"])
def test_mosaic_only_touches_box(fmt):
    fr = _frame(fmt)
    before = [v.copy() for v, _, _ in plane_views(fr)]
    mosaic_frame(fr, [[10, 8, 42, 40]], blocks=4, min_block=1)
    for (v, sx, sy), b in zip(plane_views(fr), before):
        mask = np.zeros(v.shape[:2], bool)
        mask[8 // sy: -(-40 // sy), 10 // sx: -(-42 // sx)] = True
        assert np.array_equal(v[~mask], b[~mask]), "pixels outside the box were modified"
        assert not np.array_equal(v[mask], b[mask]), "pixels inside the box were not modified"
        inner = v[8 // sy: -(-40 // sy), 10 // sx: -(-42 // sx)]
        # 4 x 4 cells: every cell is a single flat value
        bh, bw = inner.shape[0] // 4, inner.shape[1] // 4
        assert np.all(inner[:bh, :bw] == inner[0, 0])


def test_plane_views_are_writable_views():
    fr = _frame("yuv420p")
    v, sx, sy = plane_views(fr)[1]
    assert (sx, sy) == (2, 2) and v.shape == (24, 32)
    v[0, 0] = 7
    assert plane_views(fr)[1][0][0, 0] == 7
