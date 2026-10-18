from facemosaic.config import DEFAULTS
from facemosaic.tracking import finalize, link_tracks

TCFG = DEFAULTS["tracking"]


def det(x, y, s=0.9, size=40):
    return [x, y, x + size, y + size, s]


def test_gap_is_interpolated():
    dets = [[det(100, 100)], [], [], [det(106, 100)], []]
    boxes, st = finalize(dets, 5, 640, 480, {**TCFG, "extend": 0, "expand": 0})
    assert all(boxes[f] for f in range(4))
    assert boxes[1][0][0] == 102 and boxes[2][0][0] == 104  # linear interpolation
    assert boxes[4] == []
    assert st["tracks_confirmed"] == 1 and st["filled_boxes"] == 2


def test_gap_longer_than_max_gap_splits_track():
    dets = [[det(100, 100)]] + [[] for _ in range(20)] + [[det(100, 100)]]
    assert len(link_tracks(dets, 0.2, 15, 0.5)) == 2


def test_low_score_track_is_rejected_but_low_frames_of_confirmed_track_are_kept():
    low_only = [[det(10, 10, 0.3)] for _ in range(5)]
    boxes, st = finalize(low_only, 5, 640, 480, TCFG)
    assert st["tracks_confirmed"] == 0 and not any(boxes)

    mixed = [[det(10, 10, 0.25)], [det(12, 10, 0.3)], [det(14, 10, 0.8)]]
    boxes, _ = finalize(mixed, 3, 640, 480, {**TCFG, "extend": 0})
    assert all(boxes)  # low-score frames before the high-score frame are masked too


def test_extend_and_expand_and_clip():
    dets = [[], [], [det(0, 0, 0.9, 40)], [], []]
    boxes, _ = finalize(dets, 5, 640, 480, {**TCFG, "extend": 2, "max_gap": 0, "expand": 0.5})
    assert all(boxes)
    x1, y1, x2, y2 = boxes[2][0]
    assert (x1, y1) == (0, 0) and x2 == 50 and y2 == 50  # enlarged, then clipped to the image


def test_two_faces_stay_separate():
    dets = [[det(10, 10), det(300, 10)] for _ in range(3)]
    tracks = link_tracks(dets, 0.2, 15, 0.5)
    assert len(tracks) == 2 and all(len(t["frames"]) == 3 for t in tracks)
