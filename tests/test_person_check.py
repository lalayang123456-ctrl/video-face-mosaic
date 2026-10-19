from fractions import Fraction

import av
import numpy as np

from facemosaic.config import DEFAULTS
from facemosaic.detectors import face_in_person
from facemosaic.pipeline import person_check
from facemosaic.tracking import finalize

PC = DEFAULTS["tracking"]["person_check"]
PERSON = np.array([[100, 50, 300, 450, 0.9]], np.float32)  # a standing / seated person


def test_face_in_head_region_matches():
    assert face_in_person([180, 70, 220, 120], PERSON)  # head at the top of the person box


def test_object_in_lower_part_of_person_box_does_not_match():
    # e.g. a jar on the table in front of a seated person: inside the box, but far below the head
    assert not face_in_person([180, 350, 230, 410], PERSON)


def test_box_far_from_any_person_does_not_match():
    assert not face_in_person([500, 70, 540, 120], PERSON)
    assert not face_in_person([180, 70, 220, 120], np.zeros((0, 5), np.float32))


def test_finalize_drops_tracks_rejected_by_filter():
    dets = [[[10, 10, 50, 50, 0.9], [300, 10, 340, 50, 0.9]] for _ in range(3)]
    boxes, st = finalize(dets, 3, 640, 480, {**DEFAULTS["tracking"], "extend": 0},
                         track_filter=lambda tracks: [t["boxes"][0][0] < 100 for t in tracks])
    assert st["tracks_confirmed"] == 1 and st["tracks_rejected"] == 1
    assert all(len(b) == 1 and b[0][0] < 100 for b in boxes)


class FakePersons:
    """Reports one person whose head region covers x 100..300 near the top of the frame."""

    def __init__(self):
        self.calls = 0

    def detect(self, img, conf):
        self.calls += 1
        return PERSON


def _video(path, n=5):
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=Fraction(30))
        s.width, s.height, s.pix_fmt = 640, 480, "yuv420p"
        for i in range(n):
            f = av.VideoFrame.from_ndarray(np.zeros((480, 640, 3), np.uint8), format="rgb24")
            f.pts = i
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)


def test_person_check_keeps_faces_and_rejects_objects(tmp_path):
    src = tmp_path / "v.mp4"
    _video(src)

    def track(box, score):
        return {"frames": [0, 1, 2], "boxes": [box] * 3, "scores": [score] * 3}

    tracks = [
        track([180, 70, 220, 120], 0.6),  # face in a person's head region -> keep
        track([450, 300, 530, 380], 0.7),  # object, no person around -> reject
        track([600, 10, 615, 25], 0.55),  # tiny distant face -> keep without a check
        track([400, 200, 560, 380], 0.9),  # very confident face (e.g. filling the frame) -> keep without a check
    ]
    det = FakePersons()
    assert person_check(src, tracks, PC, det, None) == [True, False, True, True]
    assert det.calls >= 1
