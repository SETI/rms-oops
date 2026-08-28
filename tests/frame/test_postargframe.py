################################################################################
# tests/frame/test_postargframe.py
################################################################################

from polymath   import Vector3
from oops.frame import Frame, PosTargFrame


def test_postargframe():
    postarg = PosTargFrame(0.0001, 0.0002, "J2000")
    transform = postarg.transform_at_time(0.)
    rotated = transform.rotate(Vector3.ZAXIS)

    assert abs(rotated.vals[0] - 0.0001) < 1.e-8
    assert abs(rotated.vals[1] - 0.0002) < 1.e-8
################################################################################
