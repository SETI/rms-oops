##########################################################################################
# oops/frame/navigation.py: Subclass Navigation of class Frame
##########################################################################################

import numpy as np
import pytest

from polymath   import Scalar, Vector3
from oops.frame import Navigation, SpiceFrame
from oops.path  import SpicePath


def test_navigation(core_kernels):
    _ = SpicePath('MARS', 'SSB')
    mars = SpiceFrame('IAU_MARS', 'J2000')
    time = Scalar(1.e8)

    # Two small rotations about perpendicular axes tilt the Z-axis by their
    # root sum of squares
    (ay, ax) = (1.e-3, 2.e-3)
    nav = Navigation((ay, ax), mars, frame_id='+')
    assert nav.frame_id == 'IAU_MARS_NAV'
    assert nav.nparams == 2
    assert isinstance(type(nav).link, property)
    assert nav.link is None

    zaxis = nav.transform_at_time(time).rotate(Vector3.ZAXIS)
    assert (zaxis.sep(Vector3.ZAXIS).vals
            == pytest.approx(np.sqrt(ay**2 + ax**2), abs=0.5e-8))

    # A third angle rotates about the Z-axis, leaving the Z-axis untouched
    nav3 = Navigation((0., 0., 0.1), mars, frame_id='nav3')
    assert nav3.nparams == 3
    assert nav3.transform_at_time(time).rotate(Vector3.ZAXIS) == Vector3.ZAXIS
    xaxis = nav3.transform_at_time(time).rotate(Vector3.XAXIS)
    assert xaxis.sep(Vector3.XAXIS).vals == pytest.approx(0.1, abs=0.5e-12)

    # A linked Navigation tracks the angles of the object it is linked to
    linked = Navigation(nav, mars, frame_id='nav_linked')
    assert linked.link is nav
    assert linked.nparams == 2

    linked.set_params(np.array([5.e-3, 6.e-3]))
    nav.refresh()
    assert tuple(nav.angles) == (5.e-3, 6.e-3)
    assert tuple(linked.angles) == (5.e-3, 6.e-3)

    # Freezing severs the link but preserves the angles
    frozen = Navigation(linked, mars, frame_id='nav_frozen', freeze=True)
    assert frozen.is_frozen
    assert tuple(frozen.angles) == (5.e-3, 6.e-3)
##########################################################################################
