################################################################################
# oops/path/pathshift.py: Subclass PathShift of class Path
################################################################################

import numpy as np

from polymath   import Scalar
from oops.path  import PathShift, SpicePath


def test_pathshift(core_kernels):
    DT = 10.
    mars = SpicePath('MARS', 'SSB')
    shifted = PathShift(DT, mars, path_id='mars_shifted')
    assert shifted.dt == DT
    assert shifted.link is None

    # The shifted path at time t matches the original path at time t + dt
    time = Scalar(1.e8 + np.arange(10) * 1000.)
    assert shifted.event_at_time(time).pos == mars.event_at_time(time + DT).pos
    assert shifted.event_at_time(time).vel == mars.event_at_time(time + DT).vel

    # A linked PathShift tracks the offset of the object it is linked to
    linked = PathShift(shifted, mars, path_id='mars_shifted_2')
    assert linked.link is shifted
    assert linked.dt == DT

    shifted.set_params(np.array([2. * DT]))
    linked.refresh()
    assert linked.dt == 2. * DT
    assert linked.event_at_time(time).pos == mars.event_at_time(time + 2. * DT).pos

    # Freezing severs the link but preserves the offset
    frozen = PathShift(linked, mars, path_id='mars_shifted_3', freeze=True)
    assert frozen.is_frozen
    assert frozen.dt == 2. * DT
################################################################################
