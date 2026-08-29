##########################################################################################
# oops/path/pathshift.py: Subclass PathShift of class Path
##########################################################################################

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


def test_pathshift_pools_waypoints_only_once_frozen(core_kernels):
    DT = 10.
    mars = SpicePath('MARS', 'SSB')

    # While a PathShift is still fittable, two of them with the same offset must stay
    # distinct: either one can be fitted away from the other.
    a = PathShift(DT, mars)
    b = PathShift(DT, mars)
    assert a.waypoint is not b.waypoint

    b.set_params(np.array([2. * DT]))
    assert a.dt == DT
    assert b.dt == 2. * DT

    # Once frozen, two PathShifts that define the same offset share one waypoint
    c = PathShift(DT, mars, freeze=True)
    d = PathShift(DT, mars, freeze=True)
    assert c.waypoint is d.waypoint

    # ...and freezing an existing one moves it into the same pool
    a.freeze()
    assert a.waypoint is c.waypoint
##########################################################################################
