##########################################################################################
# oops/path/pathshift.py: Subclass PathShift of class Path
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath   import Scalar
from oops.path  import Path, PathShift, SpicePath


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


def test_pathshift_auto_generated_path_id(core_kernels) -> None:
    """A path_id of "+" appends "_SHIFT" to the ID of the shifted Path."""

    mars = SpicePath('MARS', 'SSB')
    shifted = PathShift(10., mars, path_id='+')

    assert shifted.path_id == 'MARS_SHIFT'


def test_pathshift_rejects_an_unregistered_path(core_kernels) -> None:
    """A path ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        PathShift(10., 'NOT_A_REGISTERED_PATH')


def test_pathshift_is_fittable(core_kernels) -> None:
    """The time shift is the single fittable parameter."""

    mars = SpicePath('MARS', 'SSB')
    shifted = PathShift(10., mars)

    assert shifted.params == (10.,)
    assert shifted.nparams == 1
    assert not shifted.is_frozen


def test_pathshift_freeze_blocks_fitting(core_kernels) -> None:
    """freeze=True returns an object that can no longer be fitted."""

    mars = SpicePath('MARS', 'SSB')
    frozen = PathShift(10., mars, freeze=True)

    assert frozen.is_frozen
    assert frozen.dt == 10.


def test_pathshift_of_zero_matches_the_original(core_kernels) -> None:
    """A zero shift leaves the path where it was."""

    mars = SpicePath('MARS', 'SSB')
    shifted = PathShift(0., mars)
    time = Scalar(1.e8)

    assert shifted.event_at_time(time).pos == mars.event_at_time(time).pos


def test_pathshift_pickle(core_kernels) -> None:
    """Pickling restores the shift and the underlying Path."""

    mars = SpicePath('MARS', 'SSB')
    shifted = PathShift(10., mars)
    restored = pickle.loads(pickle.dumps(shifted))
    time = Scalar(1.e8)

    assert isinstance(restored, PathShift)
    assert restored.dt == shifted.dt
    assert restored.event_at_time(time).pos == shifted.event_at_time(time).pos


def test_pathshift_getstate_roundtrip(core_kernels) -> None:
    """The state captured by __getstate__ fully restores the object."""

    mars = SpicePath('MARS', 'SSB')
    shifted = PathShift(10., mars)
    state = shifted.__getstate__()

    copied = Path.__new__(PathShift)
    copied.__setstate__(state)
    assert copied.__getstate__() == state

##########################################################################################
