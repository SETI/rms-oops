##########################################################################################
# tests/path/test_multipath.py
##########################################################################################

import pickle

import numpy as np
import pytest

from polymath  import Scalar
from oops.path import MultiPath, Path, SpicePath


def test_multipath(core_kernels):
    sun   = SpicePath("SUN", "SSB")
    earth = SpicePath("EARTH", "SSB")
    moon  = SpicePath("MOON", "EARTH")

    test = MultiPath([sun,earth,moon], "SSB", path_id='+')

    # behavior changed
    # self.assertEqual(test.path_id, "SUN+EARTH+MOON")
    assert test.shape == (3,)

    # Single time
    event0 = test.event_at_time(0.)
    assert event0.shape == (3,)

    # Triple of times, shape = [3]
    event012 = test.event_at_time((0., 1.e5, 2.e5))
    assert event012.shape == (3,)

    assert event012.pos[0] == event0.pos[0]
    assert event012.vel[0] == event0.vel[0]
    assert event012.pos[1] != event0.pos[1]
    assert event012.vel[1] != event0.vel[1]
    assert event012.pos[2] != event0.pos[2]
    assert event012.vel[2] != event0.vel[2]

    # Times shaped [2,1]
    event01x = test.event_at_time([[0.], [1.e5]])
    assert event01x.shape == (2,3)

    assert event01x.pos[0,0] == event0.pos[0]
    assert event01x.vel[0,0] == event0.vel[0]
    assert event01x.pos[0,1] == event0.pos[1]
    assert event01x.vel[0,1] == event0.vel[1]
    assert event01x.pos[0,2] == event0.pos[2]
    assert event01x.vel[0,2] == event0.vel[2]

    assert event01x.pos[1,1] == event012.pos[1]
    assert event01x.vel[1,1] == event012.vel[1]
    assert event01x.pos[1,2] != event012.pos[2]
    assert event01x.vel[1,2] != event012.pos[2]

    # Triple of times, at all times, shape [3,1]
    event012a = test.event_at_time([[0.], [1.e5], [2.e5]])
    assert event012a.shape == (3,3)

    assert event012a.pos[0,:] == event0.pos
    assert event012a.vel[0,:] == event0.vel

    assert event012a.pos[0,0] == event012.pos[0]
    assert event012a.vel[0,0] == event012.vel[0]
    assert event012a.pos[1,1] == event012.pos[1]
    assert event012a.vel[1,1] == event012.vel[1]
    assert event012a.pos[2,2] == event012.pos[2]
    assert event012a.vel[2,2] == event012.vel[2]

    assert event012a.pos[0:2] == event01x.pos
    assert event012a.vel[0:2] == event01x.vel


def test_multipath_rejects_a_multidimensional_array(core_kernels):
    """A MultiPath is 1-D by definition, and says so rather than failing further in."""

    sun   = SpicePath("SUN", "SSB")
    earth = SpicePath("EARTH", "SSB")
    moon  = SpicePath("MOON", "EARTH")
    grid = np.array([sun, earth, moon, sun], dtype='object').reshape(2, 2)

    with pytest.raises(ValueError, match='cannot be multidimensional'):
        MultiPath(grid)


def test_multipath_quick_path_returns_self_when_quick_is_false(core_kernels):
    """A numpy False and an integer zero must disable quickening the way False does."""

    test = MultiPath([SpicePath("SUN", "SSB"), SpicePath("EARTH", "SSB")], "SSB")

    assert test.quick_path(0., quick=np.bool_(False)) is test


def test_multipath_quick_path_still_quickens_when_quick_is_none(core_kernels):
    """None means "use the defaults", not "do not quicken"."""

    test = MultiPath([SpicePath("SUN", "SSB"), SpicePath("EARTH", "SSB")], "SSB")

    assert test.quick_path((0., 1.e5), quick=None) is not test


def test_multipath_quick_path_accepts_time_with_leading_axes(core_kernels):
    """`time[..., k]` selects one path's times; k is an index, not a tuple."""

    test = MultiPath([SpicePath("SUN", "SSB"), SpicePath("EARTH", "SSB")], "SSB")
    time = np.arange(6.).reshape(3, 2) * 1.e5

    assert test.quick_path(time, quick=None).shape == (2,)


def _three_planets(core_kernels) -> MultiPath:
    """A MultiPath gathering Mars, Venus, and Earth relative to the SSB."""

    for name in ('MARS', 'VENUS', 'EARTH'):
        SpicePath(name, 'SSB')

    return MultiPath(['MARS', 'VENUS', 'EARTH'])


def test_multipath_is_one_dimensional(core_kernels) -> None:
    """A MultiPath gathers its Paths into a single 1-D Path."""

    assert _three_planets(core_kernels).shape == (3,)


def test_multipath_indexing_by_an_integer_returns_one_path(core_kernels) -> None:
    """An integer index selects a single Path."""

    multi = _three_planets(core_kernels)

    assert multi[0].path_id == 'MARS'
    assert multi[2].path_id == 'EARTH'


def test_multipath_indexing_by_a_range_returns_a_multipath(core_kernels) -> None:
    """A numeric range selects a new MultiPath."""

    sliced = _three_planets(core_kernels)[0:2]

    assert isinstance(sliced, MultiPath)
    assert sliced.shape == (2,)


def test_multipath_event_gathers_every_path(core_kernels) -> None:
    """One event holds the position of every Path at the given time."""

    multi = _three_planets(core_kernels)
    event = multi.event_at_time(Scalar(1.e8))

    assert event.shape == (3,)
    assert event.pos[0] == multi[0].event_at_time(Scalar(1.e8)).pos


def test_multipath_defaults_to_the_ssb(core_kernels) -> None:
    """With no origin given, positions are measured from the SSB."""

    assert _three_planets(core_kernels).origin == Path.SSB.waypoint


def test_multipath_rejects_an_unregistered_id(core_kernels) -> None:
    """A Path ID that has not been registered raises KeyError."""

    with pytest.raises(KeyError):
        MultiPath(['NOT_A_REGISTERED_PATH'])


def test_multipath_pickle(core_kernels) -> None:
    """Pickling restores every gathered Path."""

    multi = _three_planets(core_kernels)
    restored = pickle.loads(pickle.dumps(multi))
    time = Scalar(1.e8)

    assert isinstance(restored, MultiPath)
    assert restored.shape == multi.shape
    assert restored.event_at_time(time).pos == multi.event_at_time(time).pos


def test_multipath_getstate_roundtrip(core_kernels) -> None:
    """The state captured by __getstate__ fully restores the object."""

    multi = _three_planets(core_kernels)
    time = Scalar(1.e8)

    copied = Path.__new__(MultiPath)
    copied.__setstate__(multi.__getstate__())

    assert copied.shape == multi.shape
    assert copied.event_at_time(time).pos == multi.event_at_time(time).pos

##########################################################################################
