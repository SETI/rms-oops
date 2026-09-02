##########################################################################################
# tests/path/test_path.py
##########################################################################################

import numpy as np
import pickle
import pytest

import cspyce

from oops.config import QUICK
from oops.frame  import Frame, SpiceFrame
from oops.path   import (Path, LinkedPath, ReversedPath, RelativePath,
                         RotatedPath, QuickPath, LinearPath, SpicePath)
from oops.unittester_support import TEST_SPICE_PREFIX


@pytest.fixture(autouse=True)
def _ephemeris_kernel():
    cspyce.furnsh(TEST_SPICE_PREFIX.retrieve('de421.bsp'))
    Path._reset_caches()
    Frame._reset_caches()

def test_path():
    Path._USE_QUICKPATHS = False

    assert Path._PATH_REGISTRY['SSB'] == Path.SSB

    # LinkedPath tests
    _ = SpicePath('SUN', 'SSB')
    earth = SpicePath('EARTH', 'SUN')

    moon = SpicePath('MOON', 'EARTH')
    linked = LinkedPath(moon, earth)

    direct = SpicePath('MOON', 'SUN')

    times = np.arange(-3.e8, 3.01e8, 0.5e7)

    direct_event = direct.event_at_time(times)
    linked_event = linked.event_at_time(times)

    eps = 1.e-6
    assert ((linked_event.pos - direct_event.pos).norm() <= eps).all()
    assert ((linked_event.vel - direct_event.vel).norm() <= eps).all()

    # RelativePath
    relative = RelativePath(linked, SpicePath('MARS', 'SUN'))
    direct = SpicePath('MOON', 'MARS')

    direct_event = direct.event_at_time(times)
    relative_event = relative.event_at_time(times)

    eps = 1.e-6
    assert ((relative_event.pos - direct_event.pos).norm() <= eps).all()
    assert ((relative_event.vel - direct_event.vel).norm() <= eps).all()

    # ReversedPath
    reversed = ReversedPath(relative)
    direct = SpicePath('MARS', 'MOON')

    direct_event = direct.event_at_time(times)
    reversed_event = reversed.event_at_time(times)

    eps = 1.e-6
    assert ((reversed_event.pos - direct_event.pos).norm() <= eps).all()
    assert ((reversed_event.vel - direct_event.vel).norm() <= eps).all()

    # RotatedPath
    rotated = RotatedPath(reversed, SpiceFrame('B1950'))
    direct = SpicePath('MARS', 'MOON', 'B1950')

    direct_event = direct.event_at_time(times)
    rotated_event = rotated.event_at_time(times)

    eps = 1.e-6
    assert ((rotated_event.pos - direct_event.pos).norm() <= eps).all()
    assert ((rotated_event.vel - direct_event.vel).norm() <= eps).all()

    # QuickPath tests
    moon = SpicePath('MOON', 'EARTH')
    quick = QuickPath(moon, -5., 5., QUICK.dictionary)

    # Perfect precision is impossible
    try:
        quick = QuickPath(moon, 0., 100.,
                          dict(QUICK.dictionary, **{'path_self_check':0.}))
        assert False, 'No ValueError raised for PRECISION = 0.'
    except ValueError:
        pass

    # Timing tests...
    test = np.zeros(3000000)
    # _ = moon.event_at_time(test, quick=False)       # takes about 15 sec
    _ = quick.event_at_time(test)                   # takes maybe 2 sec

    Path._reset_caches()
    Frame._reset_caches()

    ######################################################################################
    # Test unregistered paths
    ######################################################################################

    ssb = Path.as_waypoint('SSB')

    slider1 = LinearPath(([3,0,0],[0,3,0]), 0., ssb)
    assert not slider1.is_registered

    event = slider1.event_at_time(1.)
    assert event.pos == (3,3,0)
    assert event.vel == (0,3,0)

    slider2 = LinearPath(([-2,0,0],[0,0,-2]), 0., slider1)
    assert not slider2.is_registered

    event = slider2.event_at_time(1.)
    assert event.pos == (-2,0,-2)
    assert event.vel == (0,0,-2)

    slider3 = LinearPath(([-1,0,0],[0,-3,2]), 0., slider2)
    assert not slider3.is_registered

    event = slider3.event_at_time(1.)
    assert event.pos == (-1,-3,2)
    assert event.vel == ( 0,-3,2)

    # Link unregistered frame to registered frame
    static = slider3.wrt(ssb)

    event = static.event_at_time(1.)
    assert event.pos == (0,0,0)
    assert event.vel == (0,0,0)

    # Link registered frame to unregistered frame
    static = ssb.wrt(slider3)

    event = static.event_at_time(1.)
    assert event.pos == (0,0,0)
    assert event.vel == (0,0,0)


def test_ssb_path_is_a_pickled_singleton():
    restored = pickle.loads(pickle.dumps(Path.SSB))

    assert restored is Path.SSB
    assert restored.path_id == 'SSB'

##########################################################################################
