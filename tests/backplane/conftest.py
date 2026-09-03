##########################################################################################
# tests/backplane/conftest.py: a synthetic observation for the backplane tests
##########################################################################################

from collections.abc import Iterator
from pathlib import Path as FilePath
from typing import cast

import cspyce
import pytest

from polymath                          import Scalar, Vector3
from oops.backplane                    import Backplane
from oops.body                         import Body
from oops.fov                          import FlatFOV
from oops.frame                        import Frame, TwoVectorFrame
from oops.observation                  import Snapshot
from oops.path                         import Path
from programs.gold_master.test_support  import TEST_SPICE_PREFIX

from tests.conftest import CORE_KERNELS

# A time within the interval the test kernels cover, and an exposure short enough that
# nothing moves appreciably during it.
TIME = 1.e8
TEXP = 10.

# Saturn subtends about 9.1e-5 radians at this time, so a pixel this size puts roughly
# 20 pixels across the disk and leaves the rings filling most of the 40x40 grid.
PIXEL = 9.136e-05 / 20.
SHAPE = (40, 40)


@pytest.fixture(scope='package')
def solar_system() -> Iterator[None]:
    """The bodies of the solar system over the interval the test kernels cover.

    This mirrors the `core_kernels` fixture of the root conftest, but at package scope.
    Building a Backplane is expensive, and the registries must not be torn down and
    rebuilt between these modules: an Event cached by one Backplane holds direct
    references to the waypoints and wayframes that were current when it was built, and
    a later lookup by ID would return their replacements instead.
    """

    for path in cast(list[FilePath], TEST_SPICE_PREFIX.retrieve(CORE_KERNELS)):
        cspyce.furnsh(path)

    Path._reset_caches()
    Frame._reset_caches()
    Body.reset_registry()
    Body._undefine_solar_system()
    Body.define_solar_system('2000-01-01', '2010-01-01')

    yield

    Path._reset_caches()
    Frame._reset_caches()
    Body.reset_registry()


@pytest.fixture(scope='package')
def saturn_obs(solar_system: None) -> Snapshot:
    """A Snapshot of Saturn taken from Earth, pointed at the center of the planet.

    The observation is synthetic: the camera frame is constructed so that its Z-axis
    points straight at Saturn, which puts the planet and a wide span of its rings inside
    the field of view. Nothing here depends on the gold master files.
    """

    earth = Path.as_path('EARTH')
    saturn = Body.lookup('SATURN')
    los = saturn.path.event_at_time(Scalar(TIME)).wrt_path(earth).pos.unit()

    TwoVectorFrame(Frame.J2000, los, 'z', Vector3.XAXIS, 'x',
                   frame_id='TEST_SATURN_CAMERA')
    fov = FlatFOV((PIXEL, PIXEL), SHAPE)

    return Snapshot(('u', 'v'), TIME, TEXP, fov, 'EARTH', 'TEST_SATURN_CAMERA')


@pytest.fixture(scope='package')
def bp(saturn_obs: Snapshot) -> Backplane:
    """A Backplane sampling the center of every pixel of the Saturn observation.

    The Backplane caches every array it evaluates, so one instance for the whole package
    keeps the tests fast. The tests only read from it.
    """

    return Backplane(saturn_obs)

##########################################################################################
