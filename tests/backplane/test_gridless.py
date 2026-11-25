##########################################################################################
# tests/backplane/test_gridless.py: gridless backplanes need no pointing
##########################################################################################

import inspect
from collections.abc import Iterator
from pathlib import Path as FilePath

import cspyce
import numpy as np
import pytest

from polymath                          import Matrix3
from oops.backplane                    import Backplane
from oops.cadence                      import SnapCadence
from oops.frame                        import Cmatrix, SpiceFrame
from oops.observation                  import Snapshot

from tests.backplane.conftest import TEXP, TIME


class _NoPointing(RuntimeError):
    """Raised by the camera frame of an observation that has no pointing."""


class _UnpointedFrame(Cmatrix):
    """A camera frame with no pointing, like a C-kernel frame outside its coverage."""

    def transform_at_time(self, time, *, quick=False):
        raise _NoPointing('no pointing')


def _is_gridless(name: str) -> bool:
    """True if the named backplane method declares itself a gridless backplane."""

    doc = inspect.getdoc(getattr(Backplane, name)) or ''
    return 'registered as a gridless backplane' in ' '.join(doc.split())


# Every gridless backplane method, as the methods themselves declare it.
GRIDLESS = sorted(name for name in Backplane._CALLABLES if _is_gridless(name))

# The gridless backplanes that need the pointing after all. A body's center has field of
# view (u,v) coordinates only where the camera points.
NEEDS_POINTING = {'center_coordinate'}

# The event key each method is called with: its rings for a ring method, an orbiting moon
# for orbit_longitude, and Saturn itself otherwise.
def _event_key(name: str) -> str:
    if name.startswith('ring_'):
        return 'SATURN:RING'
    if name == 'orbit_longitude':
        return 'MIMAS'
    return 'SATURN'


# Options worth checking beyond each method's defaults.
EXTRA_OPTIONS = {
    'center_resolution': [('v',)],
    'orbit_longitude': [(reference,) for reference in ('aries', 'node', 'oha', 'sun',
                                                       'sha')],
    'sub_observer_latitude': [('graphic',)],
    'sub_solar_latitude': [('graphic',)],
}

GRIDLESS_KEYS = [(name, _event_key(name)) + options
                 for name in GRIDLESS if name not in NEEDS_POINTING
                 for options in [()] + EXTRA_OPTIONS.get(name, [])]


# A text frame kernel defining a C-kernel camera frame, for which no C kernel is ever
# loaded: the state of a real camera whose C kernels are missing.
_CK_FRAME_KERNEL = """KPL/FK
\\begindata
FRAME_TEST_CK_CAMERA     = -999001
FRAME_-999001_NAME       = 'TEST_CK_CAMERA'
FRAME_-999001_CLASS      = 3
FRAME_-999001_CLASS_ID   = -999001
FRAME_-999001_CENTER     = 399
CK_-999001_SCLK          = -999
\\begintext
"""


def test_the_gridless_backplanes_are_found() -> None:
    """The methods that declare themselves gridless are the ones expected."""

    assert len(GRIDLESS) == 23
    assert NEEDS_POINTING <= set(GRIDLESS)
    assert 'orbit_longitude' in GRIDLESS
    assert 'pole_position_angle' in GRIDLESS


@pytest.fixture(scope='module')
def pointed(saturn_obs: Snapshot) -> Backplane:
    """A backplane of the synthetic Saturn observation, which has pointing."""

    return Backplane(saturn_obs)


@pytest.fixture(scope='module')
def unpointed(saturn_obs: Snapshot) -> Backplane:
    """A backplane of the same observation, but with a camera frame that has no pointing."""

    obs = Snapshot(('u', 'v'), cadence=SnapCadence(TIME, TEXP), fov=saturn_obs.fov,
                   path='EARTH', frame=_UnpointedFrame(Matrix3.IDENTITY))
    return Backplane(obs)


@pytest.fixture(scope='module')
def ck_unpointed(saturn_obs: Snapshot,
                 tmp_path_factory: pytest.TempPathFactory) -> Iterator[Backplane]:
    """A backplane of the same observation, with a C-kernel frame but no C kernel."""

    kernel = FilePath(tmp_path_factory.mktemp('ck_frame')) / 'test_ck_camera.tf'
    kernel.write_text(_CK_FRAME_KERNEL)
    cspyce.furnsh(str(kernel))
    try:
        obs = Snapshot(('u', 'v'), cadence=SnapCadence(TIME, TEXP), fov=saturn_obs.fov,
                       path='EARTH', frame=SpiceFrame('TEST_CK_CAMERA'))
        yield Backplane(obs)
    finally:
        cspyce.unload(str(kernel))


@pytest.mark.parametrize('name', sorted(NEEDS_POINTING))
def test_the_unpointed_frame_stands_in_for_missing_pointing(name: str,
                                                            unpointed: Backplane) -> None:
    """A backplane that needs the pointing fails, so the frame is really consulted."""

    with pytest.raises(_NoPointing):
        unpointed.evaluate((name, _event_key(name)))


@pytest.mark.parametrize('key', GRIDLESS_KEYS, ids=lambda key: ':'.join(key))
def test_a_gridless_backplane_needs_no_pointing(key: tuple, pointed: Backplane,
                                                unpointed: Backplane) -> None:
    """A gridless backplane is computed without pointing, with the same value."""

    expected = pointed.evaluate(key)
    result = unpointed.evaluate(key)
    assert np.all(result.vals == expected.vals)
    assert np.all(result.mask == expected.mask)


@pytest.mark.parametrize('name', sorted(NEEDS_POINTING))
def test_missing_c_kernels_raise_the_raw_spice_error(name: str,
                                                     ck_unpointed: Backplane) -> None:
    """Without C kernels, a backplane that needs pointing raises cspyce's own error.

    Callers such as rms-metadata-tools and SpinDoctor recognize missing pointing by this
    exact form, a RuntimeError naming SPICE(NOFRAMECONNECT), so it must reach them
    unwrapped.
    """

    with pytest.raises(RuntimeError, match=r'SPICE\(NOFRAMECONNECT\)') as error:
        ck_unpointed.evaluate((name, _event_key(name)))
    assert type(error.value) is RuntimeError


@pytest.mark.parametrize('key', GRIDLESS_KEYS, ids=lambda key: ':'.join(key))
def test_a_gridless_backplane_needs_no_c_kernel(key: tuple, pointed: Backplane,
                                                ck_unpointed: Backplane) -> None:
    """With a real C-kernel frame and no C kernel, gridless values are unchanged."""

    expected = pointed.evaluate(key)
    result = ck_unpointed.evaluate(key)
    assert np.all(result.vals == expected.vals)
    assert np.all(result.mask == expected.mask)
