##########################################################################################
# tests/backplane/test_gridless.py: gridless backplanes need no pointing
##########################################################################################

import numpy as np
import pytest

from polymath                          import Matrix3
from oops.backplane                    import Backplane
from oops.cadence                      import SnapCadence
from oops.frame                        import Cmatrix
from oops.observation                  import Snapshot

from tests.backplane.conftest import TEXP, TIME


class _NoPointing(RuntimeError):
    """Raised by the camera frame of an observation that has no pointing."""


class _UnpointedFrame(Cmatrix):
    """A camera frame with no pointing, like a C-kernel frame outside its coverage."""

    def transform_at_time(self, time, *, quick=False):
        raise _NoPointing('no pointing')


# The gridless backplanes of a body and of its rings: each depends only on the paths of
# the observer, the body and the Sun, not on where the camera points.
GRIDLESS_KEYS = [
    ('center_distance', 'SATURN'),
    ('center_light_time', 'SATURN'),
    ('center_time', 'SATURN'),
    ('center_incidence_angle', 'SATURN'),
    ('center_emission_angle', 'SATURN'),
    ('center_phase_angle', 'SATURN'),
    ('center_scattering_angle', 'SATURN'),
    ('center_resolution', 'SATURN', 'u'),
    ('center_resolution', 'SATURN', 'v'),
    ('center_right_ascension', 'SATURN'),
    ('center_declination', 'SATURN'),
    ('body_diameter_in_pixels', 'SATURN'),
    ('pole_clock_angle', 'SATURN'),
    ('pole_position_angle', 'SATURN'),
    ('sub_observer_latitude', 'SATURN'),
    ('sub_observer_longitude', 'SATURN'),
    ('sub_solar_latitude', 'SATURN'),
    ('sub_solar_longitude', 'SATURN'),
    ('ring_sub_observer_longitude', 'SATURN:RING'),
    ('ring_sub_solar_longitude', 'SATURN:RING'),
    ('ring_center_incidence_angle', 'SATURN:RING'),
    ('ring_center_emission_angle', 'SATURN:RING'),
]


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


def test_the_unpointed_frame_stands_in_for_missing_pointing(unpointed: Backplane) -> None:
    """A backplane that needs the pointing fails, so the frame is really consulted."""

    with pytest.raises(_NoPointing):
        unpointed.evaluate(('center_coordinate', 'SATURN', 'u'))


@pytest.mark.parametrize('key', GRIDLESS_KEYS, ids=lambda key: ':'.join(key))
def test_a_gridless_backplane_needs_no_pointing(key: tuple, pointed: Backplane,
                                                unpointed: Backplane) -> None:
    """A gridless backplane is computed without pointing, with the same value."""

    expected = pointed.evaluate(key)
    result = unpointed.evaluate(key)
    assert np.all(result.vals == expected.vals)
    assert np.all(result.mask == expected.mask)
