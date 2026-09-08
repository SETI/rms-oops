##########################################################################################
# tests/frame/test_synchronousframe.py
##########################################################################################

import pickle
from typing import cast

import numpy as np
import pytest

from polymath   import Scalar, Vector3
from oops       import Body
from oops.frame import SynchronousFrame
from oops.path  import Path
from oops.path.circlepath import CirclePath


def _dot(a: Vector3, b: Vector3) -> float:
    """The dot product of two shapeless Vector3 objects, as a Python float.

    Parameters:
        a: The first vector.
        b: The second vector.

    Returns:
        float: The dot product.
    """

    return float(cast(Scalar, a.dot(b)))

@pytest.fixture(autouse=True)
def _solar_system():
    Body.reset_registry()
    Body.define_solar_system('2000-01-01', '2020-01-01')

    yield

    Body._undefine_solar_system()
    Body.define_solar_system()

def test_synchronousframe():
    # Path of Saturn relative to Enceladus
    inward = Path.as_path('SATURN').wrt('ENCELADUS')
    synchro = SynchronousFrame('ENCELADUS', 'SATURN')

    time = Scalar(np.arange(1000.) * 86400.)

    # Make sure direction to Saturn is along X-axis
    pos = inward.event_at_time(time).wrt_frame(synchro).pos
    ### self.assertTrue(np.all(pos.vals[:,0] > 0.))
    assert np.max(np.abs(pos.vals[:,1])) < 1.e-10
    assert np.max(np.abs(pos.vals[:,2])) < 1.e-10

    # Make sure this frame and IAU_ENCELADUS are close
    xform = synchro.wrt('IAU_ENCELADUS').transform_at_time(time)

    assert np.max(np.abs(xform.omega.vals[:,0])) < 5.e-8
    assert np.max(np.abs(xform.omega.vals[:,1])) < 5.e-8
    assert np.max(np.abs(xform.omega.vals[:,2])) < 1.e-6

    unit = np.array([[1,0,0],[0,1,0],[0,0,1]])
    assert np.median(np.abs(xform.matrix.vals - unit).ravel()) < 5.e-4
    assert np.median(np.abs(xform.matrix.vals - unit).ravel()) < 0.1


@pytest.mark.parametrize('rate, sense', [(1.e-4, 'prograde'), (-1.e-4, 'retrograde')])
def test_the_axes_follow_the_planet_and_the_angular_momentum(rate, sense) -> None:
    """The x-axis points at the planet and the z-axis along the orbital angular momentum.

    That leaves the y-axis opposite the motion. The z-axis follows the angular momentum
    rather than the reference frame, so it flips for a retrograde orbit.
    """

    orbit = CirclePath(1.e5, 0., rate, 0., 'SSB', frame='J2000',
                       path_id='TEST_ORBIT_' + sense)
    frame = SynchronousFrame(orbit, 'SSB', frame_id='TEST_SYNC_' + sense)

    for time in (0., 3000., 9000.):
        xform = frame.transform_at_time(Scalar(time))
        event = orbit.event_at_time(Scalar(time))

        toward_planet = -event.pos.unit()
        angular_momentum = event.pos.cross(event.vel).unit()
        against_motion = -event.vel.unit()

        # The rows of the matrix are this Frame's axes, in the reference frame
        (x_axis, y_axis, z_axis) = [Vector3(xform.matrix.vals[i]) for i in range(3)]

        assert _dot(x_axis, toward_planet) == pytest.approx(1., abs=1.e-12)
        assert _dot(z_axis, angular_momentum) == pytest.approx(1., abs=1.e-12)
        assert _dot(y_axis, against_motion) == pytest.approx(1., abs=1.e-12)

def test_the_planet_defaults_to_the_origin_of_the_orbit() -> None:
    """With no planet named, the body orbits the origin of its own path."""

    orbit = CirclePath(1.e5, 0., 1.e-4, 0., 'SSB', frame='J2000',
                       path_id='TEST_DEFAULT_ORBIT')

    frame = SynchronousFrame(orbit, frame_id='TEST_DEFAULT_SYNC')

    assert frame._planet_path is Path.as_waypoint('SSB')


def test_a_generated_frame_id_names_the_orbiting_body() -> None:
    """A frame_id of "+" appends "_SYNCHRONOUS" to the ID of the orbiting path."""

    frame = SynchronousFrame('ENCELADUS', 'SATURN', frame_id='+')

    assert frame.frame_id == 'ENCELADUS_SYNCHRONOUS'


def test_a_shaped_planet_path_is_rejected() -> None:
    """The planet has to be one body, because the frame has one z-axis."""

    planet = CirclePath(Scalar([1.e5, 2.e5]), 0., 1.e-4, 0., 'SSB', frame='J2000',
                        path_id='TEST_SHAPED_PLANET')

    with pytest.raises(ValueError, match='requires a shapeless body path'):
        SynchronousFrame('ENCELADUS', planet, frame_id='TEST_SHAPED_SYNC')


def test_the_description_names_the_planet_when_one_was_given() -> None:
    """An explicit planet appears in the description; a defaulted one does not."""

    named = SynchronousFrame('ENCELADUS', 'SATURN', frame_id='TEST_SYNC_NAMED')
    defaulted = SynchronousFrame('ENCELADUS', frame_id='TEST_SYNC_DEFAULTED')

    assert 'SATURN' in named.show(2)
    assert 'SATURN' not in defaulted.show(2)


def test_a_frame_survives_a_round_trip_through_pickle() -> None:
    """Unpickling rebuilds the frame from its two paths, and yields the same rotation."""

    frame = SynchronousFrame('ENCELADUS', 'SATURN', frame_id='TEST_SYNC_PICKLED')
    expected = frame.transform_at_time(Scalar(0.)).matrix

    revived = pickle.loads(pickle.dumps(frame))

    assert revived._orbit_path.waypoint is frame._orbit_path.waypoint
    assert revived._planet_path is frame._planet_path
    assert revived.transform_at_time(Scalar(0.)).matrix == expected

##########################################################################################
