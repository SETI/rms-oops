##########################################################################################
# tests/frame/test_synchronousframe.py
##########################################################################################

import numpy as np
import pytest

from polymath   import Scalar, Vector3
from oops       import Body
from oops.frame import SynchronousFrame
from oops.path.circlepath import CirclePath


@pytest.fixture(autouse=True)
def _solar_system():
    Body.reset_registry()
    Body.define_solar_system('2000-01-01', '2020-01-01')

    yield

    Body._undefine_solar_system()
    Body.define_solar_system()

def test_synchronousframe():
    from oops.path import Path

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

        assert float(x_axis.dot(toward_planet).vals) == pytest.approx(1., abs=1.e-12)
        assert float(z_axis.dot(angular_momentum).vals) == pytest.approx(1., abs=1.e-12)
        assert float(y_axis.dot(against_motion).vals) == pytest.approx(1., abs=1.e-12)

##########################################################################################
