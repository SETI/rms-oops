##########################################################################################
# tests/frame/test_synchronousframe.py
##########################################################################################

import numpy as np
import pytest

from polymath   import Scalar
from oops       import Body
from oops.frame import SynchronousFrame


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
##########################################################################################
