################################################################################
# oops/frame/synchronousframe.py: Subclass SynchronousFrame of class Frame
################################################################################

import numpy as np
from polymath import Qube, Boolean, Scalar, Pair, Vector
from polymath import Vector3, Matrix3, Quaternion

from .           import Frame
from ..path      import Path
from ..transform import Transform

class SynchronousFrame(Frame):
    """A Frame subclass describing a a body that always keeps the x-axis pointed
    toward a central planet and the y-axis in the negative direction of motion.
    """

    FRAME_IDS = {}  # frame_id to use if a frame already exists upon un-pickling

    #===========================================================================
    def __init__(self, body_path, planet_path, frame_id=None, unpickled=False):
        """Constructor for a SynchronousFrame.

        Input:
            body_path       the path or path ID followed by the body.
            planet_path     the path or path ID followed by the central planet.
            frame_id        the ID to use; None to leave the frame unregistered.
            unpickled       True if this frame has been read from a pickle file.
        """

        self.body_path = Path.as_path(body_path)
        self.planet_path = Path.as_path(planet_path)
        self.path = Path.wrt(self.planet_path, self.body_path)

        assert self.planet_path.shape == ()

        self.frame_id  = frame_id
        self.reference = Frame.as_wayframe(self.planet_path.frame)
        self.origin    = self.planet_path.origin
        self.shape     = Qube.broadcasted_shape(self.body_path,
                                                self.planet_path)
        self.keys      = set()

        # Update wayframe and frame_id; register if not temporary
        self.register(unpickled=unpickled)

        # Save in internal dict for name lookup upon serialization
        if (not unpickled and self.shape == ()
            and self.frame_id in Frame.WAYFRAME_REGISTRY):
                key = (self.body_path.path_id, self.planet_path.path_id)
                SynchronousFrame.FRAME_IDS[key] = self.frame_id

    # Unpickled frames will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.body_path, self.planet_path, self.shape)

    def __setstate__(self, state):
        # If this frame matches a pre-existing frame, re-use its ID
        (body_path, planet_path, shape) = state
        if shape == ():
            key = (body_path.path_id, planet_path.path_id)
            frame_id = SynchronousFrame.FRAME_IDS.get(key, None)
        else:
            frame_id = None

        self.__init__(body_path, planet_path, frame_id=frame_id,
                      unpickled=True)

    #===========================================================================
    def transform_at_time(self, time, quick=False):
        """The Transform into the this Frame at a Scalar of times."""

        event = self.path.event_at_time(time, quick=quick)
        matrix = Matrix3.twovec(event.pos, 0, event.vel, 1)
        omega = event.pos.cross(event.vel) / event.pos.dot(event.pos)

        return Transform(matrix, omega, self.frame_id, self.reference,
                                        self.body_path)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_SynchronousFrame(unittest.TestCase):

    def setUp(self):
        from ..body import Body

        Body.reset_registry()
        Body.define_solar_system('2000-01-01', '2020-01-01')

    def tearDown(self):
        pass

    def runTest(self):
        from ..path import Path

        # Path of Saturn relative to Enceladus
        inward = Path.as_path('SATURN').wrt('ENCELADUS')
        synchro = SynchronousFrame('ENCELADUS', 'SATURN', frame_id='SYNCHRO')

        time = Scalar(np.arange(1000.) * 86400.)

        # Make sure direction to Saturn is along X-axis
        pos = inward.event_at_time(time).wrt_frame(synchro).pos
        self.assertTrue(np.all(pos.values[:,0] > 0.))
        self.assertTrue(np.max(np.abs(pos.values[:,1])) < 1.e-10)
        self.assertTrue(np.max(np.abs(pos.values[:,2])) < 1.e-10)

        # Make sure this frame and IAU_ENCELADUS are close
        xform = synchro.wrt('IAU_ENCELADUS').transform_at_time(time)

        self.assertTrue(np.max(np.abs(xform.omega.values[:,0])) < 5.e-8)
        self.assertTrue(np.max(np.abs(xform.omega.values[:,1])) < 5.e-8)
        self.assertTrue(np.max(np.abs(xform.omega.values[:,2])) < 1.e-6)

        unit = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.assertTrue(np.median(np.abs(xform.matrix.values - unit).ravel())
                        < 5.e-4)
        self.assertTrue(np.median(np.abs(xform.matrix.values - unit).ravel())
                        < 0.1)

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
