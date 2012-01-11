import numpy as np
import unittest

import oops

################################################################################
# RotatedFrame
################################################################################

class MatrixFrame(oops.Frame):
    """A MatrixFrame is a frame defined by a constant rotation matrix.
    """

    def __init__(self, matrix, reference, id):
        """Constructor for a MatrixFrame.

        Input:
            matrix      a Matrix3 object.
            reference   the ID or frame relative to which this frame is defind.
            id          the ID under which the frame will be registered.

        Note that the matrix and the reference frame need not have the same
        shape, as long as they can broadcasted to the same shape.
        """

        matrix = oops.Matrix3.as_matrix3(matrix)

        reference = oops.as_frame(reference)
        self.reference_id = reference.frame_id
        self.origin_id = reference.origin_id

        self.shape = oops.Array.broadcast_shape((matrix, reference))

        self.frame_id = id

        self.register() # We have to register it before we can construct the
                        # Transform

        self.transform = oops.Transform(matrix, (0,0,0),
                                        self.frame_id,
                                        self.reference_id)

########################################

    def transform_at_time(self, time):
        """Returns the Transform to the given Frame at a specified Scalar of
        times."""

        return self.transform

################################################################################
# UNIT TESTS
################################################################################

class Test_MatrixFrame(unittest.TestCase):

    def runTest(self):
        oops.Frame.initialize_registry()
        oops.Path.initialize_registry()

        ignore = oops.SpicePath("MARS", "SSB")
        mars = oops.SpiceFrame("IAU_MARS", "J2000")

        # Define a version of the IAU Mars frame always rotated by 180 degrees
        # around the Z-axis
        mars180 = MatrixFrame([[-1,0,0],[0,-1,0],[0,0,1]], "IAU_MARS",
                                                           "MARS180")

        time = oops.Scalar(np.random.rand(100) * 1.e8)
        posvel = np.random.rand(100,6)
        event = oops.Event(time, posvel[...,0:3],
                                 posvel[...,3:6], "MARS", "J2000")

        wrt_mars = event.wrt_frame("IAU_MARS")
        wrt_mars180 = event.wrt_frame("MARS180")

        # Confirm that the components are related as expected
        self.assertTrue(np.all(wrt_mars.pos.x == -wrt_mars180.pos.x))
        self.assertTrue(np.all(wrt_mars.pos.y == -wrt_mars180.pos.y))
        self.assertTrue(np.all(wrt_mars.pos.z ==  wrt_mars180.pos.z))

        self.assertTrue(np.all(wrt_mars.vel.x == -wrt_mars180.vel.x))
        self.assertTrue(np.all(wrt_mars.vel.y == -wrt_mars180.vel.y))
        self.assertTrue(np.all(wrt_mars.vel.z ==  wrt_mars180.vel.z))

        # Define a version of the IAU Mars frame containing four 90-degree
        # rotations
        matrices = []
        for (cos,sin) in ((1,0), (0,1), (-1,0), (0,-1)):
            matrices.append([[cos,sin,0],[-sin,cos,0],[0,0,1]])

        mars90s = MatrixFrame(matrices, "IAU_MARS", "MARS90S")

        time = oops.Scalar(np.random.rand(100,1) * 1.e8)
        posvel = np.random.rand(100,1,6)
        event = oops.Event(time, posvel[...,0:3],
                                 posvel[...,3:6], "MARS", "J2000")

        wrt_mars = event.wrt_frame("IAU_MARS")
        wrt_mars90s = event.wrt_frame("MARS90S")

        self.assertEqual(wrt_mars.shape, [100,1])
        self.assertEqual(wrt_mars90s.shape, [100,4])

        # Confirm that the components are related as expected
        self.assertTrue(wrt_mars.pos[:,0,:] == wrt_mars90s.pos[:,0,:])
        self.assertTrue(wrt_mars.vel[:,0,:] == wrt_mars90s.vel[:,0,:])

        self.assertTrue(np.all(wrt_mars.pos.x[:,0] == -wrt_mars90s.pos.x[:,2]))
        self.assertTrue(np.all(wrt_mars.pos.y[:,0] == -wrt_mars90s.pos.y[:,2]))
        self.assertTrue(np.all(wrt_mars.pos.z[:,0] ==  wrt_mars90s.pos.z[:,2]))
        self.assertTrue(np.all(wrt_mars.vel.x[:,0] == -wrt_mars90s.vel.x[:,2]))
        self.assertTrue(np.all(wrt_mars.vel.y[:,0] == -wrt_mars90s.vel.y[:,2]))
        self.assertTrue(np.all(wrt_mars.vel.z[:,0] ==  wrt_mars90s.vel.z[:,2]))

        self.assertTrue(np.all(wrt_mars.pos.x[:,0] == -wrt_mars90s.pos.y[:,1]))
        self.assertTrue(np.all(wrt_mars.pos.y[:,0] ==  wrt_mars90s.pos.x[:,1]))
        self.assertTrue(np.all(wrt_mars.pos.z[:,0] ==  wrt_mars90s.pos.z[:,1]))
        self.assertTrue(np.all(wrt_mars.vel.x[:,0] == -wrt_mars90s.vel.y[:,1]))
        self.assertTrue(np.all(wrt_mars.vel.y[:,0] ==  wrt_mars90s.vel.x[:,1]))
        self.assertTrue(np.all(wrt_mars.vel.z[:,0] ==  wrt_mars90s.vel.z[:,1]))

        self.assertTrue(np.all(wrt_mars.pos.x[:,0] ==  wrt_mars90s.pos.y[:,3]))
        self.assertTrue(np.all(wrt_mars.pos.y[:,0] == -wrt_mars90s.pos.x[:,3]))
        self.assertTrue(np.all(wrt_mars.pos.z[:,0] ==  wrt_mars90s.pos.z[:,3]))
        self.assertTrue(np.all(wrt_mars.vel.x[:,0] ==  wrt_mars90s.vel.y[:,3]))
        self.assertTrue(np.all(wrt_mars.vel.y[:,0] == -wrt_mars90s.vel.x[:,3]))
        self.assertTrue(np.all(wrt_mars.vel.z[:,0] ==  wrt_mars90s.vel.z[:,3]))

        oops.Frame.initialize_registry()
        oops.Path.initialize_registry()

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
