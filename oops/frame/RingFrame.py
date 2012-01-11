import numpy as np
import unittest

import oops

################################################################################
# RingFrame
################################################################################

class RingFrame(oops.Frame):
    """A RingFrame is a non-rotating Frame object centered on the Z-axis of
    another Frame, but oriented with the X-axis fixed along the ascending node
    of the equator within the reference frame.
    """

    def __init__(self, frame, epoch=None, id=None):
        """Constructor for a RingFrame.

        Input:
            frame       a frame describing the central planet of the ring plane
                        relative to J2000.

            epoch       the time TDB at which the frame is to be evaluated. If
                        this is specified, then the frame will be precisely
                        inertial, based on the orientation of the pole at the
                        specified epoch. If it is unspecified, then the frame
                        could wobble slowly due to precession of the planet's
                        pole.

            id          the ID under which the frame will be registered. By
                        default, it is the planet frame's name with "_DESPUN"
                        added as a suffix when epoch is None, or "_INERTIAL"
                        added as a suffix when an epoch is specified.
        """

        frame = oops.as_frame(frame)

        self.frame_id = frame.frame_id
        self.reference_id = frame.reference_id
        self.origin_id = frame.origin_id
        self.shape = frame.shape

        self.planet_frame = frame
        self.epoch = epoch
        self.transform = None

        # For a fixed epoch, derive the inertial tranform now
        if epoch is not None:
            self.transform = self.transform_at_time(self.epoch)

        # Fill in the frame name
        if id is None:
            if self.epoch is None:
                self.frame_id = frame.frame_id + "_DESPUN"
            else:
                self.frame_id = frame.frame_id + "_INERTIAL"
        else:
            self.frame_id = id

        self.register()

########################################

    def transform_at_time(self, time):
        """Returns the Transform to the given Frame at a specified Scalar of
        times."""

        # For a fixed epoch, return the fixed transform
        if self.transform is not None:
            return self.transform

        # Otherwise, calculate it for the current time
        matrix = self.planet_frame.transform_at_time(time).matrix.vals

        # Note matrix[...,2,:] is already the desired Z-axis of the frame
        z_axis = matrix[...,2,:]

        # Replace the X-axis of the matrix using (0,0,1) cross Z-axis
        #   (0,0,1) X (a,b,c) = (-b,a,0)
        # with the norm of the vector scaled to unity.

        norm = np.sqrt(z_axis[...,0]**2 + z_axis[...,1]**2)
        matrix[...,0,0] = -z_axis[...,1] / norm
        matrix[...,0,1] =  z_axis[...,0] / norm
        matrix[...,0,2] =  0.
        
        # Replace the Y-axis of the matrix using Y = Z cross X
        matrix[...,1,:] = oops.utils.cross3d(z_axis, matrix[...,0,:])

        return oops.Transform(oops.Matrix3(matrix), oops.Vector3([0,0,0]),
                              self.frame_id, self.reference_id)

################################################################################
# UNIT TESTS
################################################################################

class Test_RingFrame(unittest.TestCase):

    def runTest(self):
        oops.Frame.initialize_registry()
        oops.Path.initialize_registry()

        center = oops.SpicePath("MARS", "SSB")
        planet = oops.SpiceFrame("IAU_MARS", "J2000")
        rings  = RingFrame(planet)
        self.assertEqual(oops.Frame.lookup("IAU_MARS"), planet)
        self.assertEqual(oops.Frame.lookup("IAU_MARS_DESPUN"), rings)

        time = oops.Scalar(np.random.rand(3,4,2) * 1.e8)
        posvel = np.random.rand(3,4,2,6)
        event = oops.Event(time, posvel[...,0:3],
                                 posvel[...,3:6], "SSB", "J2000")
        rotated = event.wrt_frame("IAU_MARS")
        fixed   = event.wrt_frame("IAU_MARS_DESPUN")

        # Confirm Z axis is tied to Saturn's pole
        diff = rotated.pos.as_scalar(2) - fixed.pos.as_scalar(2)
        self.assertTrue(np.all(np.abs(diff.vals < 1.e-14)))

        # Confirm X-axis is always in the J2000 equator
        xaxis = oops.Event(time, oops.Vector3([1,0,0]),
                                 oops.Vector3([0,0,0]), "SSB", rings)
        test = xaxis.wrt_frame("J2000")
        self.assertTrue(np.all(np.abs(test.pos.as_scalar(2).vals < 1.e-14)))

        # Confirm it's at the ascending node
        xaxis = oops.Event(time, oops.Vector3([1,1.e-13,0]),
                                 oops.Vector3([0,0,0]),
                                 "SSB", rings)
        test = xaxis.wrt_frame("J2000")
        self.assertTrue(np.all(test.pos.as_scalar(1).vals > 0.))

        # Check that pole wanders when epoch is fixed
        rings2 = RingFrame(planet, 0.)
        self.assertEqual(oops.Frame.lookup("IAU_MARS_INERTIAL"), rings2)
        inertial = event.wrt_frame("IAU_MARS_INERTIAL")

        diff = rotated.pos.as_scalar(2) - inertial.pos.as_scalar(2)
        self.assertTrue(np.all(np.abs(diff.vals) < 1.e-4))
        self.assertTrue(np.mean(np.abs(diff.vals) > 1.e-8))

        oops.Frame.initialize_registry()
        oops.Path.initialize_registry()

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
