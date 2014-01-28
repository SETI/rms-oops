################################################################################
# oops_/frame/ringframe.py: Subclass RingFrame of class Frame
#
# 2/8/12 Modified (MRS) - Update for consistent style.
# 1/4/12 MRS: Added attribute "node" and function node_at_time().
################################################################################

import numpy as np

from oops_.frame.frame_ import Frame
from oops_.array.all import *
from oops_.transform import Transform

import oops_.registry as registry

TWOPI  = np.pi * 2.

class RingFrame(Frame):
    """RingFrame is a Frame subclass describing a non-rotating frame centered on
    the Z-axis of another frame, but oriented with the X-axis fixed along the
    ascending node of the equator within the reference frame.
    """

    def __init__(self, frame, epoch=None, id=None, retrograde=False):
        """Constructor for a RingFrame Frame.

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

            retrograde  True to flip the sign of the Z-axis. Necessary for
                        retrograde systems like Uranus.
        """

        frame = registry.as_frame(frame)

        self.frame_id = frame.frame_id
        self.reference_id = frame.reference_id

        # The frame might not be exactly inertial due to polar precession, but
        # it is good enough
        self.origin_id = None

        self.shape = frame.shape

        self.planet_frame = frame
        self.epoch = epoch
        self.retrograde = retrograde

        # For a fixed epoch, derive the inertial tranform now
        self.transform = None
        self.node = None

        if epoch is not None:
            self.transform = self.transform_at_time(self.epoch)
            self.node = self.node_at_time(self.epoch)

        # Fill in the frame name
        if id is None:
            if self.epoch is None:
                self.frame_id = frame.frame_id + "_DESPUN"
            else:
                self.frame_id = frame.frame_id + "_INERTIAL"
        else:
            self.frame_id = id

        self.reregister()

########################################

    def transform_at_time(self, time, quick=None):
        """Returns the Transform to the given Frame at a specified Scalar of
        times."""

        # For a fixed epoch, return the fixed transform
        if self.transform is not None:
            return self.transform

        # Otherwise, calculate it for the current time
        xform = self.planet_frame.transform_at_time(time, quick)
        matrix = xform.matrix.vals

        # The bottom row of the matrix is the z-axis of the frame
        if self.retrograde:
            matrix[...,2,:] = -matrix[...,2,:]

        z_axis = matrix[...,2,:]

        # Replace the X-axis of the matrix using (0,0,1) cross Z-axis
        #   (0,0,1) x (a,b,c) = (-b,a,0)
        # with the norm of the vector scaled to unity.

        norm = np.sqrt(z_axis[...,0]**2 + z_axis[...,1]**2)
        matrix[...,0,0] = -z_axis[...,1] / norm
        matrix[...,0,1] =  z_axis[...,0] / norm
        matrix[...,0,2] =  0.
        
        # Replace the Y-axis of the matrix using Y = Z cross X
        matrix[...,1,:] = utils.cross3d(z_axis, matrix[...,0,:])

        return Transform(Matrix3(matrix, xform.matrix.mask), Vector3.ZERO,
                         self.frame_id, self.reference_id, None)

########################################

    def node_at_time(self, time, quick=None):
        """Returns the rotation angle from the X-axis of the original reference
        frame to the ascending node of the ring plane, which serves as the
        X-axis of this frame."""

        # For a fixed epoch, return the fixed node
        if self.transform is not None:
            return self.transform

        # Otherwise, calculate it for the current time
        xform = self.planet_frame.transform_at_time(time, quick)
        matrix = xform.matrix.vals

        # The bottom row of the matrix is the pole
        z_axis = matrix[...,2,:]

        if self.retrograde:
            z_axis = -z_axis

        # The ascending node is 90 degrees ahead of the pole
        angle = np.arctan2(z_axis[...,0], -z_axis[...,1])

        return Scalar(angle % TWOPI)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_RingFrame(unittest.TestCase):

    def runTest(self):

        # Imports are here to reduce conflicts
        from oops_.frame.spiceframe import SpiceFrame
        from oops_.path.spicepath import SpicePath
        from oops_.event import Event

        registry.initialize()

        center = SpicePath("MARS", "SSB")
        planet = SpiceFrame("IAU_MARS", "J2000")
        rings  = RingFrame(planet)
        self.assertEqual(registry.frame_lookup("IAU_MARS"), planet)
        self.assertEqual(registry.frame_lookup("IAU_MARS_DESPUN"), rings)

        time = Scalar(np.random.rand(3,4,2) * 1.e8)
        posvel = np.random.rand(3,4,2,6)
        event = Event(time, posvel[...,0:3], posvel[...,3:6], "SSB", "J2000")
        rotated = event.wrt_frame("IAU_MARS")
        fixed   = event.wrt_frame("IAU_MARS_DESPUN")

        # Confirm Z axis is tied to planet's pole
        diff = rotated.pos.as_scalar(2) - fixed.pos.as_scalar(2)
        self.assertTrue(np.all(np.abs(diff.vals < 1.e-14)))

        # Confirm X-axis is always in the J2000 equator
        xaxis = Event(time, Vector3.XAXIS,
                            Vector3.ZERO, "SSB", rings.frame_id)
        test = xaxis.wrt_frame("J2000")
        self.assertTrue(np.all(np.abs(test.pos.as_scalar(2).vals < 1.e-14)))

        # Confirm it's at the ascending node
        xaxis = Event(time, (1,1.e-13,0), Vector3.ZERO, "SSB", rings.frame_id)
        test = xaxis.wrt_frame("J2000")
        self.assertTrue(np.all(test.pos.as_scalar(1).vals > 0.))

        # Check that pole wanders when epoch is fixed
        rings2 = RingFrame(planet, 0.)
        self.assertEqual(registry.frame_lookup("IAU_MARS_INERTIAL"), rings2)
        inertial = event.wrt_frame("IAU_MARS_INERTIAL")

        diff = rotated.pos.as_scalar(2) - inertial.pos.as_scalar(2)
        self.assertTrue(np.all(np.abs(diff.vals) < 1.e-4))
        self.assertTrue(np.mean(np.abs(diff.vals) > 1.e-8))

        registry.initialize()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
