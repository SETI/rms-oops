################################################################################
# oops/frame_/ringframe.py: Subclass RingFrame of class Frame
################################################################################

import numpy as np
from polymath import *

from oops.frame_.frame import Frame
from oops.transform    import Transform
from oops.constants    import *
import oops.utils    as utils

class RingFrame(Frame):
    """RingFrame is a Frame subclass describing a non-rotating frame centered on
    the Z-axis of another frame, but oriented with the X-axis fixed along the
    ascending node of the equator within the reference frame.
    """

    def __init__(self, frame, epoch=None, retrograde=False, id='+'):
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

            retrograde  True to flip the sign of the Z-axis. Necessary for
                        retrograde systems like Uranus.

            id          the ID under which the frame will be registered. None to
                        leave the frame unregistered. If the value is "+", then
                        the registered name is the planet frame's name with the
                        suffix "_DESPUN" if epoch is None, or "_INERTIAL" if an
                        epoch is specified.
        """

        self.planet_frame = Frame.as_frame(frame).wrt(Frame.J2000)
        self.reference    = self.planet_frame.reference
        self.epoch = epoch
        self.retrograde = retrograde
        self.shape = frame.shape
        self.keys = set()

        # The frame might not be exactly inertial due to polar precession, but
        # it is good enough
        self.origin = None

        # Fill in the frame ID
        if id is None:
            self.frame_id = Frame.temporary_frame_id()
        elif id == '+':
            if self.epoch is None:
                self.frame_id = self.planet_frame.frame_id + "_DESPUN"
            else:
                self.frame_id = self.planet_frame.frame_id + "_INERTIAL"
        else:
            self.frame_id = id

        # Register if necessary
        if id:
            self.register()
        else:
            self.wayframe = self

        # For a fixed epoch, derive the inertial tranform now
        self.transform = None
        self.node = None

        if self.epoch is not None:
            self.transform = self.transform_at_time(self.epoch)
            self.node = self.node_at_time(self.epoch)

    ########################################

    def transform_at_time(self, time, quick={}):
        """The Transform into the this Frame at a Scalar of times."""

        # For a fixed epoch, return the fixed transform
        if self.transform is not None:
            return self.transform

        # Otherwise, calculate it for the current time
        xform = self.planet_frame.transform_at_time(time, quick=quick)
        matrix = xform.matrix.values.copy()

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
                         self.wayframe, self.reference, None)

    ########################################

    def node_at_time(self, time, quick={}):
        """Angle from the original X-axis to the ring plane ascending node.

        This serves as the X-axis of this frame."""

        # For a fixed epoch, return the fixed node
        if self.transform is not None:
            return 0.

        # Otherwise, calculate it for the current time
        xform = self.planet_frame.transform_at_time(time, quick=quick)
        matrix = xform.matrix.values

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
        import os
        import cspyce
        from oops.frame_.spiceframe import SpiceFrame
        from oops.path_.spicepath import SpicePath
        from oops.event import Event
        from oops.path_.path import Path
        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/naif0009.tls"))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/pck00010.tpc"))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/de421.bsp"))

        Path.reset_registry()
        Frame.reset_registry()

        center = SpicePath("MARS", "SSB")
        planet = SpiceFrame("IAU_MARS", "J2000")
        rings  = RingFrame(planet)
        self.assertEqual(Frame.as_wayframe("IAU_MARS"), planet.wayframe)
        self.assertEqual(Frame.as_wayframe("IAU_MARS_DESPUN"), rings.wayframe)

        time = Scalar(np.random.rand(3,4,2) * 1.e8)
        posvel = np.random.rand(3,4,2,6)
        event = Event(time, (posvel[...,0:3], posvel[...,3:6]), "SSB", "J2000")
        rotated = event.wrt_frame("IAU_MARS")
        fixed   = event.wrt_frame("IAU_MARS_DESPUN")

        # Confirm Z axis is tied to planet's pole
        diff = Scalar(rotated.pos.mvals[...,2]) - Scalar(fixed.pos.mvals[...,2])
        self.assertTrue(np.all(np.abs(diff.values < 1.e-14)))

        # Confirm X-axis is always in the J2000 equator
        xaxis = Event(time, Vector3.XAXIS, "SSB", rings.frame_id)
        test = xaxis.wrt_frame("J2000")
        self.assertTrue(np.all(np.abs(test.pos.mvals[...,2] < 1.e-14)))

        # Confirm it's at the ascending node
        xaxis = Event(time, (1,1.e-13,0), "SSB", rings.frame_id)
        test = xaxis.wrt_frame("J2000")
        self.assertTrue(np.all(test.pos.mvals[...,1] > 0.))

        # Check that pole wanders when epoch is fixed
        rings2 = RingFrame(planet, 0.)
        self.assertEqual(Frame.as_wayframe("IAU_MARS_INERTIAL"), rings2.wayframe)
        inertial = event.wrt_frame("IAU_MARS_INERTIAL")

        diff = Scalar(rotated.pos.mvals[...,2]) - Scalar(inertial.pos.mvals[...,2])
        self.assertTrue(np.all(np.abs(diff.values) < 1.e-4))
        self.assertTrue(np.mean(np.abs(diff.values) > 1.e-8))

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
