################################################################################
# oops/frame/cmatrix.py: Subclass Cmatrix of class Frame
################################################################################

import numpy as np
from polymath       import Matrix3, Qube, Scalar, Vector3
from oops.frame     import Frame
from oops.path      import Path
from oops.transform import Transform
from oops.constants import RPD

class Cmatrix(Frame):
    """Frame subclass in which the frame is defined by a fixed rotation matrix.

    Most commonly, it rotates J2000 coordinates into the frame of a camera, in
    which the Z-axis points along the optic axis, the X-axis points rightward,
    and the Y-axis points downward.
    """

    # Note: Navigation frames are not generally re-used, so their IDs are
    # expendable. Frame IDs are not preserved during pickling.

    #===========================================================================
    def __init__(self, cmatrix, reference=None, frame_id=None):
        """Constructor for a Cmatrix frame.

        Input:
            cmatrix     a Matrix3 object.
            reference   the ID or frame relative to which this frame is defined;
                        None for J2000.
            frame_id    the ID under which the frame will be registered; None
                        to leave the frame unregistered
        """

        self.cmatrix = Matrix3.as_matrix3(cmatrix)

        # Required attributes
        self.frame_id  = frame_id
        self.reference = Frame.as_wayframe(reference) or Frame.J2000
        self.origin    = self.reference.origin
        self.shape     = Qube.broadcasted_shape(self.cmatrix, self.reference)
        self.keys      = set()

        # Update wayframe and frame_id; register if not temporary
        self.register()

        # It needs a wayframe before we can construct the transform
        self.transform = Transform(cmatrix, Vector3.ZERO,
                                   self.wayframe, self.reference)

    # Unpickled frames will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.cmatrix, Frame.as_primary_frame(self.reference))

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    @staticmethod
    def from_ra_dec(ra, dec, clock, reference=None, frame_id=None):
        """Construct a Cmatrix from RA, dec and celestial north clock angles.

        Input:
            ra          a Scalar defining the right ascension of the optic axis
                        in degrees.
            dec         a Scalar defining the declination of the optic axis in
                        degrees.
            clock       a Scalar defining the angle of celestial north in
                        degrees, measured clockwise from the "up" direction in
                        the observation.
            reference   the reference frame or ID; None for J2000.
            frame_id    the ID to use when registering this frame; None to leave
                        it unregistered

        Note that this Frame can have an arbitrary shape. This shape is defined
        by broadcasting the shapes of the ra, dec, twist and reference.
        """

        ra    = Scalar.as_scalar(ra)
        dec   = Scalar.as_scalar(dec)
        clock = Scalar.as_scalar(clock)
        mask = Qube.or_(ra.mask, dec.mask, clock.mask)

        # The transform is fixed so save it now
        ra = RPD * ra.values
        dec = RPD * dec.values
        twist = RPD * (180. - clock.values)

        cosr = np.cos(ra)
        sinr = np.sin(ra)

        cosd = np.cos(dec)
        sind = np.sin(dec)

        cost = np.cos(twist)
        sint = np.sin(twist)

        (cosr, cosd, cost,
         sinr, sind, sint) = np.broadcast_arrays(cosr, cosd, cost,
                                                 sinr, sind, sint)

        # Extracted from the PDS Data Dictionary definition, which is appended
        # below
        cmatrix_values = np.empty(cosr.shape + (3,3))
        cmatrix_values[...,0,0] = -sinr * cost - cosr * sind * sint
        cmatrix_values[...,0,1] =  cosr * cost - sinr * sind * sint
        cmatrix_values[...,0,2] =  cosd * sint
        cmatrix_values[...,1,0] =  sinr * sint - cosr * sind * cost
        cmatrix_values[...,1,1] = -cosr * sint - sinr * sind * cost
        cmatrix_values[...,1,2] =  cosd * cost
        cmatrix_values[...,2,0] =  cosr * cosd
        cmatrix_values[...,2,1] =  sinr * cosd
        cmatrix_values[...,2,2] =  sind

        return Cmatrix(Matrix3(cmatrix_values,mask), reference, frame_id)

    #===========================================================================
    def transform_at_time(self, time, quick=False):
        """Transform into this Frame at a Scalar of times."""

        return self.transform

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Cmatrix(unittest.TestCase):

    def runTest(self):

        np.random.seed(7316)

        # Imports are here to avoid conflicts
        import os
        import cspyce

        from oops.event              import Event
        from oops.frame.spiceframe   import SpiceFrame
        from oops.path.spicepath     import SpicePath
        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/naif0009.tls'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/pck00010.tpc'))
        cspyce.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/de421.bsp'))

        Path.reset_registry()
        Frame.reset_registry()

        _ = SpicePath('MARS', 'SSB')
        _ = SpiceFrame('IAU_MARS', 'J2000')

        # Define a version of the IAU Mars frame always rotated by 180 degrees
        # around the Z-axis
        mars180 = Cmatrix([[-1,0,0],[0,-1,0],[0,0,1]], 'IAU_MARS')

        time = Scalar(np.random.rand(100) * 1.e8)
        posvel = np.random.rand(100,6)
        event = Event(time, (posvel[...,0:3], posvel[...,3:6]), 'MARS', 'J2000')

        wrt_mars = event.wrt_frame('IAU_MARS')
        wrt_mars180 = event.wrt_frame(mars180)

        # Confirm that the components are related as expected
        self.assertTrue(np.all(wrt_mars.pos.vals[...,0] == -wrt_mars180.pos.vals[...,0]))
        self.assertTrue(np.all(wrt_mars.pos.vals[...,1] == -wrt_mars180.pos.vals[...,1]))
        self.assertTrue(np.all(wrt_mars.pos.vals[...,2] ==  wrt_mars180.pos.vals[...,2]))

        self.assertTrue(np.all(wrt_mars.vel.vals[...,0] == -wrt_mars180.vel.vals[...,0]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,1] == -wrt_mars180.vel.vals[...,1]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,2] ==  wrt_mars180.vel.vals[...,2]))

        # Define a version of the IAU Mars frame containing four 90-degree
        # rotations
        matrices = []
        for (cos,sin) in ((1,0), (0,1), (-1,0), (0,-1)):
            matrices.append([[cos,sin,0],[-sin,cos,0],[0,0,1]])

        mars90s = Cmatrix(matrices, 'IAU_MARS')

        time = Scalar(np.random.rand(100,1) * 1.e8)
        posvel = np.random.rand(100,1,6)
        event = Event(time, (posvel[...,0:3], posvel[...,3:6]), 'MARS', 'J2000')

        wrt_mars = event.wrt_frame('IAU_MARS')
        wrt_mars90s = event.wrt_frame(mars90s)

        self.assertEqual(wrt_mars.shape, (100,1))
        self.assertEqual(wrt_mars90s.shape, (100,4))

        # Confirm that the components are related as expected
        self.assertTrue(wrt_mars.pos[:,0] == wrt_mars90s.pos[:,0])
        self.assertTrue(wrt_mars.vel[:,0] == wrt_mars90s.vel[:,0])

        self.assertTrue(np.all(wrt_mars.pos.vals[...,0][:,0] == -wrt_mars90s.pos.vals[...,0][:,2]))
        self.assertTrue(np.all(wrt_mars.pos.vals[...,1][:,0] == -wrt_mars90s.pos.vals[...,1][:,2]))
        self.assertTrue(np.all(wrt_mars.pos.vals[...,2][:,0] ==  wrt_mars90s.pos.vals[...,2][:,2]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,0][:,0] == -wrt_mars90s.vel.vals[...,0][:,2]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,1][:,0] == -wrt_mars90s.vel.vals[...,1][:,2]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,2][:,0] ==  wrt_mars90s.vel.vals[...,2][:,2]))

        self.assertTrue(np.all(wrt_mars.pos.vals[...,0][:,0] == -wrt_mars90s.pos.vals[...,1][:,1]))
        self.assertTrue(np.all(wrt_mars.pos.vals[...,1][:,0] ==  wrt_mars90s.pos.vals[...,0][:,1]))
        self.assertTrue(np.all(wrt_mars.pos.vals[...,2][:,0] ==  wrt_mars90s.pos.vals[...,2][:,1]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,0][:,0] == -wrt_mars90s.vel.vals[...,1][:,1]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,1][:,0] ==  wrt_mars90s.vel.vals[...,0][:,1]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,2][:,0] ==  wrt_mars90s.vel.vals[...,2][:,1]))

        self.assertTrue(np.all(wrt_mars.pos.vals[...,0][:,0] ==  wrt_mars90s.pos.vals[...,1][:,3]))
        self.assertTrue(np.all(wrt_mars.pos.vals[...,1][:,0] == -wrt_mars90s.pos.vals[...,0][:,3]))
        self.assertTrue(np.all(wrt_mars.pos.vals[...,2][:,0] ==  wrt_mars90s.pos.vals[...,2][:,3]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,0][:,0] ==  wrt_mars90s.vel.vals[...,1][:,3]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,1][:,0] == -wrt_mars90s.vel.vals[...,0][:,3]))
        self.assertTrue(np.all(wrt_mars.vel.vals[...,2][:,0] ==  wrt_mars90s.vel.vals[...,2][:,3]))

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
# Notes from the PDS Data Dictionary
#
# OBJECT = ELEMENT_DEFINITION
#   NAME = TWIST_ANGLE
#   STATUS_TYPE = APPROVED
#   GENERAL_DATA_TYPE = REAL
#   UNIT_ID = DEG
#   STANDARD_VALUE_TYPE = RANGE
#   MINIMUM = 0
#   MAXIMUM = 360
#   DESCRIPTION = "
#      The twist_angle element provides the angle of rotation about an
#      optical axis relative to celestial coordinates. The RIGHT_ASCENSION,
#      DECLINATION and TWIST_ANGLE elements define the pointing direction
#      and orientation of an image or scan platform.
#      Note: The specific mathematical definition of TWIST_ANGLE depends on
#            the value of the TWIST_ANGLE_TYPE element. If unspecified,
#            TWIST_ANGLE_TYPE = GALILEO for Galileo data and
#      TWIST_ANGLE_TYPE = DEFAULT for all other data.
#
#      Note: This element bears a simple relationship to the value of
#            CELESTIAL_NORTH_CLOCK_ANGLE.
#            When TWIST_ANGLE_TYPE = DEFAULT,
#                 TWIST_ANGLE = (180 - CELESTIAL_NORTH_CLOCK_ANGLE) mod 360;
#            when TWIST_ANGLE_TYPE = GALILEO,
#                 TWIST_ANGLE = (270 - CELESTIAL_NORTH_CLOCK_ANGLE) mod 360."
# END_OBJECT = ELEMENT_DEFINITION
# END
# OBJECT = ELEMENT_DEFINITION
#   NAME = TWIST_ANGLE_TYPE
#   STATUS_TYPE = APPROVED
#   GENERAL_DATA_TYPE = IDENTIFIER
#   UNIT_ID = NONE
#   STANDARD_VALUE_TYPE = STATIC
#   MAXIMUM_LENGTH = 10
#   DESCRIPTION = "
#      The twist_angle_type element determines the specific mathematical
#      meaning of the element TWIST_ANGLE when it is used to specify the
#      pointing of an image or scan platform. Allowed values are DEFAULT and
#      GALILEO. If unspecified, the value is GALILEO for Galileo data and
#      DEFAULT for all other data.
#
#      The three elements RIGHT_ASCENSION,
#      DECLINATION and TWIST_ANGLE define the C-matrix, which transforms a
#      3-vector in celestial coordinates into a frame fixed to an image plane.
#      Celestial coordinates refer to a frame in which the x-axis points
#      toward the First Point of Aries and the z-axis points to the celestial
#      pole; these coordinates are assumed to be in J2000 unless otherwise
#      specified. Image plane coordinates are defined such that the x-axis
#      points right, the y-axis points down, and the z-axis points along the
#      camera's optic axis, when an image is displayed as defined by the
#      SAMPLE_DISPLAY_DIRECTION and LINE_DISPLAY_DIRECTION elements.
#
#      For TWIST_ANGLE_TYPE = DEFAULT, the C-matrix is equal to
#        C-matrix = [T]3 [90-D]1 [R+90]3
#
#       = |-sinR cosT-cosR sinD sinT  cosR cosT-sinR sinD sinT  cosD sinT|
#         | sinR sinT-cosR sinD cosT -cosR sinT-sinR sinD cosT  cosD cosT|
#         |        cosR cosD                 sinR cosD             sinD  |
#
#      For TWIST_ANGLE_TYPE = GALILEO, the C-matrix is defined by
#...
#
#      Here the notation [X]n specifies a rotation about the nth axis by
#      angle X (in degrees). R refers to right ascension, D to declination,
#      and T to twist angle."
#   STANDARD_VALUE_SET = {
#      "DEFAULT",
#      "GALILEO"}
# END_OBJECT = ELEMENT_DEFINITION
################################################################################
