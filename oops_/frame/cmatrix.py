################################################################################
# oops_/frame/cmatrix.py: Subclass Cmatrix of class Frame
#
# 1/17/12 (BSW) - id and reference, declared as arguments to __init__, were not
#   refereced (instead frane_id and referece_id were refereced, which did not
#   exist, so replaced frame_id with id and reference_id with reference in body
#   of __init__.
# 2/2/12 Modified (MRS) - converted to new class names and hierarchy.
################################################################################

import numpy as np

from oops_.frame.frame_ import Frame
from oops_.array.all import *
from oops_.config import QUICK
from oops_.transform import Transform
import oops_.constants as constants
import oops_.registry as registry

class Cmatrix(Frame):
    """Cmatrix is a Frame subclass in which the frame is defined by a fixed
    rotation matrix. Most commonly, it rotates J2000 coordinates into the frame
    of a camera, in which the Z-axis points along the optic axis, the X-axis
    points rightward, and the Y-axis points downward.
    """

    def __init__(self, cmatrix, id, reference="J2000"):
        """Constructor for a Cmatrix frame.

        Input:
            cmatrix     a Matrix3 object.
            id          the ID under which the frame will be registered. It
                        replaces a pre-existing frame of the same name.
            reference   the ID or frame relative to which this frame is defined.
        """

        cmatrix = Matrix3.as_matrix3(cmatrix)

        reference = registry.as_frame(reference)
        self.reference_id = reference.frame_id
        self.origin_id = reference.origin_id

        self.shape = cmatrix.shape
        self.frame_id = id

        self.reregister() # We have to register it before we can construct the
                          # Transform

        self.transform = Transform(cmatrix, (0,0,0),
                                   self.frame_id,
                                   self.reference_id)

    @staticmethod
    def from_ra_dec(ra, dec, clock, id, reference="J2000"):
        """Constructs a Cmatrix frame given the RA, dec and celestial north
        clock angles.

        Input:
            ra              a Scalar or UnitScalar defining the right ascension
                            of the optic axis in default units of degrees.

            dec             a Scalar or UnitScalar defining the declination of
                            the optic axis in degrees.

            clock           a Scalar or UnitScalar defining the angle of
                            celestial north in degrees, measured clockwise from
                            the "up" direction in the observation.

            id              the name ID to use when registering this frame. Note
                            that any prior frame using the same ID is replaced.

            reference       the ID of the coordinate reference frame, typically
                            J2000.

        Note that this Frame can have an arbitrary shape. This shape is defined
        by broadcasting the shapes of the ra, dec and twist Scalars.
        """

        ra    = Scalar.as_scalar(ra)
        dec   = Scalar.as_scalar(dec)
        clock = Scalar.as_scalar(clock)

        # The transform is fixed so save it now
        r = constants.RPD * ra.vals
        d = constants.RPD * dec.vals
        t = constants.RPD * (180. - clock.vals)

        cosr = np.cos(r)
        sinr = np.sin(r)

        cosd = np.cos(d)
        sind = np.sin(d)

        cost = np.cos(t)
        sint = np.sin(t)

        (cosr, sinr,
         cosd, sind,
         cost, sint) = np.broadcast_arrays(cosr, sinr, cosd, sind, cost, sint)

        # Extracted from the PDS Data Dictionary definition, which is appended
        # below
        cmatrix = Matrix3(
            [[-sinr * cost - cosr * sind * sint,
               cosr * cost - sinr * sind * sint, cosd * sint],
             [ sinr * sint - cosr * sind * cost,
              -cosr * sint - sinr * sind * cost, cosd * cost],
             [ cosr * cosd,  sinr * cosd,        sind       ]])

        return Cmatrix(cmatrix, id, reference)

########################################

    def transform_at_time(self, time, quick=QUICK):
        """Returns the Transform to the given Frame at a specified Scalar of
        times."""

        return self.transform

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Cmatrix(unittest.TestCase):

    def runTest(self):

        # Imports are here to avoid conflicts
        from oops_.frame.spiceframe import SpiceFrame
        from oops_.path.spicepath import SpicePath
        from oops_.event import Event

        registry.initialize_frame_registry()
        registry.initialize_path_registry()

        ignore = SpicePath("MARS", "SSB")
        mars = SpiceFrame("IAU_MARS", "J2000")

        # Define a version of the IAU Mars frame always rotated by 180 degrees
        # around the Z-axis
        mars180 = Cmatrix([[-1,0,0],[0,-1,0],[0,0,1]], "MARS180", "IAU_MARS")

        time = Scalar(np.random.rand(100) * 1.e8)
        posvel = np.random.rand(100,6)
        event = Event(time, posvel[...,0:3], posvel[...,3:6], "MARS", "J2000")

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

        mars90s = Cmatrix(matrices, "MARS90S", "IAU_MARS")

        time = Scalar(np.random.rand(100,1) * 1.e8)
        posvel = np.random.rand(100,1,6)
        event = Event(time, posvel[...,0:3],
                                 posvel[...,3:6], "MARS", "J2000")

        wrt_mars = event.wrt_frame("IAU_MARS")
        wrt_mars90s = event.wrt_frame("MARS90S")

        self.assertEqual(wrt_mars.shape, [100,1])
        self.assertEqual(wrt_mars90s.shape, [100,4])

        # Confirm that the components are related as expected
        self.assertTrue(wrt_mars.pos[:,0] == wrt_mars90s.pos[:,0])
        self.assertTrue(wrt_mars.vel[:,0] == wrt_mars90s.vel[:,0])

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

        registry.initialize_frame_registry()
        registry.initialize_path_registry()

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
