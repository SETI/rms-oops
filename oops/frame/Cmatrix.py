import numpy as np
import unittest

import oops

################################################################################
# Cmatrix
################################################################################

class Cmatrix(oops.Frame):
    """A CmatrixFrame is a fixed rotation matrix, used to describe the pointing
    and orientation of an observation such as a camera image. It rotates J2000
    coordinates into the frame of a camera, in which the Z-axis points along the
    optic axis, the X-axis points rightward, and the Y-axis points downward.
    """

    RPD = np.pi/180.

    def __init__(self, ra, dec, clock, id, reference="J2000"):
        """Constructor for a CmatrixFrame.

        Input:
            ra              a Scalar defining the right ascension of the optic
                            axis in degrees.

            dec             a Scalar defining the declination of the optic axis
                            in degrees.

            clock           a Scalar defining the angle of celestial north in
                            degrees, measured clockwise from the "up" direction
                            in the observation.

            id              the name ID to use when registering this frame. Note
                            that any prior frame using the same ID is replaced.

            reference       the ID of the coordinate reference frame, typically
                            J2000.

        Note that this Frame can have an arbitrary shape. This shape is defined
        by broadcasting the shapes of the ra, dec and twist Scalars.
        """

        self.ra = oops.Scalar.as_scalar(ra)
        self.dec = oops.Scalar.as_scalar(dec)
        self.clock = oops.Scalar.as_scalar(clock)

        self.frame_id = frame_id
        self.reference_id = reference_id
        self.origin_id = None       # Frame is fixed, inertial
        self.shape = oops.Array.broadcast_shape((self.ra, self.dec, self.clock))

        # We sometimes re-use the names if frames describing images or other
        # data products
        self.reregister()

        # The transform is fixed so save it now
        r = Cmatrix.RPD * self.ra.vals
        d = Cmatrix.RPD * self.dec.vals
        t = Cmatrix.RPD * (180. - self.clock.vals)

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
        matrix_vals = np.array(
            [[-sinr * cost - cosr * sind * sint,
               cosr * cost - sinr * sind * sint, cosd * sint],
             [ sinr * sint - cosr * sind * cost,
              -cosr * sint - sinr * sind * cost, cosd * cost],
             [ cosr * cosd,  sinr * cosd,        sind       ]])

        self.transform = oops.Transform(matrix_vals, (0.,0.,0.),
                                        self.frame_id, self.reference_id)

########################################

    def transform_at_time(self, time):
        """Returns the Transform to the given Frame at a specified Scalar of
        times. The shape of the time Scalar must be suitable for broadcasting
        with the shape of the Frame.
        """

        return self.transform

################################################################################
# UNIT TESTS
################################################################################ 

class Test_Cmatrix(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

################################################################################
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
