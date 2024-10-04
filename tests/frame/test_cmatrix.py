################################################################################
# oops/frame/cmatrix.py: Subclass Cmatrix of class Frame
################################################################################

import numpy as np
import os
import unittest

import cspyce

from polymath   import Scalar
from oops.body  import Body
from oops.event import Event
from oops.frame import Frame, Cmatrix, SpiceFrame
from oops.path  import Path, SpicePath
from oops.unittester_support import TESTDATA_PARENT_DIRECTORY, TEST_FILECACHE


class Test_Cmatrix(unittest.TestCase):

    def setUp(self):
        cspyce.furnsh(TEST_FILECACHE.retrieve(
            os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/naif0009.tls')))
        cspyce.furnsh(TEST_FILECACHE.retrieve(
            os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/pck00010.tpc')))
        cspyce.furnsh(TEST_FILECACHE.retrieve(
            os.path.join(TESTDATA_PARENT_DIRECTORY, 'SPICE/de421.bsp')))
        Path.reset_registry()
        Frame.reset_registry()

    def tearDown(self):
        pass

    def runTest(self):

        np.random.seed(7316)

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
