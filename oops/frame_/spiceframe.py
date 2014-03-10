################################################################################
# oops/frame_/spiceframe.py: Subclass SpiceFrame of class Frame
################################################################################

import numpy as np
from polymath import *
import cspice
import os.path

from oops.frame_.frame import Frame
from oops.config       import QUICK
from oops.transform    import Transform

import oops.registry as registry
import oops.spice_support as spice

from oops.constants import *

class SpiceFrame(Frame):
    """A SpiceFrame is a Frame object defined within the SPICE toolkit."""

    def __init__(self, spice_frame, spice_reference="J2000", id=None):
        """Constructor for a SpiceFrame.

        Input:
            spice_frame     the name or integer ID of the destination frame or
                            of the central body as used in the SPICE toolkit.

            spice_reference the name or integer ID of the reference frame as
                            used in the SPICE toolkit; "J2000" by default.

            id              the name or ID under which the frame will be
                            registered. By default, this will be the value of
                            spice_id if that is given as a string; otherwise
                            it will be the name as used by the SPICE toolkit.
        """

        # Interpret the SPICE frame and reference IDs
        (self.spice_frame_id,
         self.spice_frame_name) = spice.frame_id_and_name(spice_frame)

        (self.spice_reference_id,
         self.spice_reference_name) = spice.frame_id_and_name(spice_reference)

        # Fill in the OOPS frame_id and reference_id
        if id is not None:
            self.frame_id = id
        else:
            self.frame_id = self.spice_frame_name

        self.reference_id = spice.FRAME_TRANSLATION[self.spice_reference_id]

        # Save it in the global dictionary of translations under alternative
        # names
        spice.FRAME_TRANSLATION[self.spice_frame_id]   = self.frame_id
        spice.FRAME_TRANSLATION[self.spice_frame_name] = self.frame_id

        # Fill in the origin ID
        self.spice_origin_id   = cspice.frinfo(self.spice_frame_id)[0]
        self.spice_origin_name = cspice.bodc2n(self.spice_origin_id)

        self.origin_id = spice.PATH_TRANSLATION[self.spice_origin_id]
        if self.origin_id == "SSB": self.origin_id = None

        # No shape
        self.shape = []

        # Always register a SpiceFrame
        self.register()

########################################

    def transform_at_time(self, time, quick=None):
        """Returns a Transform object that rotates coordinates in a reference
        frame into the new frame.

        Input:
            time            a Scalar time.

        Return:             the corresponding Tranform applicable at the
                            specified time(s).
        """

        # A single input time can be handled quickly
        time = Scalar.as_scalar(time)
        if time.shape == []:
            matrix6 = cspice.sxform(self.spice_reference_name,
                                    self.spice_frame_name,
                                    float(time.vals))
            (matrix, omega) = cspice.xf2rav(matrix6)
            return Transform(matrix, omega, self.frame_id, self.reference_id)

        # Apply the quick_frame if requested, possibly making a recursive call
        if quick is None: quick = QUICK.flag
        if quick:
            return self.quick_frame(time, quick).transform_at_time(time, False)

        # Create the buffers
        matrix = np.empty(time.shape + (3,3))
        omega  = np.empty(time.shape + (3,))

        # Fill in the matrix and omega using CSPICE
        for i,t in np.ndenumerate(time.vals):
            matrix6 = cspice.sxform(self.spice_reference_name,
                                    self.spice_frame_name,
                                    t)
            (matrix[i], omega[i]) = cspice.xf2rav(matrix6)

        return Transform(matrix, omega, self.frame_id, self.reference_id)

################################################################################
# UNIT TESTS
################################################################################

# Here we also test many of the overall Frame operations, because we can be
# confident that cspice produces valid results.

import unittest

class Test_SpiceFrame(unittest.TestCase):

    def runTest(self):

        # Import is here to avoid conflicts
        from oops.path_.path import Path
        from oops.path_.spicepath import SpicePath
        from oops.event import Event
        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/naif0009.tls"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/pck00010.tpc"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE/de421.bsp"))

        Path.USE_QUICKPATHS = False
        Frame.USE_QUICKFRAMES = False

        registry.initialize()

        ignore = SpicePath("EARTH", "SSB")

        earth = SpiceFrame("IAU_EARTH", "J2000")
        time  = Scalar(np.random.rand(3,4,2) * 1.e8)
        posvel = np.random.rand(3,4,2,6,1)
        event = Event(time, posvel[...,0:3,0], posvel[...,3:6,0],
                            "SSB", "J2000")
        rotated = event.wrt_frame("IAU_EARTH")

        for i,t in np.ndenumerate(time.vals):
            matrix6 = cspice.sxform("J2000", "IAU_EARTH", t)
            spiceval = np.matrix(matrix6) * np.matrix(posvel[i])

            dpos = rotated.pos[i].vals[...,np.newaxis] - spiceval[0:3,0]
            dvel = rotated.vel[i].vals[...,np.newaxis] - spiceval[3:6,0]

            self.assertTrue(np.all(np.abs(dpos) < 1.e-15))
            self.assertTrue(np.all(np.abs(dvel) < 1.e-15))

        # Tests of combined frames
        registry.initialize_frame_registry()
        registry.initialize_path_registry()

        ignore = SpicePath("EARTH", "SSB")
        ignore = SpicePath("VENUS", "EARTH")
        ignore = SpicePath("MARS", "VENUS")
        ignore = SpicePath("MOON", "VENUS")

        earth  = SpiceFrame("IAU_EARTH", "J2000")
        b1950  = SpiceFrame("B1950", "IAU_EARTH")
        venus  = SpiceFrame("IAU_VENUS", "B1950")
        mars   = SpiceFrame("IAU_MARS",  "J2000")
        mars   = SpiceFrame("IAU_MOON",  "B1950")

        x2000  = registry.frame_lookup("J2000")
        xearth = registry.frame_lookup("IAU_EARTH")
        x1950  = registry.frame_lookup("B1950")
        xvenus = registry.frame_lookup("IAU_VENUS")
        xmars  = registry.frame_lookup("IAU_MARS")
        xmoon  = registry.frame_lookup("IAU_MOON")

        self.assertEqual(Frame.common_ancestry(xmars, xvenus),
                         ([xmars, x2000],
                          [xvenus, x1950, xearth, x2000]))

        self.assertEqual(Frame.common_ancestry(xmoon, xvenus),
                         ([xmoon, x1950],
                          [xvenus, x1950]))

        times = Scalar(np.arange(-3.e8, 3.01e8, 0.5e7))

        frame = registry.connect_frames("IAU_EARTH","J2000")
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspice.sxform("J2000", "IAU_EARTH", times[i].vals)
            (matrix, omega) = cspice.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = registry.connect_frames("J2000","IAU_EARTH")
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspice.sxform("IAU_EARTH", "J2000", times[i].vals)
            (matrix, omega) = cspice.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = registry.connect_frames("B1950","J2000")
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspice.sxform("J2000", "B1950", times[i].vals)
            (matrix, omega) = cspice.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = registry.connect_frames("J2000","B1950")
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspice.sxform("B1950", "J2000", times[i].vals)
            (matrix, omega) = cspice.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        registry.initialize_frame_registry()
        registry.initialize_path_registry()

        ########################################
        # Test for a Cassini C kernel
        ########################################

        # Load all the required kernels for Cassini ISS on 2007-312
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "naif0009.tls"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "cas00149.tsc"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "cas_v40.tf"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "cas_status_v04.tf"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "cas_iss_v10.ti"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "pck00010.tpc"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "cpck14Oct2011.tpc"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "de421.bsp"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "sat052.bsp"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "sat083.bsp"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "sat125.bsp"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "sat128.bsp"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "sat164.bsp"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "07312_07317ra.bc"))
        cspice.furnsh(os.path.join(TESTDATA_PARENT_DIRECTORY, "SPICE", "080123R_SCPSE_07309_07329.bsp"))

        ignore = SpicePath("CASSINI", "SSB")
        ignore = SpiceFrame("CASSINI_ISS_NAC")
        ignore = SpiceFrame("CASSINI_ISS_WAC")

        # Look up N1573186009_1.IMG from COISS_2039/data/1573186009_1573197826/
        timestring = "2007-312T03:34:16.391"
        tdb = cspice.str2et(timestring)

        nacframe = registry.connect_frames("J2000", "CASSINI_ISS_NAC")
        matrix = nacframe.transform_at_time(tdb).matrix
        optic_axis = (matrix * Vector3((0,0,1))).vals

        test_ra  = (np.arctan2(optic_axis[1], optic_axis[0]) * DPR) % 360
        test_dec = np.arcsin(optic_axis[2]) * DPR

        right_ascension = 194.30861     # from the index table
        declination = 3.142808

        self.assertTrue(np.all(np.abs(test_ra - right_ascension) < 0.5))
        self.assertTrue(np.all(np.abs(test_dec - declination) < 0.5))

        registry.initialize()
        spice.initialize()

        Path.USE_QUICKPATHS = True
        Frame.USE_QUICKFRAMES = True

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
