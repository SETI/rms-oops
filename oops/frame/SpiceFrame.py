import numpy as np
import unittest
import cspice

import oops

################################################################################
# SpiceFrame
################################################################################

class SpiceFrame(oops.Frame):
    """A SpiceFrame is a Frame object defined within the SPICE toolkit."""

    # Maintain a dictionary that translates names in SPICE toolkit with their
    # corresponding names in the Frame registry.

    SPICE_TRANSLATION = {"J2000":"J2000"}

    def __init__(self, spice_id, spice_reference="J2000", frame_id=None):
        """Constructor for a SpiceFrame.

        Input:
            spice_id        the name or integer ID of the destination frame or
                            of the central body as used in the SPICE toolkit.
            spice_reference the name or integer ID of the reference frame as
                            used in the SPICE toolkit; "J2000" by default.
            frame_id        the name or ID under which the frame will be
                            registered. By default, this will be the value of
                            spice_id if that is given as a string; otherwise
                            it will be the name as used by the SPICE toolkit.
        """

        self.spice_name      = SpiceFrame.spice_name(spice_id)
        self.spice_reference = SpiceFrame.spice_name(spice_reference)

        # Test the SPICE toolkit. This raises a RuntimeError if it fails
        test = cspice.sxform(self.spice_reference, self.spice_name, 0.)

        # Fill in the frame_id
        if frame_id is None:
            if type(spice_id) == type(""):
                self.frame_id = spice_id
            else:
                self.frame_id = self.spice_name
        else:
            self.frame_id = frame_id

        # Save it in the global dictionary of translations under alternative
        # names
        SpiceFrame.SPICE_TRANSLATION[self.spice_name] = self.frame_id
        SpiceFrame.SPICE_TRANSLATION[spice_id]        = self.frame_id

        # Fill in the reference_id
        self.reference_id = SpiceFrame.SPICE_TRANSLATION[self.spice_reference]

        # Fill in the origin ID
        body_id = cspice.frinfo(cspice.namfrm(self.spice_name))[0]
        self.origin_id = oops.SpicePath.SPICE_TRANSLATION[body_id]

        if self.origin_id == "SSB":
            self.origin_id = None

        # No shape
        self.shape = []

        # Always register a SpiceFrame
        self.register()

########################################

    @staticmethod
    def spice_name(arg):
        """Inteprets the argument as the name or ID of a SPICE body or SPICE
        body."""

        # Interpret the argument given as a string
        if type(arg) == type(""):

            # Validate this as the name of a frame
            try:
                id = cspice.namfrm(arg)     # does not raise an error; I may fix
            except ValueError:
                id = 0

            # If a nonzero ID is found, return the official, capitalized name
            if id != 0: return cspice.frmnam(id)

            # See if this is the name of a body
            id = cspice.bodn2c(frame)       # raises LookupError if not found

            # If so, return the name of the associated frame
            return cspice.cidfrm(id)[1]

        # Otherwise, interpret the argument given as an integer

        # See if the id has an associated frame name
        try:
            name = cspice.frmnam(arg)       # does not raise an error; I may fix
        except ValueError:
            name = ""

        # If found, return it
        if name != "": return name

        # Otherwise, see if this is the ID of a body
        name = cspice.bodc2n(id)            # raises LookupError if not found

        # If so, return the name of the body's associated frame
        return cspice.cidfrm(id)[1]

########################################

    def transform_at_time(self, time):
        """Returns a Transform object that rotates coordinates in a reference
        frame into the new frame.

        Input:
            time            a Scalar time.

        Return:             the corresponding Tranform applicable at the
                            specified time(s).
        """

        # A single input time can be handled quickly
        time = oops.Scalar.as_scalar(time)
        if time.shape == []:
            matrix6 = cspice.sxform(self.spice_reference, self.spice_name,
                                    time.vals)
            (matrix, omega) = cspice.xf2rav(matrix6)
            return oops.Transform(matrix, omega, self.frame_id,
                                                 self.reference_id)

        # Create the buffers
        matrix = np.empty(time.shape + [3,3])
        omega  = np.empty(time.shape + [3])

        # Fill in the matrix and omega using CSPICE
        for i,t in np.ndenumerate(time.vals):
            matrix6 = cspice.sxform(self.spice_reference, self.spice_name, t)
            (matrix[i], omega[i]) = cspice.xf2rav(matrix6)

        return oops.Transform(matrix, omega, self.frame_id, self.reference_id)

################################################################################
# UNIT TESTS
################################################################################

# Here we also test many of the overall Frame operations, because we can be
# confident that cspice produces valid results.

class Test_SpiceFrame(unittest.TestCase):

    def runTest(self):
        oops.Frame.initialize_registry()
        oops.Path.initialize_registry()

        ignore = oops.SpicePath("EARTH", "SSB")

        earth = SpiceFrame("IAU_EARTH", "J2000")
        time  = oops.Scalar(np.random.rand(3,4,2) * 1.e8)
        posvel = np.random.rand(3,4,2,6,1)
        event = oops.Event(time, posvel[...,0:3,0], posvel[...,3:6,0],
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
        oops.Frame.initialize_registry()
        oops.Path.initialize_registry()

        ignore = oops.SpicePath("EARTH", "SSB")
        ignore = oops.SpicePath("VENUS", "EARTH")
        ignore = oops.SpicePath("MARS", "VENUS")
        ignore = oops.SpicePath("MOON", "VENUS")

        earth  = SpiceFrame("IAU_EARTH", "J2000")
        b1950  = SpiceFrame("B1950", "IAU_EARTH")
        venus  = SpiceFrame("IAU_VENUS", "B1950")
        mars   = SpiceFrame("IAU_MARS",  "J2000")
        mars   = SpiceFrame("IAU_MOON",  "B1950")

        x2000  = oops.Frame.lookup("J2000")
        xearth = oops.Frame.lookup("IAU_EARTH")
        x1950  = oops.Frame.lookup("B1950")
        xvenus = oops.Frame.lookup("IAU_VENUS")
        xmars  = oops.Frame.lookup("IAU_MARS")
        xmoon  = oops.Frame.lookup("IAU_MOON")

        self.assertEqual(oops.Frame.common_ancestry(xmars, xvenus),
                         ([xmars, x2000],
                          [xvenus, x1950, xearth, x2000]))

        self.assertEqual(oops.Frame.common_ancestry(xmoon, xvenus),
                         ([xmoon, x1950],
                          [xvenus, x1950]))

        times = oops.Scalar(np.arange(-3.e8, 3.01e8, 0.5e7))

        frame = oops.Frame.connect("IAU_EARTH","J2000")
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspice.sxform("J2000", "IAU_EARTH", times[i])
            (matrix, omega) = cspice.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = oops.Frame.connect("J2000","IAU_EARTH")
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspice.sxform("IAU_EARTH", "J2000", times[i])
            (matrix, omega) = cspice.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = oops.Frame.connect("B1950","J2000")
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspice.sxform("J2000", "B1950", times[i])
            (matrix, omega) = cspice.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        frame = oops.Frame.connect("J2000","B1950")
        transform = frame.transform_at_time(times)
        for i in range(times.vals.size):
            matrix6 = cspice.sxform("B1950", "J2000", times[i])
            (matrix, omega) = cspice.xf2rav(matrix6)

            dmatrix = transform.matrix[i].vals - matrix
            domega  = transform.omega[i].vals  - omega

            self.assertTrue(np.all(np.abs(dmatrix) < 1.e-14))
            self.assertTrue(np.all(np.abs(domega)  < 1.e-14))

        oops.Frame.initialize_registry()
        oops.Path.initialize_registry()

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################