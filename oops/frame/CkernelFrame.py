import numpy as np
import unittest

from Frame import Frame
from oops.Transform import Transform

import cspice

################################################################################
# CkernelFrame
################################################################################

class CkernelFrame(Frame):
    """A CkernelFrame is a Frame object the defines the pointing of an
    an instrument at or near the time of an observation, using information from
    a C kernel within the SPICE toolkit."""

    def __init__(self, inst, ticks=1., reference="J2000", ):
        """Constructor for a SpiceFrame.

        Input:
            inst            the name or ID of the instrument. The associated
                            SPICE C kernels, Frame kernels and Spacecraft Clock
                            kernels must already have been loaded via
                            cspice.furnsh().
            ticks           the error tolerance of the C kernel lookup, in units
                            of the ticks of the spacecraft clock.
            reference       name or ID of the reference frame; the default is
                            "J2000".
        """

        # Interpret the instrument ID
        if type(inst) == type(0):
            self.inst_id = inst
        else:
            name = SpiceKernel.FrameName(inst)
            self.inst_id = cspice.namfrm(name)

        # Determine the spacecraft ID
        self.sc_id = -int(-self.inst_id / 1000)

        # Interpret the reference frame name
        self.reference = SpiceKernel.FrameName(reference)

        return

########################################

    def at_time(self, time):
        """Returns a Transform object that rotates coordinates in a reference
        frame into the new frame.

        Input:
            time            a Scalar time.

        Return:             the corresponding Tranform applicable at the
                            specified time(s).
        """

        # Convert the time to a spacecraft clock value
        self.time_requested = time
        self.sclk_requested = cspice.sce2c(self.sc_id, time)

        # Look up the information for the requested time
        # Raises a LookupError on failure
        (self.matrix,
         self.omega,
         self.sclk_returned) = cspice.ckgpav(self.inst_id,
                                             time,
                                             ticks,
                                             self.reference)

        self.time_returned = cspice.sct2e(self.sc_id, self.sclk_returned)

        return

########################################
# UNIT TESTS
########################################

class Test_CkernelFrame(unittest.TestCase):

    def runTest(self):

        print "\n\nWARNING: CkernelFrame unit testing is not yet implemented"
        print "KNOWN ERROR: C kernel rotation is not used in at_time()\n

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
