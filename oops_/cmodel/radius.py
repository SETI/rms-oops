################################################################################
# oops_/cmodel/radius.py: Subclass Radius of class CoordinateModel
#
# 1/24/12 Added (MRS)
# 2/8/12 Modified (MRS) - supports new class heirarchy.
################################################################################

import numpy as np

from baseclass import CoordinateModel
from oops_.units import Units

class Radius(CoordinateModel):
    """Radius is a subclass of CoordinateModel used to describe the length of a
    radial vector, typically in spherical or cylindrical coordinates. It cannot
    be negative.
    """

    def __init__(self, units=None, reference=0., inward=False):
        """The constructor for a Radius coordinate model.

        Input:
            units       the Units object used by the coordinate.
            reference   the reference location in standard coordinates
                        corresponding to a value of zero in this defined
                        coordinate system. Default is 0.
            inward      True to measure radii inward from the reference radius
                        rather than outward.
        """

        if units is None: units = Units.KM

        Coordinate.__init__(self, units,
                            minimum = 0.,
                            maximum = np.inf,
                            modulus = None,
                            reference = reference,
                            negated = inward)

        if self.units.exponents != (1,0,0):
            raise ValueError("illegal units for a Radius coordinate: " +
                             unit.name)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Radius(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
