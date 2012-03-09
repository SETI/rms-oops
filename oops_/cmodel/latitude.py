################################################################################
# oops_/cmodel/latitude.py: Subclass Latitude of class CoordinateModel
#
# 1/24/12 Added (MRS)
# 2/8/12 Modified (MRS) - supports new class heirarchy.
################################################################################

import numpy as np

from oops_.cmodel.cmodel_ import CoordinateModel
from oops_.units import Units

class Latitude(CoordinateModel):
    """Latitude is a subclass of CoordinateModel used to describe an angle
    between -90 and +90 degrees.
    """

    def __init__(self, units=None, southward=False):
        """The constructor for a Latitude coordinate model.

        Input:
            units       the Units object used by the coordinate.
            southward   if True, latitudes are measured in the reverse
                        direction; default is False
        """

        if units is None: units = Units.DEG

        minimum = Scalar(-90., Units.DEG).convert(units).vals
        maximum = Scalar( 90., Units.DEG).convert(units).vals

        CoordinateModel.__init__(self, name, abbrev, units, format,
                                 minimum, maximum,
                                 modulus   = None,
                                 reference = 0.,
                                 negated   = southward)

        if self.units.exponents != (0,0,1):
            raise ValueError("illegal units for a Latitude coordinate model: " +
                             unit.name)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Latitude(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
