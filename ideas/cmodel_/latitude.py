################################################################################
# oops/cmodel_/latitude.py: Subclass Latitude of class CoordinateModel
################################################################################

import numpy as np

from oops.cmodel_.cmodel import CoordinateModel
from polymath import Units

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

        if units is None:
            units = Units.DEG

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

        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
