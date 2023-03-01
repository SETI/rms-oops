################################################################################
# oops/cmodel_/longitude.py: Subclass Longitude of class CoordinateModel
################################################################################

import numpy as np

from oops.cmodel_.cmodel import CoordinateModel
from polymath import Units

class Longitude(CoordinateModel):
    """Longitude is a subclass of CoordinateModel used to describe a rotation
    angle that cycles through 360 degrees.
    """

    def __init__(self, units=None, minimum=0., reference=0., retrograde=False):
        """The constructor for a Longitude coordinate model.

        Input:
            units       the Units object used by the coordinate.
            minimum     the minimum value of the longitude, in degrees; default
                        = 0. The maximum value is always larger by 360 degrees.
            reference   the reference location in standard coordinates
                        corresponding to a value of zero in this defined
                        coordinate system. Default is 0.
            retrograde  if True, longitudes are measured in the reverse
                        direction; default is False
        """

        if units is None:
            units = Units.DEG

        modulus = Scalar(360., Units.DEG).convert(units).vals

        CoordinateModel.__init__(self, units,
                                 minimum,
                                 minimum + modulus,
                                 modulus, reference,
                                 negated = retrograde)

        if self.units.exponents != (0,0,1):
          raise ValueError("illegal units for a Longitude coordinate model: " +
                           unit.name)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Longitude(unittest.TestCase):

    def runTest(self):

        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
