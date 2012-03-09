################################################################################
# oops_/cmodel/distance.py: Subclass Distance of class CoordinateModel
#
# 1/24/12 Added (MRS)
# 2/8/12 Modified (MRS) - supports new class heirarchy.
################################################################################

import numpy as np

from oops_.cmodel.cmodel_ import CoordinateModel
from oops_.units import Units

class Distance(CoordinateModel):
    """Distance is a subclass of CoordinateModel used to describe one component
    of a position vector, typically in rectangular coordinates.
    """

    def __init__(self, units=None, reference=0., downward=False):
        """The constructor for a Distance coordinate model.

        Input:
            units       the Units object used by the coordinate.
            reference   the reference location in standard coordinates
                        corresponding to a value of zero in this defined
                        coordinate system. Default is 0.
            downward    True to measure positions "downward" from the reference
                        position, i.e., in a direction opposite to the direction
                        of increase of the standard coordinate value.
        """

        if units is None: units = Units.KM

        CoordinateModel.__init__(self, name, abbrev, units, format,
                                 minimum = -np.inf,
                                 maximum =  np.inf,
                                 modulus = None,
                                 reference = reference,
                                 negated = downward)

        if self.units.exponents != (1,0,0):
            raise ValueError("illegal units for a Distance coordinate model: " +
                             str(unit))

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Distance(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
