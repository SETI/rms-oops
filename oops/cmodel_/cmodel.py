################################################################################
# oops/cmodel_/cmodel.py: Abstract class CoordinateModel
#
# 1/24/12 Added (MRS)
# 2/8/12 Modified (MRS) - supports new class heirarchy.
################################################################################

import numpy as np

from oops.array_ import *

class CoordinateModel(object):
    """CoordinateModel is an abstract class used to describe the default
    numeric range, units, and formating of a geometric quantity.
    """

    def __init__(self, units, minimum=-np.inf, maximum=np.inf, modulus=None,
                       reference=0., negated=False):
        """The general constructor for a Coordinate object.

        Input:
            units       the default Units object.
            minimum     the global minimum value of a coordinate, if any,
                        specified in the default units for the coordinate.
            maximum     the global maximum value of a coordinate, if any.
            modulus     the modulus value of the coordinate, if any, specified
                        in the default units for the coordinate. If not None,
                        then all coordinate values returned will fall in the
                        range (minimum, minimum+modulus).
            reference   the default reference coordinate relative to which
                        values are specified, in default units. If defined,
                        then this is this location relative to the origin of the
                        standard coordinate system corresponding to a value of
                        zero for this coordinate value.
            negated     if True, then these coordinate values increase from the
                        reference point in a direction opposite to the standard
                        coordinates.

        Note: Currently the maximum value is unused, and the minimum is only
        used when a modulus is also given. We might eventually use minimum and
        maximum to implement range checks.
        """

        self.units     = units
        self.minimum   = minimum
        self.maximum   = maximum
        self.modulus   = modulus
        self.reference = reference
        self.negated   = negated

    def to_standard(self, scalar):
        """Converts a scalar of coordinates to standard units involving km,
        seconds and radians, and measured in the default direction relative to
        the default origin."""

        result = Scalar.as_scalar(scalar).convert_units(self.units)
        if self.negated: result = -result
        if self.reference != 0: result += self.reference
        return result.convert_units(None)

    def to_model(self, scalar):
        """Converts the scalar of a coordinate from standard units involving km,
        seconds and radians into a value relative to the specified units, origin
        and direction. Applies the modulus if any.
        """

        result = Scalar.as_scalar(scalar).convert_units(self.units)
        if self.reference != 0: result -= self.reference
        if self.negated: result = -result

        if self.modulus is not None:
            result = self.minimum + (result - self.minimum) % self.modulus

        return result

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_CoordinateModel(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
