################################################################################
# UnitScalar
#
# Added 1/24/12 (MRS)
################################################################################

import numpy as np
import unittest

from oops.broadcastable.Array import Array
from oops.broadcastable.Scalar import Scalar
from oops.Unit import Unit, CM, M, KM

class UnitScalar(Scalar):
    """An arbitrary Array of scalars with an associated Unit."""

    OOPS_CLASS = "Scalar"

    def __init__(self, arg, unit=None):

        if isinstance(arg, Scalar): arg = arg.vals

        if np.shape(arg) == ():
            self.vals  = float(arg)
            self.rank  = 0
            self.item  = []
            self.shape = []
        else:
            self.vals  = np.asfarray(arg)
            self.rank  = 0
            self.item  = []
            self.shape = list(self.vals.shape)

        if unit is None:
            self.unit = Unit.UNITLESS
        else:
            self.unit = unit

        return

    def as_standard(self):
        """Returns the values converted to standard units involving km, seconds
        and radians."""

        return Scalar.as_scalar(self.vals) * self.unit.factor

    def as_unitless(self):
        """Returns the values as a Scalar object, with no numeric conversion."""

        return Scalar.as_scalar(self.vals)

    def convert(self, unit):
        """Returns a UnitScalar converted to the specified units."""

        return UnitScalar(self.unit.convert(self.vals, unit), unit)

    def __str__(self):
        string = str(self.vals)
        return string[:-1] + ", " + str(self.unit) + ")"

    def __repr__(self):
        string = repr(self.vals)
        return string[:-1] + ", " + str(self.unit) + ")"

    # unary (+) operator
    def __pos__(self):
        return self

    # unary (-) operator
    def __neg__(self):
        return UnitScalar(-self.vals, self.unit)

    # binary (+) operator
    def __add__(self, arg):
        if isinstance(arg, UnitScalar): arg = arg.convert(self.unit).vals
        if isinstance(arg, Scalar): arg = arg.vals
        return UnitScalar(self.vals + arg, self.unit)

    # binary (-) operator
    def __sub__(self, arg):
        if isinstance(arg, UnitScalar): arg = arg.convert(self.unit).vals
        if isinstance(arg, Scalar): arg = arg.vals
        return UnitScalar(self.vals - arg, self.unit)

    # binary (+=) operator
    def __iadd__(self, arg):
        if isinstance(arg, UnitScalar): arg = arg.convert(self.unit).vals
        if isinstance(arg, Scalar): arg = arg.vals

        self.vals += arg
        return self

    # binary (-=) operator
    def __isub__(self, arg):
        if isinstance(arg, UnitScalar): arg = arg.convert(self.unit).vals
        if isinstance(arg, Scalar): arg = arg.vals

        self.vals -= arg
        return self

    # binary (*) operator
    def __mul__(self, arg):
        if isinstance(arg, Unit):
            return UnitScalar(self.vals, self.unit * arg)

        if isinstance(arg, UnitScalar):
            return UnitScalar(self.vals * arg.vals, self.unit * arg.unit)

        return UnitScalar(self.vals * arg, self.unit)

    # binary (/) operator
    def __div__(self, arg):
        if isinstance(arg, Unit):
            return UnitScalar(self.vals, self.unit / arg)

        if isinstance(arg, UnitScalar):
            return UnitScalar(self.vals / arg.vals, self.unit / arg.unit)

        return UnitScalar(self.vals / arg, self.unit)

    # binary (*=) operator
    def __imul__(self, arg):
        if isinstance(arg, Unit):
            self.unit = self.unit * arg
            return self

        if isinstance(arg, UnitScalar):
            self.vals *= arg.vals
            self.unit = self.unit * arg.unit
            return self

        self.vals *= arg
        return self

    # binary (/=) operator
    def __idiv__(self, arg):
        if isinstance(arg, Unit):
            self.unit = self.unit / arg
            return self

        if isinstance(arg, UnitScalar):
            self.vals /= arg.vals
            self.unit = self.unit / arg.unit
            return self

        self.vals /= arg
        return self

    # (==) operator
    def __eq__(self, arg):
        if isinstance(arg, UnitScalar):
            arg = arg.convert(self.unit).as_unitless()

        return self.as_unitless() == arg

    # (!=) operator
    def __ne__(self, arg):
        if isinstance(arg, UnitScalar):
            arg = arg.convert(self.unit).as_unitless()

        return self.as_unitless() != arg

    # (<) operator
    def __lt__(self, arg):
        if isinstance(arg, UnitScalar):
            arg = arg.convert(self.unit).as_unitless()

        return self.as_unitless() < arg

    # (>) operator
    def __gt__(self, arg):
        if isinstance(arg, UnitScalar):
            arg = arg.convert(self.unit).as_unitless()

        return self.as_unitless() > arg

    # (<=) operator
    def __le__(self, arg):
        if isinstance(arg, UnitScalar):
            arg = arg.convert(self.unit).as_unitless()

        return self.as_unitless() <= arg

    # (>=) operator
    def __ge__(self, arg):
        if isinstance(arg, UnitScalar):
            arg = arg.convert(self.unit).as_unitless()

        return self.as_unitless() >= arg

########################################
# UNIT TESTS
########################################

class Test_UnitScalar(unittest.TestCase):

    def runTest(self):

        test = UnitScalar([1.e5, 2.e5, 3.e5], CM)
        self.assertEqual(test, UnitScalar([1.e5, 2.e5, 3.e5], CM))
        self.assertEqual(test, UnitScalar([1.e3, 2.e3, 3.e3], M))
        self.assertEqual(test, UnitScalar([1.,   2.,   3.  ], KM))
        self.assertEqual(test, Scalar([1.e5, 2.e5, 3.e5]))
        self.assertEqual(test, [1.e5, 2.e5, 3.e5])

        self.assertEqual(test == UnitScalar([1.e3, 2.e3, 3.1e3], M),
                         (True, True, False))
        self.assertEqual(test <= UnitScalar([1.e3, 2.e3, 3.1e3], M),
                         (True, True, True))
        self.assertEqual(test < UnitScalar([1.e3, 2.e3, 3.1e3], M),
                         (False, False, True))
        self.assertEqual(test > UnitScalar([1.e3, 1.9e3, 3.1e3], M),
                         (False, True, False))
        self.assertEqual(test >= UnitScalar([1.e3, 1.9e3, 3.1e3], M),
                         (True, True, False))
        self.assertEqual(test != UnitScalar([1.e3, 1.9e3, 3.1e3], M),
                         (False, True, True))

        self.assertEqual(test * 2.,   UnitScalar([ 2.,  4.,  6.], KM))
        self.assertEqual(test + 2.e5, UnitScalar([ 3.,  4.,  5.], KM))
        self.assertEqual(test - 1.e5, UnitScalar([ 0.,  1.,  2.], KM))
        self.assertEqual(test / 1.e5, UnitScalar([ 1.,  2.,  3.], CM))
        self.assertEqual(-test,       UnitScalar([-1., -2., -3.], KM))
        self.assertEqual(+test,       UnitScalar([ 1.,  2.,  3.], KM))

        test += UnitScalar(1000., M)
        self.assertEqual(test, UnitScalar([ 2., 3., 4.], KM))

        test *= -1.
        self.assertEqual(test, UnitScalar([-2.,-3.,-4.], KM))

        test *= UnitScalar([-1,2,-1], M)
        self.assertEqual(test, UnitScalar([2.,-6.,4.], KM*M))

        test /= M
        self.assertEqual(test, UnitScalar([2.,-6.,4.], KM))

        test += (-1.e5,8.e5,-1.e5)
        self.assertEqual(test, UnitScalar([1.,2.,3.], KM))

        test -= UnitScalar(1000.,M)
        self.assertEqual(test, UnitScalar([0.,1.,2.], KM))

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
