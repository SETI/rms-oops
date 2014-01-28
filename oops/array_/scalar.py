################################################################################
# oops/array_/scalar.py: Scalar subclass of class Array
#
# Modified 1/2/12 (MRS) -- Refactored Scalar.py to eliminate circular loading
#   dependencies with Array and Empty classes. Array.py, Scalar.py and Empty.py
#   are now separate files.
# Modified 2/8/12 (MRS) -- Supports array masks; includes new unit tests.
# 3/2/12 MRS: Integrated with VectorN and MatrixN.
# 6/7/12 MRS: Added the replace_zeros() method.
################################################################################

import numpy as np
import numpy.ma as ma

from oops.array_.array  import Array
from oops.array_.empty  import Empty
from oops.units import Units

class Scalar(Array):
    """An arbitrary Array of scalars."""

    def __init__(self, arg, mask=False, units=None):

        return Array.__init__(self, arg, mask, units, 0, item=None,
                                    floating=False, dimensionless=False)

    @staticmethod
    def as_scalar(arg):
        if isinstance(arg, Scalar): return arg
        if isinstance(arg, Units): return Scalar(1.,units=arg)
        return Scalar(arg)

    @staticmethod
    def as_float(arg, copy=False):
        """Convert to float if necessary; copy=True to return a new copy."""

        # If not a Scalar, convert it
        if not isinstance(arg, Scalar):
            arg = Scalar(arg)

        # If vals is a single value...
        if arg.shape == []:
            return Scalar(float(arg.vals), arg.mask, arg.units)

        # If vals is already an floating subtype...
        if np.issubdtype(arg.vals.dtype, np.core.numerictypes.floating):
            if copy:
                return arg.copy()
            return arg

        # Otherwise convert to floating
        return Scalar(arg.vals.astype("float"), arg.mask, arg.units)

    @staticmethod
    def as_int(arg, copy=False):
        """Convert to int if necessary; copy=True to return a new copy."""

        # If not a Scalar, convert it
        if not isinstance(arg, Scalar):
            arg = Scalar(arg)

        # If vals is a single value...
        if arg.shape == []:
            return Scalar(int(arg.vals), arg.mask, arg.units)

        # If vals is already an integer subtype...
        if np.issubdtype(arg.vals.dtype, np.core.numerictypes.integer):
            if copy:
                return arg.copy()
            return arg

        # Otherwise convert to integer
        return Scalar((arg.vals // 1).astype("int"), arg.mask, arg.units)

    @staticmethod
    def as_standard(arg):
        if not isinstance(arg, Scalar): arg = Scalar(arg)
        return arg.convert_units(None)

    def as_index(self):
        """Returns vals in a form suitable for indexing a numpy ndarray.
        """

        if not isinstance(self.vals, np.ndarray):
            return int(self.vals)
        elif np.issubdtype(self.vals.dtype, np.core.numerictypes.integer):
            return self.vals
        else:
            return (self.vals // 1).astype("int")

    def int(self):
        """Returns the integer (floor) component of each value."""

        return Scalar.as_int(self, copy=False)

    def frac(self):
        """Returns the fractional component of each value."""

        return Scalar(self.vals % 1, self.mask, self.units)

    def float(self):
        """Returns the same Scalar but containing floating-point values."""

        return Scalar.as_float(self, copy=False)

    def sin(self):
        """Returns the sine of each value. Works for units other than radians.
        """

        if (self.units is not None and self.units.triple != [0,0,1] and
                                       self.units.triple != [0,0,0]):
            raise ValueError("illegal units for sin(): " + units.name)

        return Scalar(np.sin(Scalar.as_standard(self).vals), self.mask)

    def cos(self):
        """Returns the cosine of each value. Works for units other than radians.
        """

        if (self.units is not None and self.units.triple != [0,0,1] and
                                       self.units.triple != [0,0,0]):
            raise ValueError("illegal units for cos(): " + units.name)

        return Scalar(np.cos(Scalar.as_standard(self).vals), self.mask)

    def tan(self):
        """Returns the tangent of each value. Works for units other than
        radians."""

        if (self.units is not None and self.units.triple != [0,0,1] and
                                       self.units.triple != [0,0,0]):
            raise ValueError("illegal units for tan(): " + units.name)

        return Scalar(np.tan(Scalar.as_standard(self).vals), self.mask)

    def arcsin(self):
        """Returns the arcsine of each value."""

        if self.units is not None and self.units.triple != [0,0,0]:
            raise ValueError("illegal units for arcsin(): " + units.name)

        new_mask = (self.vals < -1) | (self.vals > 1)
        if np.any(new_mask):
            values = self.vals.copy()
            values[new_mask] = 0.
            return Scalar(np.arcsin(values), self.mask | new_mask)
        else:
            return Scalar(np.arcsin(self.vals), self.mask)

    def arccos(self):
        """Returns the arccosine of each value."""

        if self.units is not None and self.units.triple != [0,0,0]:
            raise ValueError("illegal units for arccos(): " + units.name)

        new_mask = (self.vals < -1) | (self.vals > 1)
        if np.any(new_mask):
            values = self.vals.copy()
            values[new_mask] = 0.
            return Scalar(np.arccos(values), self.mask | new_mask)
        else:
            return Scalar(np.arccos(self.vals), self.mask)

    def arctan(self):
        """Returns the arctangent of each value."""

        if self.units is not None and self.units.triple != [0,0,0]:
            raise ValueError("illegal units for arctan(): " + units.name)

        return Scalar(np.arctan(self.vals), self.mask)

    def arctan2(self, arg):
        """Returns the four-quadrant value of arctan2(y,x)."""

        if self.units is not None and self.units.triple != [0,0,0]:
            raise ValueError("illegal units for arctan2(): " + units.name)

        arg = Scalar.as_scalar(arg)
        if arg.units is not None and arg.units.triple != [0,0,0]:
            raise ValueError("illegal units for arctan2(): " + units.name)

        return Scalar(np.arctan2(self.vals, arg.vals), self.mask)

    def sqrt(self):
        """Returns the square root, masking imaginary values."""

        if self.units is not None:
            new_units = self.units.sqrt()
        else:
            new_units = None

        if self.shape == []:
            if self.vals < 0.:
                return Scalar(0., True, new_units)
            else:
                return Scalar(np.sqrt(self.vals), self.mask, new_units)

        else:
            new_mask = (self.vals < 0.)
            if np.any(new_mask):
                new_vals = self.vals.copy()
                new_vals[new_mask] = 0.
                new_vals = np.sqrt(new_vals)
                return Scalar(new_vals, new_mask | self.mask, new_units)
            else:
                return Scalar(np.sqrt(self.vals), self.mask, new_units)

    def sign(self):
        """Returns the sign of each value as +1, -1 or 0."""

        return Scalar(np.sign(self.vals), self.mask)

    def max(self):
        """Returns the maximum of the unmasked values."""

        if np.shape(self.mask) == () and self.mask:
            return self.masked_version()

        if not np.any(self.mask):
            if self.units is None: return np.max(self.vals)
            return Scalar(np.max(self.vals), False, self.units)

        result = ma.max(self.mvals)
        if ma.is_masked(result):
            return self.masked_version()

        if self.units is None: return result
        return Scalar(result, False, self.units)

    def min(self):
        """Returns the minimum of the unmasked values."""

        if np.shape(self.mask) == () and self.mask:
            return self.masked_version()

        if not np.any(self.mask):
            if self.units is None: return np.min(self.vals)
            return Scalar(np.min(self.vals), False, self.units)

        result = ma.min(self.mvals)
        if ma.is_masked(result):
            return self.masked_version()

        if self.units is None: return result
        return Scalar(result, False, self.units)

    def clip(self, minval, maxval):
        """Returns a scalar in which values outside the specified range have
        been truncated."""

        if np.shape(self.mask) == ():
            if self.mask:
                return self.masked_version()
            else:
                return Scalar(np.clip(self.vals, minval, maxval), False,
                                                                  self.units)

        return Scalar(self.vals.clip(minval, maxval), self.mask, self.units)

    def mean(self):
        """Returns the mean of the unmasked values."""

        if np.shape(self.mask) == ():
            if self.mask:
                return self.masked_version()
            else:
                mean = np.mean(self.vals)
        else:
            mean = ma.mean(self.mvals)

        return Scalar(mean, self.units)

    def sum(self):
        """Returns the sum of the unmasked values."""

        if np.shape(self.mask) == ():
            if self.mask:
                return self.masked_version()
            else:
                sum = np.sum(self.vals)
        else:
            sum = ma.sum(self.mvals)

        return Scalar(sum, self.units)

    def is_between(self, range):
        """Returns a scalar of boolean values equal to True where every value
        of the scalar falls between limits defined by the first and second
        values of a Pair. Standard rules of broadcasting apply."""

        (minval, maxval) = Array.PAIR_CLASS.as_pair(range).as_scalars()
        between = (self >= minval & self <= maxval)

    def replace(self, mask, value=1., newmask=None):
        """Replaces masked entries with the given value and its optional mask.
        """

        if not np.any(mask): return

        if np.all(mask):    # replace everything...
            if isinstance(self.vals, np.ndarray):
                self.vals[...] = value
            else:
                self.vals = value

            if newmask is not None:
                self.mask = newmask

        else:               # replace selectively...
            self.vals[mask] = value
            if newmask is not None and newmask is not self.mask:
                self.expand_mask()
                self.mask[mask] = newmask

    def zero_mask(self):
        """Returns a boolean mask of zero-valued entries."""

        return (self.vals == 0.)

    def replace_zeros(self, value=1.):
        """Replaces zero-valued entries with the given value."""

        self.replace(self.zero_mask(), value)

    ####################################
    # Binary logical operators
    ####################################

    # (<) operator
    def __lt__(self, arg):
        arg = Scalar.as_scalar(arg).confirm_units(self.units)
        return Scalar._scalar_unless_shapeless(self.vals < arg.vals,
                                               self.mask | arg.mask)

    # (>) operator
    def __gt__(self, arg):
        arg = Scalar.as_scalar(arg).confirm_units(self.units)
        return Scalar._scalar_unless_shapeless(self.vals > arg.vals,
                                               self.mask | arg.mask)

    # (<=) operator
    def __le__(self, arg):
        arg = Scalar.as_scalar(arg).confirm_units(self.units)
        return Scalar._scalar_unless_shapeless(self.vals <= arg.vals,
                                               self.mask | arg.mask)

    # (>=) operator
    def __ge__(self, arg):
        arg = Scalar.as_scalar(arg).confirm_units(self.units)
        return Scalar._scalar_unless_shapeless(self.vals >= arg.vals,
                                               self.mask | arg.mask)

    # (~) operator
    def __invert__(self):
        return Scalar._scalar_unless_shapeless(np.logical_not(self.vals),
                                               self.mask)

    # (&) operator
    def __and__(self, arg):
        arg = Scalar.as_scalar(arg)
        return Scalar._scalar_unless_shapeless(self.vals & arg.vals,
                                               self.mask | arg.mask)

    # (|) operator
    def __or__(self, arg):
        arg = Scalar.as_scalar(arg)
        return Scalar._scalar_unless_shapeless(self.vals | arg.vals,
                                               self.mask | arg.mask)

    # (^) operator
    def __xor__(self, arg):
        arg = Scalar.as_scalar(arg)
        return Scalar._scalar_unless_shapeless(self.vals ^ arg.vals,
                                               self.mask | arg.mask)

    # This is needed for Scalars of booleans; it ensures that "if" tests execute
    # properly, because otherwise "if Scalar(False)" executes
    @staticmethod
    def _scalar_unless_shapeless(values, mask):
        if np.shape(values) == () and not mask:
            if isinstance(values, np.ndarray): return values[()]
            return values
        return Scalar(values, mask)

    ####################################
    # In-place binary logical operators
    ####################################

    # (&=) operator
    def __iand__(self, arg):
        arg = Scalar.as_scalar(arg)
        self.vals &= arg.vals
        self.mask |= arg.mask
        return self

    # (|=) operator
    def __ior__(self, arg):
        arg = Scalar.as_scalar(arg)
        self.vals |= arg.vals
        self.mask |= arg.mask
        return self

    # (^=) operator
    def __ixor__(self, arg):
        arg = Scalar.as_scalar(arg)
        self.vals ^= arg.vals
        self.mask |= arg.mask
        return self

    ####################################################
    # Scalar overrides of default binary operators
    ####################################################

    # Scalar multiply returns the type of the second operand if it is an Array;
    # otherwise, it returns a Scalar.
    def __mul__(self, arg):

        if isinstance(arg, Array):
            if type(arg) == Empty: return arg

            my_vals = self.vals
            if np.shape(my_vals) != () and arg.rank > 0:
                my_vals = my_vals.reshape(self.shape + arg.rank * [1])

            obj = Array.__new__(type(arg))
            obj.__init__(my_vals * arg.vals,
                         self.mask | arg.mask,
                         Units.mul_units(self.units, arg.units))

            obj.mul_subfields(self, arg)

            return obj

        return Scalar.as_scalar(arg) * self

    def __rmul__(self, arg):
        return self.__mul__(arg)

    def __imul__(self, arg):
        arg = Scalar.as_scalar(arg)

        self.vals *= arg.vals
        self.mask |= arg.mask
        self.units = Units.mul_units(self.units, arg.units)

        self.imul_subfields(arg)

        return self

    # Scalar right-divide returns the type of the second operand if it is an
    # Array; otherwise, it returns a Scalar.
    def __rdiv__(self, arg):

        if isinstance(arg, Array):
            if type(arg) == Empty: return arg

            my_vals = self.vals
            div_by_zero = (my_vals == 0)

            if np.shape(my_vals) == ():
                if div_by_zero:
                    my_vals = 1
            else:
                my_vals = my_vals.reshape(self.shape + arg.rank * [1])
                if np.any(div_by_zero):
                    my_vals = my_vals.copy()
                    my_vals[div_by_zero] = 1
                else:
                    div_by_zero = False

            obj = Array.__new__(type(arg))
            obj.__init__(arg.vals / my_vals,
                         self.mask | arg.mask | div_by_zero,
                         Units.div_units(arg.units, self.units))

            obj.div_subfields(arg, self)

            return obj

        return Scalar.as_scalar(arg) / self

    def __rmod__(self, arg):

        if isinstance(arg, Array):
            if type(arg) == Empty: return arg

            my_vals = self.vals
            div_by_zero = (my_vals == 0)

            if np.shape(my_vals) == ():
                if div_by_zero:
                    my_vals = 1
            else:
                my_vals = my_vals.reshape(self.shape + arg.rank * [1])
                if np.any(div_by_zero):
                    my_vals = my_vals.copy()
                    my_vals[div_by_zero] = 1
                else:
                    div_by_zero = False

            obj = Array.__new__(type(arg))
            obj.__init__(arg.vals % my_vals,
                         self.mask | arg.mask | div_by_zero,
                         Units.div_units(arg.units, self.units))

            # This operator does not preserve subfields

            return obj

        return Scalar.as_scalar(arg) % self

################################################################################
# Once the load is complete, we can fill in a reference to the Scalar class
# inside the Array object.
################################################################################

Array.SCALAR_CLASS = Scalar

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Scalar(unittest.TestCase):

    def runTest(self):

        # Arithmetic operations
        ints = Scalar((1,2,3))
        test = Scalar(np.array([1,2,3]))
        self.assertEqual(ints, test)

        test = Scalar(test)
        self.assertEqual(ints, test)

        self.assertEqual(ints, (1,2,3))
        self.assertEqual(ints, [1,2,3])

        self.assertEqual(ints.shape, [3])

        self.assertEqual(-ints, [-1,-2,-3])
        self.assertEqual(+ints, [1,2,3])

        self.assertEqual(ints, abs(ints))
        self.assertEqual(ints, abs(Scalar(( 1, 2, 3))))
        self.assertEqual(ints, abs(Scalar((-1,-2,-3))))

        self.assertEqual(ints * 2, [2,4,6])
        self.assertEqual(ints / 2., [0.5,1,1.5])
        self.assertEqual(ints / 2, [0,1,1])
        self.assertEqual(ints + 1, [2,3,4])
        self.assertEqual(ints - 0.5, (0.5,1.5,2.5))
        self.assertEqual(ints % 2, (1,0,1))

        self.assertEqual(ints + Scalar([1,2,3]), [2,4,6])
        self.assertEqual(ints - Scalar((1,2,3)), [0,0,0])
        self.assertEqual(ints * [1,2,3], [1,4,9])
        self.assertEqual(ints / [1,2,3], [1,1,1])
        self.assertEqual(ints % [1,3,3], [0,2,0])

        self.assertRaises(ValueError, ints.__add__, (4,5))
        self.assertRaises(ValueError, ints.__sub__, (4,5))
        self.assertRaises(ValueError, ints.__mul__, (4,5))
        self.assertRaises(ValueError, ints.__div__, (4,5))
        self.assertRaises(ValueError, ints.__mod__, (4,5))

        self.assertRaises(ValueError, ints.__add__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__sub__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__mul__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__div__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__mod__, Scalar((4,5)))

        ints += 1
        self.assertEqual(ints, [2,3,4])

        ints -= 1
        self.assertEqual(ints, [1,2,3])

        ints *= 2
        self.assertEqual(ints, [2,4,6])

        ints /= 2
        self.assertEqual(ints, [1,2,3])

        ints *= (3,2,1)
        self.assertEqual(ints, [3,4,3])

        ints /= (1,2,3)
        self.assertEqual(ints, [3,2,1])

        ints += (1,2,3)
        self.assertEqual(ints, 4)
        self.assertEqual(ints, [4])
        self.assertEqual(ints, [4,4,4])
        self.assertEqual(ints, Scalar([4,4,4]))

        ints -= (3,2,1)
        self.assertEqual(ints, [1,2,3])

        test = Scalar((10,10,10))
        test %= 4
        self.assertEqual(test, 2)

        test = Scalar((10,10,10))
        test %= (4,3,2)
        self.assertEqual(test, [2,1,0])

        test = Scalar((10,10,10))
        test %= Scalar((5,4,3))
        self.assertEqual(test, [0,2,1])

        self.assertRaises(ValueError, ints.__iadd__, (4,5))
        self.assertRaises(ValueError, ints.__isub__, (4,5))
        self.assertRaises(ValueError, ints.__imul__, (4,5))
        self.assertRaises(ValueError, ints.__idiv__, (4,5))
        self.assertRaises(ValueError, ints.__imod__, (4,5))

        self.assertRaises(ValueError, ints.__iadd__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__isub__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__imul__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__idiv__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__imod__, Scalar((4,5)))

        # Logical operations
        self.assertEqual(ints > 1,  [False, True,  True ])
        self.assertEqual(ints < 2,  [True,  False, False])
        self.assertEqual(ints >= 1, [True,  True,  True ])
        self.assertEqual(ints <= 2, [True,  True,  False])
        self.assertEqual(ints != 1, [False, True,  True ])
        self.assertEqual(ints == 2, [False, True,  False])

        self.assertEqual(ints >  Scalar(1),  [False, True,  True ])
        self.assertEqual(ints <  Scalar(2),  [True,  False, False])
        self.assertEqual(ints >= Scalar((1,4,3)), [True,  False, True ])
        self.assertEqual(ints <= Scalar((2,2,1)), [True,  True,  False])
        self.assertEqual(ints != [2,2,3], [True,  False, False])
        self.assertEqual(ints == (3,2,1), [False, True,  False])


        self.assertEqual(ints == (4,5), False)
        self.assertEqual(ints != (4,5), True)
        self.assertRaises(ValueError, ints.__gt__, (4,5))
        self.assertRaises(ValueError, ints.__lt__, (4,5))
        self.assertRaises(ValueError, ints.__ge__, (4,5))
        self.assertRaises(ValueError, ints.__le__, (4,5))

        self.assertRaises(ValueError, ints.__gt__, Scalar([4,5]))
        self.assertRaises(ValueError, ints.__lt__, Scalar([4,5]))
        self.assertRaises(ValueError, ints.__ge__, Scalar([4,5]))
        self.assertRaises(ValueError, ints.__le__, Scalar([4,5]))

        bools = Scalar(([True, True, True],[False, False, False]))
        self.assertEqual(bools.shape, [2,3])
        self.assertEqual(bools[0], True)
        self.assertEqual(bools[1], False)
        self.assertEqual(bools[:,0], (True, False))
        self.assertEqual(bools[:,:].swapaxes(0,1), (True, False))
        self.assertEqual(bools[:].swapaxes(0,1),   (True, False))
        self.assertEqual(bools.swapaxes(0,1),      (True, False))

        self.assertEqual(~bools.swapaxes(0,1), (False, True))
        self.assertEqual(~bools, ((False, False, False), (True, True, True)))
        self.assertEqual(~bools, Scalar([(False, False, False),
                                         (True,  True,  True )]))

        self.assertEqual(bools & True,  bools)
        self.assertEqual(bools & False, False)
        self.assertEqual(bools & (True,False,True), [[True, False,True ],
                                                     [False,False,False]])
        self.assertEqual(bools & Scalar(True),  bools)
        self.assertEqual(bools & Scalar(False), False)
        self.assertEqual(bools & Scalar((True,False,True)),[[True, False,True ],
                                                           [False,False,False]])

        self.assertEqual(bools | True,  True)
        self.assertEqual(bools | False, bools)
        self.assertEqual(bools | (True,False,True), [[True, True, True ],
                                                     [True, False,True ]])
        self.assertEqual(bools | Scalar(True),  True)
        self.assertEqual(bools | Scalar(False), bools)
        self.assertEqual(bools | Scalar((True,False,True)),[[True, True, True ],
                                                           [True, False,True ]])

        self.assertEqual((bools ^ True).swapaxes(0,1), (False,True))
        self.assertEqual(bools ^ True, ~bools)
        self.assertEqual(bools ^ False, bools)
        self.assertEqual(bools ^ (True,False,True), [[False,True, False],
                                                     [True, False,True ]])
        self.assertEqual((bools ^ Scalar(True)).swapaxes(0,1), (False,True))
        self.assertEqual( bools ^ Scalar(True), ~bools)
        self.assertEqual( bools ^ Scalar(False), bools)
        self.assertEqual( bools ^ Scalar((True,False,True)),[[False,True,False],
                                                            [True,False,True ]])

        self.assertEqual(bools == Scalar([True,True]), False)
        self.assertEqual(bools != Scalar([True,True]), True)

        bools &= bools
        self.assertEqual(bools, [[True, True, True],[False, False, False]])

        bools |= bools
        self.assertEqual(bools, [[True, True, True],[False, False, False]])

        test = bools.copy().swapaxes(0,1)
        test |= test
        self.assertEqual(test, [[True, False],[True, False],[True, False]])

        test ^= bools.swapaxes(0,1)
        self.assertEqual(test,  False)
        self.assertEqual(test,  (False,False))

#         test[0] = True
#         self.assertEqual(test, [[True, True],[False, False],[False, False]])
# 
#         test[1:,1] ^= True
#         self.assertEqual(test, [[True, True],[False, True],[False, True]])
# 
#         test[1:,0] |= test[1:,1]
#         self.assertEqual(test, True)

        self.assertRaises(ValueError, bools.__ior__,  (True, False))
        self.assertRaises(ValueError, bools.__iand__, (True, False))
        self.assertRaises(ValueError, bools.__ixor__, (True, False))

        self.assertRaises(ValueError, bools.__ior__,  Scalar((True, False)))
        self.assertRaises(ValueError, bools.__iand__, Scalar((True, False)))
        self.assertRaises(ValueError, bools.__ixor__, Scalar((True, False)))

        # Generic Array operations
        self.assertEqual(ints[0], 1)

        floats = ints.astype("float")
        self.assertEqual(floats[0], 1.)

        strings = ints.astype("string")
        self.assertEqual(strings[1], "2")

        six = Scalar([1,2,3,4,5,6])
        self.assertEqual(six.shape, [6])

        test = six.copy().reshape((3,1,2))
        self.assertEqual(test.shape, [3,1,2])
        self.assertEqual(test, [[[1,2]],[[3,4]],[[5,6]]])
        self.assertEqual(test.swapaxes(0,1).shape, [1,3,2])
        self.assertEqual(test.swapaxes(0,2).shape, [2,1,3])
        self.assertEqual(test.ravel().shape, [6])
        self.assertEqual(test.flatten().shape, [6])

        four = Scalar([1,2,3,4]).reshape((2,2))
        self.assertEqual(four, [[1,2],[3,4]])

        self.assertEqual(Array.broadcast_shape((four,test)), [3,2,2])
        self.assertEqual(four.rebroadcast((3,2,2)), [[[1,2],[3,4]],
                                                     [[1,2],[3,4]],
                                                     [[1,2],[3,4]]])
        self.assertEqual(test.rebroadcast((3,2,2)), [[[1,2],[1,2]],
                                                     [[3,4],[3,4]],
                                                     [[5,6],[5,6]]])

        ten = four + test
        self.assertEqual(ten.shape, [3,2,2])
        self.assertEqual(ten, [[[2, 4], [4, 6]],
                               [[4, 6], [6, 8]],
                               [[6, 8], [8,10]]])

        x24 = four * test
        self.assertEqual(x24.shape, [3,2,2])
        self.assertEqual(x24, [[[1, 4], [ 3, 8]],
                               [[3, 8], [ 9,16]],
                               [[5,12], [15,24]]])

        # Mask tests 2/1/12 (MRS)
        test = Scalar(range(6))
        self.assertEqual(str(test), "Scalar[0 1 2 3 4 5]")

        test.mask = np.array(3*[True] + 3*[False])
        self.assertEqual(str(test),   "Scalar[-- -- -- 3 4 5, mask]")
        self.assertEqual(str(test+1), "Scalar[-- -- -- 4 5 6, mask]")
        self.assertEqual(str(test-2), "Scalar[-- -- -- 1 2 3, mask]")
        self.assertEqual(str(test*2), "Scalar[-- -- -- 6 8 10, mask]")
        self.assertEqual(str(test/2), "Scalar[-- -- -- 1 2 2, mask]")
        self.assertEqual(str(test%2), "Scalar[-- -- -- 1 0 1, mask]")

        self.assertEqual(str(test-2.), "Scalar[-- -- -- 1.0 2.0 3.0, mask]")
        self.assertEqual(str(test+2.), "Scalar[-- -- -- 5.0 6.0 7.0, mask]")
        self.assertEqual(str(test*2.), "Scalar[-- -- -- 6.0 8.0 10.0, mask]")
        self.assertEqual(str(test/2.), "Scalar[-- -- -- 1.5 2.0 2.5, mask]")

        self.assertEqual(str(test + [1, 2, 3, 4, 5, 6]),
                         "Scalar[-- -- -- 7 9 11, mask]")
        self.assertEqual(str(test - [1, 2, 3, 4, 5, 6]),
                         "Scalar[-- -- -- -1 -1 -1, mask]")
        self.assertEqual(str(test * [1, 2, 3, 4, 5, 6]),
                         "Scalar[-- -- -- 12 20 30, mask]")
        self.assertEqual(str(test / [1, 7, 5, 1, 2, 1]),
                         "Scalar[-- -- -- 3 2 5, mask]")
        self.assertEqual(str(test / [0, 7, 5, 1, 2, 0]),
                         "Scalar[-- -- -- 3 2 --, mask]")
        self.assertEqual(str(test % [0, 7, 5, 1, 2, 0]),
                         "Scalar[-- -- -- 0 0 --, mask]")

        temp = Scalar(6*[1], 5*[False] + [True])
        self.assertEqual(str(temp), "Scalar[1 1 1 1 1 --, mask]")


        self.assertEqual(str(test + temp), "Scalar[-- -- -- 4 5 --, mask]")

        foo = test + temp
        self.assertTrue(foo.vals[0] == test.vals[0] + temp.vals[0])

        foo.vals[0] = 99
        self.assertFalse(foo.vals[0] == test.vals[0] + temp.vals[0])

        self.assertEqual(foo, test + temp)

        bar = Scalar(1*[False] + 5*[True])
        self.assertFalse(bar)

        bar.mask = np.array(1*[True] + 5*[False])   # Mask out the False value
        self.assertTrue(bar)

        # This behavior has been redefined to raise a ValueError
        # bar.mask = np.array(6*[True])               # Mask out every value
        # self.assertTrue(bar)

        self.assertEqual(str(test), "Scalar[-- -- -- 3 4 5, mask]")

        self.assertEqual(test[5],  5)
        self.assertEqual(test[-1], 5)
        self.assertEqual(test[3:], [3,4,5])
        self.assertEqual(test[3:5], [3,4])
        self.assertEqual(test[3:-1], [3,4])

        self.assertEqual(test[0], Scalar(0, True))

        self.assertEqual(str(test[0]), "Scalar(--, mask)")
        self.assertEqual(str(test[0:4]), "Scalar[-- -- -- 3, mask]")
        self.assertEqual(str(test[0:1]), "Scalar[--, mask]")
        self.assertEqual(str(test[5]), "5")
        self.assertEqual(str(test[4:]), "Scalar[4 5]")
        self.assertEqual(str(test[5:]), "Scalar[5]")
        self.assertEqual(str(test[0:6:2]), "Scalar[-- -- 4, mask]")

        mvals = test.mvals
        self.assertEqual(type(mvals), ma.MaskedArray)
        self.assertEqual(str(mvals), "[-- -- -- 3 4 5]")

        temp = Scalar(range(6))
        mvals = temp.mvals
        self.assertEqual(type(mvals), ma.MaskedArray)
        self.assertEqual(str(mvals), "[0 1 2 3 4 5]")
        self.assertEqual(mvals.mask, ma.nomask)

        temp.mask = True
        self.assertEqual(str(temp), "Scalar[-- -- -- -- -- --, mask]")

        mvals = temp.mvals
        self.assertEqual(type(mvals), ma.MaskedArray)
        self.assertEqual(str(mvals), "[-- -- -- -- -- --]")

        # Units tests 2/7/12 (MRS)
        test = Scalar(range(6))
        self.assertEqual(test, np.arange(6))
        eps = 1.e-7

        cm = test.convert_units(Units.CM)
        self.assertEqual(cm, Scalar(np.arange(6)*100000, units=Units.CM))
        self.assertEqual(cm, Scalar(np.arange(6)*1000,   units=Units.M))
        self.assertEqual(cm, Scalar(np.arange(6),        units=Units.KM))

        km = cm.convert_units(Units.KM)
        self.assertEqual(cm, km)

        self.assertTrue(cm.attach_units(None) < (Scalar(np.arange(6)*1.e5)+eps))
        self.assertTrue(cm.attach_units(None) > (Scalar(np.arange(6)*1.e5)-eps))

        self.assertTrue(cm.convert_units(None) < (Scalar(np.arange(6)) + eps))
        self.assertTrue(cm.convert_units(None) > (Scalar(np.arange(6)) - eps))

        self.assertTrue(km.attach_units(None) < (Scalar(np.arange(6)) + eps))
        self.assertTrue(km.attach_units(None) > (Scalar(np.arange(6)) - eps))

        self.assertTrue(km.attach_units(None) < (Scalar(np.arange(6)) + eps))
        self.assertTrue(km.attach_units(None) > (Scalar(np.arange(6)) - eps))

        self.assertTrue(km.convert_units(None) < (Scalar(np.arange(6)) + eps))
        self.assertTrue(km.convert_units(None) > (Scalar(np.arange(6)) - eps))

        self.assertTrue(km.confirm_units(Units.KM) < (Scalar(np.arange(6),
                                                        units=Units.KM) + eps))
        self.assertTrue(km.confirm_units(Units.KM) > (Scalar(np.arange(6),
                                                        units=Units.KM) - eps))

        self.assertTrue(km.confirm_units(Units.CM) < (Scalar(np.arange(6)*1.e5,
                                                        units=Units.CM) + eps))
        self.assertTrue(km.confirm_units(Units.CM) > (Scalar(np.arange(6)*1.e5,
                                                        units=Units.CM) - eps))

        self.assertTrue(km.confirm_units(Units.CM) < (Scalar(np.arange(6),
                                                        units=Units.KM) + eps))
        self.assertTrue(km.confirm_units(Units.CM) > (Scalar(np.arange(6),
                                                        units=Units.KM) - eps))

        self.assertRaises(ValueError, km.confirm_units, None)
        self.assertRaises(ValueError, km.confirm_units, Units.DEG)

        self.assertTrue((km + 1) < (Scalar(np.arange(1,7),
                                           units=Units.KM) + eps))
        self.assertTrue((km + 1) > (Scalar(np.arange(1,7),
                                           units=Units.KM) - eps))

        self.assertTrue((km + 1) <= (Scalar(np.arange(1,7)*1.e5,
                                            units=Units.CM) + eps))
        self.assertTrue((km + 1) >= (Scalar(np.arange(1,7)*1.e5,
                                            units=Units.CM) - eps))

        self.assertTrue((cm + 1) < (Scalar(np.arange(6)*100000 + 1,
                                           units=Units.CM) + eps))
        self.assertTrue((cm + 1) > (Scalar(np.arange(6)*100000 + 1,
                                           units=Units.CM) - eps))

        self.assertTrue((cm + 1.e5) < (Scalar(np.arange(1,7),
                                              units=Units.KM) + eps))
        self.assertTrue((cm + 1.e5) > (Scalar(np.arange(1,7),
                                              units=Units.KM) - eps))

        self.assertTrue((cm + km[1]) < (Scalar(np.arange(1,7),
                                               units=Units.KM) + eps))
        self.assertTrue((cm + km[1]) > (Scalar(np.arange(1,7),
                                               units=Units.KM) - eps))

        self.assertTrue((cm + km) < (Scalar(np.arange(0,12,2)*100000,
                                            units=Units.CM) + eps))
        self.assertTrue((cm + km) > (Scalar(np.arange(0,12,2)*100000,
                                            units=Units.CM) - eps))

        self.assertTrue((cm + km) < (Scalar(np.arange(0,12,2),
                                            units=Units.KM) + eps))
        self.assertTrue((cm + km) > (Scalar(np.arange(0,12,2),
                                            units=Units.KM) - eps))

        self.assertEqual((cm + km).units, Units.CM)
        self.assertEqual((km + cm).units, Units.KM)

        self.assertRaises(ValueError, cm.__lt__, test)
        self.assertRaises(ValueError, cm.__lt__, 1)

        self.assertEquals(test * Units.KM, km)
        self.assertEquals(Units.KM * test, km)

        self.assertTrue(Scalar.as_standard(cm) < Scalar(range(6)) + eps)
        self.assertTrue(Scalar.as_standard(cm) > Scalar(range(6)) - eps)
        self.assertEqual(Scalar.as_standard(cm).units, None)

        # sqrt tests 2/17/12 (MRS)
        a = Scalar(np.arange(10)**2)
        b = a.sqrt()
        self.assertEqual(b, np.arange(10))

        self.assertEqual(Scalar(4.).sqrt(), 2)

        self.assertTrue(Scalar(-1.).sqrt().mask)

        self.assertTrue(Scalar((-1,1,2)).sqrt().mask[0])
        self.assertFalse(Scalar((-1,1,2)).sqrt().mask[1])
        self.assertFalse(Scalar((-1,1,2)).sqrt().mask[2])

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
