################################################################################
# Scalar
#
# Modified 1/2/12 (MRS) -- Refactored Scalar.py to eliminate circular loading
#   dependencies with Array and Empty classes. Array.py, Scalar.py and Empty.py
#   are now separate files.
# Modified 2/8/12 (MRS) -- Supports array masks; includes new unit tests.
################################################################################

import numpy as np
import numpy.ma as ma

from baseclass  import Array
from oops.units import Units

class Scalar(Array):
    """An arbitrary Array of scalars."""

    def __init__(self, arg, mask=False, units=None):

        if mask is not False: mask = np.asarray(mask)

        if isinstance(arg, Scalar):
            mask = mask | arg.mask
            if units is None:
                units = arg.units
                arg = arg.vals
            elif arg.units is not None:
                arg = arg.units.convert(arg.vals, units)
            else:
                arg = arg.vals

        elif isinstance(arg, Array):
            raise ValueError("class " + type(arg).__name__ +
                             " cannot be converted to class " +
                             type(self).__name__)

        elif isinstance(arg, ma.MaskedArray):
            if arg.mask != ma.nomask: mask = mask | arg.mask
            arg = arg.data

        if np.shape(arg) == ():
            self.vals  = arg
            self.rank  = 0
            self.item  = []
            self.shape = []
        else:
            self.vals  = np.asarray(arg)
            self.rank  = 0
            self.item  = []
            self.shape = list(self.vals.shape)

        self.mask = mask
        if (self.mask is not False) and (list(self.mask.shape) != self.shape):
            raise ValueError("mask array is incompatible with Scalar shape")

        self.units = Units.as_units(units)

        return

    @staticmethod
    def as_scalar(arg):
        if isinstance(arg, Scalar): return arg
        return Scalar(arg)

    @staticmethod
    def as_float_scalar(arg):
        if isinstance(arg, Scalar) and arg.vals.dtype == np.dtype("float"):
            return arg

        return Scalar.as_scalar(arg) * 1.

    @staticmethod
    def as_standard(arg):
        if not isinstance(arg, Scalar): arg = Scalar(arg)
        return arg.convert_units(None)

    def int(self):
        """Returns the integer (floor) component of each value."""

        return Scalar((self.vals // 1.).astype("int"), self.mask)

    def frac(self):
        """Returns the fractional component of each value."""

        return Scalar(self.vals % 1., self.mask)

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
        return Scalar._scalar_unless_shapeless(~self.vals, self.mask)

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
        if np.shape(values) == () and not mask: return values
        return Scalar(values)

    ####################################
    # In-place binary logical operators
    ####################################

    # (&=) operator
    def __iand__(self, arg):
        try:
            arg = Scalar.as_scalar(arg)
        except:
            self.raise_type_mismatch("&=", arg)

        try:
            self.vals &= arg.vals
            self.mask |= arg.mask
            return self
        except:
            self.raise_shape_mismatch("&=", arg.vals)

    # (|=) operator
    def __ior__(self, arg):
        try:
            arg = Scalar.as_scalar(arg)
        except:
            self.raise_type_mismatch("|=", arg)

        try:
            self.vals |= arg.vals
            self.mask |= arg.mask
            return self
        except:
            self.raise_shape_mismatch("|=", arg.vals)


    # (^=) operator
    def __ixor__(self, arg):
        try:
            arg = Scalar.as_scalar(arg)
        except:
            self.raise_type_mismatch("^=", arg)

        try:
            self.vals ^= arg.vals
            self.mask |= arg.mask
            return self
        except:
            self.raise_shape_mismatch("^=", arg.vals)

################################################################################
# Once the load is complete, we can fill in a reference to the Scalar class
# inside the Array object.
################################################################################

Array.SCALAR_CLASS = Scalar

################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.units import Units

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

        test[0] = True
        self.assertEqual(test, [[True, True],[False, False],[False, False]])

        test[1:,1] ^= True
        self.assertEqual(test, [[True, True],[False, True],[False, True]])

        test[1:,0] |= test[1:,1]
        self.assertEqual(test, True)

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

        bar.mask = np.array(6*[True])               # Mask out every value
        self.assertTrue(bar)

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

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
