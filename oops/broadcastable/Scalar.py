################################################################################
# Scalar
#
# Modified 1/2/12 (MRS) -- Refactored Scalar.py to eliminate circular loading
#   dependencies with Array and Empty classes. Array.py, Scalar.py and Empty.py
#   are now separate files.
################################################################################

import numpy as np
import unittest

from oops.broadcastable.Array import Array

class Scalar(Array):
    """An arbitrary Array of scalars."""

    OOPS_CLASS = "Scalar"

    def __init__(self, arg):

        if isinstance(arg, Scalar): arg = arg.vals

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

    def int(self):
        """Returns the integer (floor) component of each value."""

        return Scalar((self.vals // 1.).astype("int"))

    def frac(self):
        """Returns the fractional component of each value."""

        return Scalar(self.vals - np.floor(self.vals))

    # abs() operator
    def __abs__(self):
        return Scalar(np.abs(self.vals))

    # (%) operator
    def __mod__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar(self.vals % arg)

    # (%=) operator
    def __imod__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        self.vals %= arg
        return self

    # (<) operator
    def __lt__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals < arg)

    # (>) operator
    def __gt__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals > arg)

    # (<=) operator
    def __le__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals <= arg)

    # (>=) operator
    def __ge__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals >= arg)

    # (~) operator
    def __invert__(self):
        return Scalar._scalar_unless_shapeless(~self.vals)

    # (&) operator
    def __and__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals & arg)

    # (|) operator
    def __or__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals | arg)

    # (^) operator
    def __xor__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals ^ arg)

    # (&=) operator
    def __iand__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        self.vals &= arg
        return self

    # (|=) operator
    def __ior__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        self.vals |= arg
        return self

    # (^=) operator
    def __ixor__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        self.vals ^= arg
        return self

    # This is needed for Scalars of booleans; it ensures that "if" tests execute
    # properly, because otherwise "if Scalar(False)" executes
    @staticmethod
    def _scalar_unless_shapeless(values):
        if np.shape(values) == (): return values
        return Scalar(values)

########################################
# UNIT TESTS
########################################

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

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
