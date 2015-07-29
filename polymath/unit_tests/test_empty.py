################################################################################
# Tests for subclass Empty
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Empty, Scalar, Boolean

class Test_Empty(unittest.TestCase):

    def runTest(self):

        # Arithmetic operations
        empty = Empty()
        self.assertEqual(empty, Empty())

        ints = Scalar((1,2,3))
        self.assertEqual(empty + ints, Empty())
        self.assertEqual(empty - ints, Empty())
        self.assertEqual(empty * ints, Empty())
        self.assertEqual(empty / ints, Empty())
        self.assertEqual(empty % ints, Empty())

        self.assertEqual(ints + empty, Empty())
        self.assertEqual(ints - empty, Empty())
        self.assertEqual(ints * empty, Empty())
        self.assertEqual(ints / empty, Empty())
        self.assertEqual(ints % empty, Empty())

        empty += ints
        self.assertEqual(empty, Empty())

        empty -= ints
        self.assertEqual(empty, Empty())

        empty *= ints
        self.assertEqual(empty, Empty())

        empty /= ints
        self.assertEqual(empty, Empty())

        empty %= ints
        self.assertEqual(empty, Empty())

        test = Scalar((1,2,3))
        test += empty
        self.assertEqual(test, Empty())

        test = Scalar((1,2,3))
        test -= empty
        self.assertEqual(test, Empty())

        test = Scalar((1,2,3))
        test *= empty
        self.assertEqual(test, Empty())

        test = Scalar((1,2,3))
        test /= empty
        self.assertEqual(test, Empty())

        test = Scalar((1,2,3))
        test //= empty
        self.assertEqual(test, Empty())

        test = Scalar((1,2,3))
        test %= empty
        self.assertEqual(test, Empty())

        self.assertEqual(empty + 0, Empty())
        self.assertEqual(empty - 0, Empty())
        self.assertEqual(empty * 0, Empty())
        self.assertEqual(empty / 0, Empty())
        self.assertEqual(empty % 0, Empty())

        self.assertEqual(0 + empty, Empty())
        self.assertEqual(0 - empty, Empty())
        self.assertEqual(0 * empty, Empty())
        self.assertEqual(0 / empty, Empty())
        self.assertEqual(0 % empty, Empty())

        # Logical operations
        self.assertEqual(~empty, Empty())

        bool = Boolean((True,False))
        self.assertEqual(empty & bool, Empty())
        self.assertEqual(empty | bool, Empty())
        self.assertEqual(empty ^ bool, Empty())

        self.assertEqual(bool & empty, Empty())
        self.assertEqual(bool | empty, Empty())
        self.assertEqual(bool ^ empty, Empty())

        empty &= bool
        self.assertEqual(empty, Empty())

        empty |= bool
        self.assertEqual(empty, Empty())

        empty ^= bool
        self.assertEqual(empty, Empty())

        test = Boolean((True,False))
        test &= empty
        self.assertEqual(test, Empty())

        test = Boolean((True,False))
        test |= empty
        self.assertEqual(test, Empty())

        test = Boolean((True,False))
        test ^= empty
        self.assertEqual(test, Empty())

        self.assertEqual(str(empty), 'Empty()')

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
