################################################################################
# oops/array_/empty.py: Empty subclass of class Array
#
# Modified 1/2/11 (MRS) -- Refactored Scalar.py to eliminate circular loading
#   dependencies with Array and Empty classes. Array.py, Scalar.py and Empty.py
#   are now separate files.
# Modified 2/8/12 (MRS) -- Refactored with a cleaner set of math operators.
################################################################################

import numpy as np

from oops.array_.array import Array

class Empty(Array):
    """An empty array, needed in some situations so the moral equivalent of a
    "None" can still respond to basic Array operations."""

    def __init__(self, vals=None, mask=False, units=None):

        self.shape = []
        self.rank  = 0
        self.item  = []
        self.vals  = np.array(0)
        self.mask  = False
        self.units = None
        self.subfields = {}
        self.subfield_math = False

    # Overrides of standard Array methods
    def __repr__(self): return "Empty()"

    def __str__(self): return "Empty()"

    def __nonzero__(self): return False

    def __getitem__(self, i): return self

    def __getslice__(self, i, j): return self

    # All arithmetic operations involving Empty return Empty

    def __pos__(self): return self

    def __neg__(self): return self

    def __abs__(self): return self

    def __add__(self, arg): return self

    def __sub__(self, arg): return self

    def __mul__(self, arg): return self

    def __div__(self, arg): return self

    def __mod__(self, arg): return self

    def __radd__(self, arg): return self

    def __rsub__(self, arg): return self

    def __rmul__(self, arg): return self

    def __rdiv__(self, arg): return self

    def __rmod__(self, arg): return self

    def __iadd__(self, arg): return self

    def __isub__(self, arg): return self

    def __imul__(self, arg): return self

    def __idiv__(self, arg): return self

    def __imod__(self, arg): return self

################################################################################
# Once defined, register with Array class
################################################################################

Array.EMPTY_CLASS = Empty

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Empty(unittest.TestCase):

    def runTest(self):

        from oops.array_.scalar import Scalar

        empty = Empty()
        self.assertEqual(empty, Empty())

        ints = Scalar((1,2,3))
        self.assertEqual(ints + empty, Empty())
        self.assertEqual(ints - empty, Empty())
        self.assertEqual(ints * empty, Empty())
        self.assertEqual(ints / empty, Empty())
        self.assertEqual(ints % empty, Empty())

        self.assertEqual(empty + ints, Empty())
        self.assertEqual(empty - ints, Empty())
        self.assertEqual(empty * ints, Empty())
        self.assertEqual(empty / ints, Empty())
        self.assertEqual(empty % ints, Empty())

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

        self.assertEqual(empty + 0, Empty())
        self.assertEqual(empty - 0, Empty())
        self.assertEqual(empty * 0, Empty())
        self.assertEqual(empty / 0, Empty())
        self.assertEqual(empty % 0, Empty())

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
