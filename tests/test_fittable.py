##########################################################################################
# tests/test_fittable.py
##########################################################################################

import unittest

from oops.fittable import Fittable
import oops.mutable as mutable


class A(Fittable):
    nparams = 1

    def __init__(self, x):
        self.x = x
        self._refresh()

    def _refresh(self):
        self.x_squared = self.x**2

    def _set_params(self, params):
        self.x = params[0]

    @property
    def params(self):
        return (self.x,)

class B:
    def __init__(self, x, a):
        self.x = x
        self.a = a
        self._refresh()

    def _refresh(self):
        self.x_plus_a2 = self.x + self.a.x_squared


class C(Fittable):
    nparams = 1

    def __init__(self, x, a):
        self.x = x
        self.a = a
        self.c = self
        self._refresh()

    def _set_params(self, params):
        self.x = params[0]

    @property
    def params(self):
        return (self.x,)

    def _refresh(self):
        self.x_plus_a2_plus_cx_plus_ccx = (self.x + self.a.x_squared + self.c.x
                                           + self.c.c.x)


class D:
    def __init__(self, x):
        self.x = x


class Test_Fittable(unittest.TestCase):

    def runTest(self):

        x = ()
        self.assertFalse(mutable.is_fittable(x))
        self.assertEqual(mutable.mutable_names(x), [])
        self.assertEqual(mutable.version(x), 0)

        a = A(7)
        self.assertEqual(a.x_squared, 49)
        self.assertEqual(a.params, (7,))
        self.assertEqual(a.version, 0)
        self.assertIsInstance(a.params, tuple)
        self.assertTrue(mutable.is_fittable(a))
        self.assertTrue(mutable.is_mutable(a))
        self.assertEqual(mutable.mutable_names(a), [])
        self.assertEqual(mutable.version(a), 0)

        a.set_params([5])
        self.assertEqual(a.params, (5,))
        self.assertEqual(a.x_squared, 25)
        self.assertGreater(a.version, 0)
        self.assertEqual(mutable.mutable_names(a), [])
        self.assertTrue(mutable.is_fittable(a))

        b = B(1, a)
        mutable.set_param_order(b, 'a')
        self.assertEqual(b.x_plus_a2, 26)
        self.assertEqual(mutable.get_params(b), (5.,))
        self.assertIsInstance(mutable.get_params(b)[0], float)
        self.assertFalse(mutable.is_fittable(b))
        self.assertTrue(mutable.is_mutable(b))

        mutable.set_params(b, 7)
        self.assertEqual(b.x_plus_a2, 50)
        self.assertEqual(mutable.mutable_names(b), ['a'])
        self.assertEqual(mutable.unfrozen_names(b), ['a'])

        mutable.freeze(a)
        self.assertEqual(mutable.mutable_names(b), ['a'])
        self.assertEqual(mutable.unfrozen_names(b), [])

        a = A(5)
        c = C(1, a)
        self.assertEqual(mutable.mutable_names(c), ['a'])
        self.assertTrue(mutable.is_fittable(c))
        self.assertEqual(c.x_plus_a2_plus_cx_plus_ccx, 28)

        a.set_params([6])
        self.assertTrue(c.refresh())
        self.assertEqual(c.x_plus_a2_plus_cx_plus_ccx, 39)
        self.assertFalse(c.refresh())
        self.assertFalse(c.refresh())

        a = A(5)
        c = C(1, a)
        self.assertEqual(c.x_plus_a2_plus_cx_plus_ccx, 28)

        mutable.set_param_order(c, ['', 'a'])
        self.assertEqual(mutable.get_param_order(c), ['', 'a'])
        self.assertEqual(mutable.get_nparams(c), 2)
        self.assertEqual(mutable.get_params(c), (1,5))

        mutable.set_params(c, [1, 6])
        self.assertEqual(c.x_plus_a2_plus_cx_plus_ccx, 39)
        self.assertFalse(c.refresh())
        self.assertEqual(mutable.get_params(a), (6,))
        self.assertEqual(mutable.get_params(c), (1,6))

        self.assertFalse(mutable.is_frozen(a))
        self.assertFalse(mutable.is_frozen(c))
        self.assertFalse(a.is_frozen)
        self.assertFalse(c.is_frozen)

        mutable.freeze(a)
        self.assertTrue(mutable.is_frozen(a))
        self.assertFalse(mutable.is_frozen(c))
        self.assertRaises(ValueError, mutable.set_params, a, 2)

        self.assertEqual(mutable.mutable_names(c), ['a'])
        self.assertEqual(mutable.unfrozen_names(c), [])
        self.assertTrue(mutable.is_fittable(a))
        self.assertTrue(mutable.is_fittable(c))
        self.assertEqual(mutable.get_params(c), (1,6))
        self.assertEqual(c.params, (1,))

        a = A(5)
        c = C(1, a)
        mutable.freeze(c)
        self.assertTrue(mutable.is_frozen(a))
        self.assertTrue(mutable.is_frozen(c))
        self.assertTrue(a.is_frozen)
        self.assertTrue(c.is_frozen)
        self.assertRaises(ValueError, mutable.set_params, a, 2)
        self.assertRaises(ValueError, mutable.set_params, c, 2)

        self.assertEqual(mutable.unfrozen_names(c), [])
        self.assertEqual(mutable.mutable_names(c), ['a'])
        self.assertTrue(mutable.is_fittable(a))
        self.assertTrue(mutable.is_fittable(c))

        # type `class` has __data__ but is immutable
        d = D(int)
        self.assertEqual(len(mutable._IMMUTABLE_OBJECTS), 0)
        self.assertEqual(mutable.get_params(d), ())
        self.assertTrue(mutable.is_frozen(d))
        self.assertEqual(len(mutable._IMMUTABLE_OBJECTS), 1)
        self.assertFalse(mutable.set_params(d, ()))
        self.assertRaises(ValueError, mutable.set_params, d, (1.,))

        d = D(float)
        self.assertIs(mutable.freeze(d), False)  # tests TypeError check in freeze()

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
##########################################################################################
