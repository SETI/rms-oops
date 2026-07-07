##########################################################################################
# tests/test_fittable.py
##########################################################################################

import unittest

from oops.fittable import Fittable, Fittable_


class A(Fittable):
    def __init__(self, x):
        self.x = x
        self._refresh()

    def _refresh(self):
        self.x_squared = self.x**2

    def _set_params(self, params):
        self.x = params[0]

    @property
    def _params(self):
        return (self.x,)

class B:
    def __init__(self, x, a):
        self.x = x
        self.a = a
        self._refresh()

    def _refresh(self):
        self.x_plus_a2 = self.x + self.a.x_squared


class C(Fittable):
    def __init__(self, x, a):
        self.x = x
        self.a = a
        self.c = self
        self._refresh()

    def _set_params(self, params):
        self.x = params[0]

    @property
    def _params(self):
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
        self.assertFalse(Fittable_.is_fittable(x))
        self.assertEqual(Fittable_.fittables(x), [])
        self.assertEqual(Fittable_.version(x), 0)

        a = A(7)
        self.assertEqual(a.x_squared, 49)
        self.assertEqual(Fittable_.get_params(a), (7,))
        self.assertIsInstance(Fittable_.get_params(a), tuple)
        self.assertEqual(Fittable_.get_params(a, as_dict=True), {'':(7,)})
        self.assertTrue(Fittable_.is_fittable(a))
        self.assertEqual(Fittable_.fittables(a), [])
        self.assertEqual(Fittable_.version(a), 0)

        self.assertEqual(a.get_params(), (7,))
        self.assertEqual(a.get_params(as_dict=True), {'':(7,)})
        self.assertEqual(a.version(), 0)

        a.set_params([5])
        self.assertEqual(Fittable_.get_params(a), (5,))
        self.assertEqual(a.x_squared, 25)
        self.assertEqual(Fittable_.fittables(a), [])
        self.assertTrue(Fittable_.is_fittable(a))
        self.assertEqual(Fittable_.version(a), 1)

        b = B(1, a)
        self.assertEqual(b.x_plus_a2, 26)
        self.assertEqual(Fittable_.get_params(b), ())
        self.assertEqual(Fittable_.get_params(b, as_dict=True), {'a':(5,)})

        self.assertTrue(Fittable_.is_fittable(b))
        Fittable_.set_params(b, {'a':7})
        self.assertEqual(b.x_plus_a2, 50)
        self.assertEqual(Fittable_.fittables(b), ['a'])
        self.assertEqual(Fittable_.fittables(b, frozen=True), ['a'])
        self.assertEqual(Fittable_.fittables(b, frozen=False), ['a'])

        Fittable_.freeze(a)
        self.assertEqual(Fittable_.fittables(b, frozen=True), ['a'])
        self.assertEqual(Fittable_.fittables(b, frozen=False), [])

        a = A(5)
        c = C(1, a)
        self.assertEqual(Fittable_.fittables(c), ['a', 'c'])
        self.assertTrue(Fittable_.is_fittable(c))
        self.assertEqual(c.x_plus_a2_plus_cx_plus_ccx, 28)

        a.set_params([6])
        self.assertTrue(c.refresh())
        self.assertEqual(c.x_plus_a2_plus_cx_plus_ccx, 39)
        self.assertFalse(c.refresh())
        self.assertFalse(c.refresh())

        a = A(5)
        c = C(1, a)
        self.assertEqual(c.x_plus_a2_plus_cx_plus_ccx, 28)
        c.set_params({'':1, 'a':6})
        self.assertEqual(c.x_plus_a2_plus_cx_plus_ccx, 39)
        self.assertFalse(c.refresh())
        self.assertEqual(Fittable_.get_params(a), (6,))
        self.assertEqual(Fittable_.get_params(a, as_dict=True), {'': (6,)})
        self.assertEqual(Fittable_.get_params(c), (1,))
        self.assertEqual(Fittable_.get_params(c, as_dict=True), {'': (1,), 'a': (6,)})

        self.assertFalse(Fittable_.is_frozen(a))
        self.assertFalse(Fittable_.is_frozen(c))
        self.assertFalse(a.is_frozen())
        self.assertFalse(c.is_frozen())

        Fittable_.freeze(a)
        self.assertTrue(Fittable_.is_frozen(a))
        self.assertFalse(Fittable_.is_frozen(c))
        self.assertRaises(ValueError, Fittable_.set_params, a, 2)

        self.assertEqual(Fittable_.fittables(c), ['c'])
        self.assertEqual(Fittable_.fittables(c, frozen=True), ['a', 'c'])
        self.assertTrue(Fittable_.is_fittable(a))
        self.assertTrue(Fittable_.is_fittable(c))
        self.assertEqual(Fittable_.get_params(c), (1,))
        self.assertEqual(Fittable_.get_params(c, as_dict=True), {'': (1,)})
        self.assertEqual(Fittable_.get_params(c, as_dict=True, frozen=True),
                         {'': (1,), 'a': (6,)})
        self.assertEqual(c.get_params(), (1,))
        self.assertEqual(c.get_params(as_dict=True), {'': (1,)})
        self.assertEqual(c.get_params(as_dict=True, frozen=True),
                         {'': (1,), 'a': (6,)})

        a = A(5)
        c = C(1, a)
        Fittable_.freeze(c)
        self.assertTrue(Fittable_.is_frozen(a))
        self.assertTrue(Fittable_.is_frozen(c))
        self.assertTrue(a.is_frozen())
        self.assertTrue(c.is_frozen())
        self.assertRaises(ValueError, Fittable_.set_params, a, 2)
        self.assertRaises(ValueError, Fittable_.set_params, c, 2)

        self.assertEqual(Fittable_.fittables(c), [])
        self.assertEqual(Fittable_.fittables(c, frozen=True), ['a', 'c'])
        self.assertTrue(Fittable_.is_fittable(a))
        self.assertTrue(Fittable_.is_fittable(c))

        # type class has __data__ but is immutable
        d = D(int)
        self.assertEqual(len(Fittable_._FROZEN_IDS), 0)
        self.assertEqual(Fittable_.get_params(d), ())
        self.assertTrue(Fittable_.is_frozen(d))
        self.assertEqual(len(Fittable_._FROZEN_IDS), 1)
        self.assertRaises(ValueError, Fittable_.set_params, d, ())

        d = D(float)
        self.assertIs(Fittable_.freeze(d), None)    # tests TypeError check in freeze()

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
##########################################################################################
