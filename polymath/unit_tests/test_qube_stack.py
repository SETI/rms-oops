################################################################################
# Qube.stack() tests
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import Qube, Scalar, Vector3, Boolean, Units

class Test_Qube_stack(unittest.TestCase):

  def runTest(self):

    a = Scalar(np.arange(10))
    b = Scalar(np.arange(10,20))
    ab = Scalar(np.arange(20).reshape(2,10))

    self.assertEqual(Qube.stack(a,b), ab)
    self.assertTrue(a.is_int())
    self.assertTrue(b.is_int())
    self.assertTrue(ab.is_int())
    self.assertTrue(np.all(Qube.stack(a,b).mask == False))

    # Cast int to float
    b = Scalar(np.arange(10,20.))
    ab = Scalar(np.arange(20.).reshape(2,10))
    self.assertEqual(Qube.stack(a,b), ab)
    self.assertTrue(b.is_float())
    self.assertTrue(ab.is_float())
    self.assertTrue(np.all(Qube.stack(a,b).mask == False))

    # Cast bools, None to float
    c = Boolean(5*[True] + 5*[False])
    d = None
    abcd = Qube.stack(a,b,c,d)
    self.assertEqual(abcd[:2], ab)
    self.assertEqual(abcd[2], 5*[1.] + 5*[0.])
    self.assertEqual(abcd[3], 10*[0.])
    self.assertTrue(np.all(abcd.mask == False))
    self.assertTrue(c.is_bool())
    self.assertTrue(abcd.is_float())

    # Cast bools, None to int
    b = Scalar(np.arange(10,20))
    abcd = Qube.stack(a,b,c,d)
    self.assertEqual(abcd[:2], ab)
    self.assertEqual(abcd[2], 5*[1] + 5*[0])
    self.assertEqual(abcd[3], 10*[0])
    self.assertTrue(np.all(abcd.mask == False))
    self.assertTrue(abcd.is_int())

    # Cast bools, None to bool
    cd = Qube.stack(c,d)
    self.assertEqual(cd[0], 5*[True] + 5*[False])
    self.assertEqual(cd[1], 10*[False])
    self.assertTrue(np.all(cd.mask == False))
    self.assertTrue(cd.is_bool())

    # Derivs
    b_d_dx = Scalar(np.arange(30.).reshape(10,3), drank=1)
    a_d_dt = Scalar(np.arange(10.) / 10.)
    a.insert_deriv('t', a_d_dt)
    b.insert_deriv('x', b_d_dx)
    abcd = Qube.stack(a,b,c,d)
    self.assertEqual(abcd.d_dt[0], a_d_dt)
    self.assertEqual(abcd.d_dx[1], b_d_dx)
    self.assertEqual(abcd.d_dt[1:], 0.)
    self.assertEqual(abcd.d_dx[0], [0.,0.,0.])
    self.assertEqual(abcd.d_dx[2:], [0.,0.,0.])

    a.insert_deriv('t', a_d_dt)
    b.insert_deriv('t', b_d_dx)
    self.assertRaises(ValueError, Qube.stack, a, b)

    a = Scalar(np.arange(10))
    b = Scalar(np.arange(10,20))
    a.insert_deriv('t', a_d_dt)
    b.insert_deriv('t', b_d_dx)
    ab = Scalar(np.arange(20).reshape(2,10))
    self.assertEqual(Qube.stack(a,b,recursive=False), ab)
    self.assertEqual(Qube.stack(a,b,recursive=False).derivs, {})

    # Ranks
    a = Scalar(np.arange(30.).reshape(10,3), drank=1)
    b = Scalar(np.arange(10.))
    self.assertRaises(ValueError, Qube.stack, a, b)

    a = Scalar(np.arange(30.).reshape(10,3), drank=1)
    b = Scalar(np.arange(30.,60.).reshape(10,3), drank=1)
    ab = Qube.stack(a,b)
    self.assertTrue(np.all(ab.values.flatten() == np.arange(60)))

    # Units
    a = Scalar(np.arange(10), units=Units.KM)
    b = Scalar(np.arange(10,20))
    ab = Qube.stack(a,b)
    self.assertEqual(ab.units, Units.KM)

    a = Scalar(np.arange(10))
    b = Scalar(np.arange(10,20), units=Units.DEG)
    ab = Qube.stack(a,b)
    self.assertEqual(ab.units, Units.DEG)

    a = Scalar(np.arange(10), units=Units.KM)
    b = Scalar(np.arange(10,20), units=Units.DEG)
    self.assertRaises(ValueError, Qube.stack, a, b)

    # Masks
    a = Scalar(np.arange(10), mask=True)
    b = Scalar(np.arange(10.,20.), mask=True)
    c = Boolean(5*[True] + 5*[False], mask=True)
    d = None
    self.assertTrue(Qube.stack(a,b,c,d).mask is True)

    a = Scalar(np.arange(10), mask=False)
    b = Scalar(np.arange(10.,20.), mask=False)
    c = Boolean(5*[True] + 5*[False], mask=False)
    d = None
    self.assertTrue(np.all(Qube.stack(a,b,c,d).mask == False))

    a = Scalar(np.arange(10), mask=False)
    b = Scalar(np.arange(10.,20.), mask=True)
    c = Boolean(5*[True] + 5*[False], mask=False)
    d = None
    abcd = Qube.stack(a,b,c,d)
    self.assertEqual(type(abcd.mask), np.ndarray)
    self.assertEqual(abcd[0], np.arange(10))
    self.assertEqual(abcd[1], Scalar.MASKED)
    self.assertEqual(abcd[2], [1,1,1,1,1,0,0,0,0,0])
    self.assertEqual(abcd[3], [0,0,0,0,0,0,0,0,0,0])

    a = Scalar(np.arange(10), mask=[1,1,1,1,1,0,0,0,0,0])
    b = Scalar(np.arange(10.,20.), mask=[1,1,1,1,1,0,0,0,0,0])
    c = Boolean(5*[True] + 5*[False], mask=[1,1,1,1,1,0,0,0,0,0])
    d = None
    abcd = Qube.stack(a,b,c,d)
    self.assertTrue(np.all(abcd[0:3].mask == 3*[[1,1,1,1,1,0,0,0,0,0]]))
    self.assertTrue((abcd[3] == False).all())

    # Broadcasting
    a = Scalar(np.arange(10).reshape(10,1))
    b = Scalar(11.)
    c = Boolean(5*[True] + 5*[False])
    d = None
    self.assertEqual(Qube.stack(a,b,c,d).shape, (4,10,10))

    a = Scalar(np.arange(10), mask=[0,0,0,0,0,1,1,1,1,1])
    b = Scalar(11., mask=False)
    c = Boolean(5*[True] + 5*[False], mask=True)
    d = None
    abcd = Qube.stack(a,b,c,d)
    self.assertEqual(abcd.shape, (4,10))
    self.assertEqual(abcd.mask.shape, (4,10))
    self.assertEqual(abcd[0][:5], np.arange(5))
    self.assertEqual(abcd[0][5:], Scalar.MASKED)
    self.assertEqual(abcd[1], 10*[11.])
    self.assertEqual(abcd[2], Scalar.MASKED)
    self.assertEqual(abcd[3], 0.)

    a = Scalar(np.arange(10), mask=False)
    b = Scalar(np.arange(10.,20.), mask=True)
    c = Boolean(5*[True] + 5*[False], mask=False)
    d = None
    abcd = Qube.stack(a,b,c,d)
    self.assertEqual(type(abcd.mask), np.ndarray)
    self.assertEqual(abcd[0], np.arange(10))
    self.assertEqual(abcd[1], Scalar.MASKED)
    self.assertEqual(abcd[2], [1,1,1,1,1,0,0,0,0,0])
    self.assertEqual(abcd[3], [0,0,0,0,0,0,0,0,0,0])

    a = Scalar(np.arange(10), mask=[1,1,1,1,1,0,0,0,0,0])
    b = Scalar(np.arange(10.,20.), mask=[1,1,1,1,1,0,0,0,0,0])
    c = Boolean(5*[True] + 5*[False], mask=[1,1,1,1,1,0,0,0,0,0])
    d = None
    abcd = Qube.stack(a,b,c,d)
    self.assertTrue(np.all(abcd[0:3].mask == 3*[[1,1,1,1,1,0,0,0,0,0]]))
    self.assertTrue((abcd[3] == False).all())

    # Booleans
    c = Boolean(5*[True] + 5*[False])
    d = Scalar(np.arange(10))
    cd = Qube.stack(c,d)
    self.assertTrue(cd.is_int())
    self.assertEqual(type(cd), Scalar)

    d = np.arange(10)
    cd = Qube.stack(c,d)
    self.assertTrue(cd.is_int())
    self.assertEqual(type(cd), Qube)

    d = np.arange(10.)
    cd = Qube.stack(c,d)
    self.assertTrue(cd.is_float())
    self.assertEqual(type(cd), Qube)

    d = 1
    cd = Qube.stack(c,d)
    self.assertTrue(cd.is_int())
    self.assertEqual(type(cd), Qube)

    d = 1.
    cd = Qube.stack(c,d)
    self.assertTrue(cd.is_float())
    self.assertEqual(type(cd), Qube)

    d = True
    cd = Qube.stack(c,d)
    self.assertTrue(cd.is_bool())
    self.assertEqual(type(cd), Boolean)

    d = np.array([True])
    cd = Qube.stack(c,d)
    self.assertTrue(cd.is_bool())
    self.assertEqual(type(cd), Boolean)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
