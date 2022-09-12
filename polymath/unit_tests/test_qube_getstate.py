################################################################################
# test_qube_getstate.py: Tests of __getstate__ and __setstate__
################################################################################

from __future__ import division
import numpy as np
import pickle
import os
import unittest

from polymath import Qube, Boolean, Scalar, Vector, Vector3, Units

FILEPATH = 'unittest.pickle'

class Test_Qube_getstate(unittest.TestCase):

  def runTest(self):

    # Scalar tests, no derivatives
    for readonly in (False, True):
      for vals in (1, 1., (1,2,3), (1.,2.,3.)):
        for mask in (True, False, 3*[True], 3*[False], [True, True, False]):
            if len(np.shape(vals)) < len(np.shape(mask)):
                continue
            a = Scalar(vals, mask)
            if readonly:
                a.as_readonly()
            self.assertEqual(readonly, a.readonly)
            b = Qube.__new__(type(a))
            b.__setstate__(a.__getstate__())
            self.assertEqual(a, b)
            self.assertEqual(readonly, b.readonly)

    # Scalar tests, derivatives, units
    for readonly in (False, True):
      for Qube.DEFAULT_DERIV_PICKLE_DIGITS in (7, 16):
        for ndims in range(1,4):
          for iteration in range(20):
            shape = tuple(np.random.randint(1,5,(ndims,)))
            vals = np.random.randn(*shape)
            for mask in (False, True,
                         np.zeros(shape, dtype='bool'),
                         np.ones(shape, dtype='bool'),
                         (np.random.randn(*shape) < -0.5)):
              for units in (None, Units.KM):
                deriv1 = Scalar(np.random.randn(*shape))
                shape2 = shape + (2,)
                deriv2 = Scalar(np.random.randn(*shape2), drank=1)
                for derivs in ({}, {'t': deriv1}, {'xy': deriv2},
                                   {'t': deriv1, 'xy': deriv2}):
                  a = Scalar(vals, mask, units=units, derivs=derivs)
                  if readonly:
                    a.as_readonly()
                  b = Qube.__new__(type(a))
                  b.__setstate__(a.__getstate__())
                  self.assertEqual(a, b)
                  self.assertEqual(readonly, b.readonly)
                  for key in a.derivs:
                    antimask = np.logical_not(a.mask)
                    if Qube.DEFAULT_DERIV_PICKLE_DIGITS == 7:
                        diffs = a.derivs[key] - b.derivs[key]
                        self.assertTrue((diffs[antimask].rms() < 3.e-7).all())
                    else:
                        self.assertEqual(a.derivs[key][antimask],
                                         b.derivs[key][antimask])
                    self.assertEqual(readonly, b.derivs[key].readonly)

                # Test every 5th by writing and reading
                if iteration % 5 == 0:
                  with open(FILEPATH, 'wb') as f:
                    pickle.dump(a, f)
                  with open(FILEPATH, 'rb') as f:
                    b = pickle.load(f)
                  self.assertEqual(a, b)

                  for key in a.derivs:
                    antimask = np.logical_not(a.mask)
                    if Qube.DEFAULT_DERIV_PICKLE_DIGITS == 7:
                        diffs = a.derivs[key] - b.derivs[key]
                        self.assertTrue((diffs[antimask].rms() < 3.e-7).all())
                    else:
                        self.assertEqual(a.derivs[key][antimask],
                                         b.derivs[key][antimask])
                    self.assertEqual(readonly, b.derivs[key].readonly)

    # Scalars with corners, derivatives
    for readonly in (False, True):
      for Qube.DEFAULT_DERIV_PICKLE_DIGITS in (7, 16):
        for ndims in range(1,4):
          for iteration in range(20):
            shape = tuple(np.random.randint(2,7,(ndims,)))
            vals = np.random.randn(*shape)

            mask1 = np.random.randn(*shape) < -0.5
            mask1[0] = True
            if ndims > 1:
              mask1[:,-1] = True
            if ndims > 2:
              mask1[:,:,0] = True

            mask2 = np.zeros(shape, dtype='bool')
            mask2[-1] = True
            if ndims > 1:
              mask2[:,0] = True
            if ndims > 2:
              mask2[:,:,-1] = True

            for mask in (mask1, mask2):
              deriv1 = Scalar(np.random.randn(*shape))
              shape2 = shape + (2,)
              deriv2 = Scalar(np.random.randn(*shape2), drank=1)
              for derivs in ({}, {'t': deriv1}, {'xy': deriv2},
                                 {'t': deriv1, 'xy': deriv2}):
                a = Scalar(vals, mask, units=units, derivs=derivs)
                if readonly:
                    a.as_readonly()
                b = Qube.__new__(type(a))
                b.__setstate__(a.__getstate__())
                self.assertEqual(a, b)
                for key in a.derivs:
                    antimask = np.logical_not(a.mask)
                    if Qube.DEFAULT_DERIV_PICKLE_DIGITS == 7:
                        diffs = a.derivs[key] - b.derivs[key]
                        self.assertTrue(diffs[antimask].rms() < 3.e-7)
                    else:
                        self.assertEqual(a.derivs[key][antimask],
                                         b.derivs[key][antimask])
                    self.assertEqual(readonly, b.derivs[key].readonly)

                # Test every 5th by writing and reading
                if iteration % 5 == 0:
                  with open(FILEPATH, 'wb') as f:
                    pickle.dump(a, f)
                  with open(FILEPATH, 'rb') as f:
                    b = pickle.load(f)
                  self.assertEqual(a, b)

                  for key in a.derivs:
                    antimask = np.logical_not(a.mask)
                    if Qube.DEFAULT_DERIV_PICKLE_DIGITS == 7:
                        diffs = a.derivs[key] - b.derivs[key]
                        self.assertTrue((diffs[antimask].rms() < 3.e-7).all())
                    else:
                        self.assertEqual(a.derivs[key][antimask],
                                         b.derivs[key][antimask])
                    self.assertEqual(readonly, b.derivs[key].readonly)

    # Scalar, shapeless derivative
    for readonly in (False, True):
      for mask in (True, False):
        a = Scalar(1., mask, derivs={'t': Scalar(7.)})
        if readonly:
            a.as_readonly()
        b = Qube.__new__(type(a))
        b.__setstate__(a.__getstate__())
        self.assertEqual(a, b)
        self.assertEqual(readonly, b.readonly)

    # Boolean tests
    for readonly in (False, True):
      for vals in (0, 1, (0,0,0), (1,1,1), (0,1,0), [[1,1,0],[0,0,1]]):
        for mask in (True, False, 3*[True], 3*[False], [True, True, False]):
            if len(np.shape(vals)) < len(np.shape(mask)):
                continue
            a = Boolean(vals, mask)
            if readonly:
                a.as_readonly()
            b = Qube.__new__(type(a))
            b.__setstate__(a.__getstate__())
            self.assertEqual(a, b)
            self.assertEqual(readonly, b.readonly)

    # Vector tests, no derivatives
    for readonly in (False, True):
      for vals in ((1,2), (1.,2.), [(1,2,3),(4,5,6)], [(1.,2.),(4.,5.)]):
        for mask in (True, False, 2*[True], 2*[False], [True, False]):
            if len(np.shape(vals))-1 < len(np.shape(mask)):
                continue
            a = Vector(vals, mask)
            if readonly:
                a.as_readonly()
            b = Qube.__new__(type(a))
            b.__setstate__(a.__getstate__())
            self.assertEqual(a, b)
            self.assertEqual(readonly, b.readonly)

    # Vector3 with corners, derivatives
    for readonly in (False, True):
      for ndims in range(1,4):
        for iteration in range(20):
          shape = tuple(np.random.randint(2,7,(ndims,)))
          shape3 = shape + (3,)
          vals = np.random.randn(*shape3)

          mask1 = np.random.randn(*shape) < -0.5
          mask1[0] = True
          if ndims > 1:
              mask1[:,-1] = True
          if ndims > 2:
              mask1[:,:,0] = True

          mask2 = np.zeros(shape, dtype='bool')
          mask2[-1] = True
          if ndims > 1:
              mask2[:,0] = True
          if ndims > 2:
              mask2[:,:,-1] = True

          for mask in (mask1, mask2):
            shape3 = shape + (3,)
            shape32 = shape + (3,2)
            shape333 = shape + (3,3,3)
            deriv1 = Vector3(np.random.randn(*shape3))
            deriv2 = Vector3(np.random.randn(*shape32), drank=1)
            deriv3 = Vector3(np.random.randn(*shape333), drank=2)
            for derivs in ({}, {'t': deriv1}, {'uv': deriv2}, {'xyz': deriv3},
                               {'t': deriv1, 'uv': deriv2, 'xyz': deriv3}):
                a = Vector3(vals, mask, units=units, derivs=derivs)
                if readonly:
                    a.as_readonly()
                b = Qube.__new__(type(a))
                b.__setstate__(a.__getstate__())
                self.assertEqual(a, b)
                self.assertEqual(readonly, b.readonly)
                for key in a.derivs:
                    antimask = np.logical_not(a.mask)
                    self.assertEqual(a.derivs[key][antimask],
                                     b.derivs[key][antimask])
                    self.assertEqual(readonly, b.readonly)

                # Test every 5th by writing and reading
                if iteration % 5 == 0:
                  with open(FILEPATH, 'wb') as f:
                    pickle.dump(a, f)
                  with open(FILEPATH, 'rb') as f:
                    b = pickle.load(f)
                  self.assertEqual(a, b)
                  self.assertEqual(readonly, b.readonly)

                  for key in a.derivs:
                    antimask = np.logical_not(a.mask)
                    self.assertEqual(a.derivs[key][antimask],
                                     b.derivs[key][antimask])
                    self.assertEqual(readonly, b.readonly)

    os.remove(FILEPATH)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
