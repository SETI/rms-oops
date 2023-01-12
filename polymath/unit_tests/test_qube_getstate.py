################################################################################
# test_qube_getstate.py: Tests of __getstate__ and __setstate__
################################################################################

from __future__ import division
import numpy as np
import pickle
import os
import sys
import unittest

from polymath import Qube, Boolean, Scalar, Pair, Vector, Vector3, Units
from polymath.extensions.pickler import FPZIP_ENCODING_CUTOFF as BIGDIM

FILEPATH = 'unittest.pickle'
EPSILON = sys.float_info.epsilon
ITERATIONS = 3

class Test_Qube_getstate(unittest.TestCase):

  def runTest(self):

    np.random.seed(4735)

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
      for Qube.DEFAULT_PICKLE_DIGITS in (('double', 'single'),
                                         ('double', 'double')):
        for ndims in range(1,5):
          for iteration in range(ITERATIONS):
            shape = tuple(np.random.randint(1, int(BIGDIM**0.4), (ndims,)))
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
                    if Qube.DEFAULT_PICKLE_DIGITS[1] == 'single':
                        diffs = a.derivs[key] - b.derivs[key]
                        self.assertTrue((diffs[antimask].rms() < 3.e-7).all())
                    else:
                        self.assertEqual(a.derivs[key][antimask],
                                         b.derivs[key][antimask])
                    self.assertEqual(readonly, b.derivs[key].readonly)

                # Try writing and then reading first iteration
                if iteration == 0:
                  with open(FILEPATH, 'wb') as f:
                    pickle.dump(a, f)
                  with open(FILEPATH, 'rb') as f:
                    b = pickle.load(f)
                  self.assertEqual(a, b)

                  for key in a.derivs:
                    antimask = np.logical_not(a.mask)
                    if Qube.DEFAULT_PICKLE_DIGITS[1] == 'single':
                        diffs = a.derivs[key] - b.derivs[key]
                        self.assertTrue((diffs[antimask].rms() < 3.e-7).all())
                    else:
                        self.assertEqual(a.derivs[key][antimask],
                                         b.derivs[key][antimask])
                    self.assertEqual(readonly, b.derivs[key].readonly)

    # Scalars with corners, derivatives
    for readonly in (False, True):
      for Qube.DEFAULT_PICKLE_DIGITS in (('double', 'single'),
                                         ('double', 'double')):
        for ndims in range(2,5):
          for iteration in range(ITERATIONS):
            shape = tuple(np.random.randint(2, int(BIGDIM**0.4), (ndims,)))
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
                    if Qube.DEFAULT_PICKLE_DIGITS[1] == 'single':
                        diffs = a.derivs[key] - b.derivs[key]
                        self.assertTrue(diffs[antimask].rms() < 3.e-7)
                    else:
                        self.assertEqual(a.derivs[key][antimask],
                                         b.derivs[key][antimask])
                    self.assertEqual(readonly, b.derivs[key].readonly)

                # Try writing and then reading first iteration
                if iteration == 0:
                  with open(FILEPATH, 'wb') as f:
                    pickle.dump(a, f)
                  with open(FILEPATH, 'rb') as f:
                    b = pickle.load(f)
                  self.assertEqual(a, b)

                  for key in a.derivs:
                    antimask = np.logical_not(a.mask)
                    if Qube.DEFAULT_PICKLE_DIGITS[1] == 'single':
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
      for ndims in range(2,4):
        for iteration in range(ITERATIONS):
          shape = tuple(np.random.randint(2, int(BIGDIM**0.5), (ndims,)))
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

                # Try writing and then reading first iteration
                if iteration == 0:
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

    #### COMPRESSION

    Qube._pickle_debug(True)

    # Tests of digits and reference
    references = (1., 'smallest', 'largest', 'mean', 'median', 'logmean')
    ref_values = [1.,
                  np.min(np.abs(a.values)),
                  np.max(np.abs(a.values)),
                  np.mean(np.abs(a.values)),
                  np.median(np.abs(a.values)),
                  np.exp(np.mean(np.log(np.abs(a.values))))]

    nbytes_tested = set()
    for digits in np.arange(6.5, 17.5, 0.5):
        error = 10.**(-digits)
        for reference, ref_value in zip(references, ref_values):
            a = Scalar(np.random.randn(10000))
            a.set_pickle_digits(digits, reference)

            b = Qube.__new__(Scalar)
            b.__setstate__(a.__getstate__())

            diff = a - b
            max_error = diff.abs().max(builtins=True)

            # we need to allow some latitude for single precision
            if digits < 7:
                self.assertTrue(max_error <= ref_value * error * 1.5)
            else:
                self.assertTrue(max_error <= ref_value * error)

            # Mean error should be ~ max_error/100 for N = 10,000
            max_mean = 4. * max_error/100.
            self.assertTrue((a - b).mean(builtins=True) <= max_mean)

            encoded = b.ENCODED_VALS
            if encoded[0] == 'scaled':
                nbytes_tested.add(encoded[3])

    self.assertEqual(nbytes_tested, {3,4,5,6})

    # Tests of offsets
    nbytes_tested = set()
    for digits in np.arange(6.5, 17.5, 0.5):
        error = 10.**(-digits)
        for offset_exp in range(8):
            offset = 10.**offset_exp
            a = Scalar(np.random.randn(1000)) + offset
            a.set_pickle_digits(digits, 'median')

            b = Qube.__new__(Scalar)
            b.__setstate__(a.__getstate__())
            max_error = (a - b).abs().max(builtins=True)

            precision = max(error * offset, EPSILON * a.max(builtins=True))
            self.assertTrue(max_error <= precision)

            encoded = b.ENCODED_VALS
            if encoded[0] == 'scaled':
                nbytes_tested.add(encoded[3])

    self.assertEqual(nbytes_tested, {1,2,3,4,5,6})

    # Tests of offsets + items
    nbytes_tested = set()
    for digits in np.arange(6.5, 17.5, 0.5):
        error = 10.**(-digits)
        for offset_exp in range(8):
            offset = (-10.**offset_exp, 0)
            a = Pair(np.random.randn(1000,2)) + offset
            a.set_pickle_digits(digits, 'median')

            b = Qube.__new__(Scalar)
            b.__setstate__(a.__getstate__())

            diff = a.values - b.values
            for k in range(2):
                max_error = np.max(np.abs(diff[:,k]))
                precision = max(error * max(abs(offset[k]), 1),
                                EPSILON * np.max(np.abs(a.values[:,k])))
                self.assertTrue(max_error <= precision)

                encoded = b.ENCODED_VALS[-1][k]
                if encoded[0] == 'scaled':
                    nbytes_tested.add(encoded[3])

    self.assertEqual(nbytes_tested, {1,2,3,4,5,6})

    # Tests of fpzip
    alist = [
        np.ones(1000),
        np.random.randn(1000),
        np.random.rand(1000),
        np.arange(1, 1001.),
        np.sqrt(np.arange(1.,1001.)),
        np.sqrt(np.arange(1, 1001.)) + 0.001 * np.random.randn(1000)
    ]

    for avals in alist:
        for digits in np.arange(6.5, 17.5, 0.5):
            error = 10.**(-digits)
            a = Scalar(avals)
            a.set_pickle_digits(digits, 'fpzip')

            b = Qube.__new__(Scalar)
            b.__setstate__(a.__getstate__())

            rel_error = ((a - b)/a).abs().max(builtins=True)
            self.assertTrue(rel_error <= error)

    Qube._pickle_debug(False)

################################################################################
# Execute from command line...
################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
