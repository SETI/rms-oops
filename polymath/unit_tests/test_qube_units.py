################################################################################
# Tests methods related to units:
#   set_units(self, units, override=False)
#   without_units(self, recursive=True)
#   into_units(self, recursive=True)
#   from_units(self, recursive=True)
################################################################################

from __future__ import division
import numpy as np
import unittest

from polymath import *

#*******************************************************************************
# Test_Qube_units
#*******************************************************************************
class Test_Qube_units(unittest.TestCase):

  #=============================================================================
  # runTest
  #=============================================================================
  def runTest(self):

    ############################################################################
    # set_units(self, units, override=False)
    ############################################################################

    a = Scalar((1.,2.,3.))
    self.assertEqual(a.units, None)
    self.assertTrue(np.all(a.values == (1,2,3)))

    a.set_units(Units.KM)
    self.assertEqual(a.units, Units.KM)
    self.assertTrue(np.all(a.values == (1,2,3)))

    a.set_units(Units.CM)
    self.assertEqual(a.units, Units.CM)
    self.assertTrue(np.all(a.values == (1,2,3)))

    self.assertRaises(ValueError, a.set_units, Units.DEG)   # incompatible

    a.set_units(Units.M)
    self.assertEqual(a.units, Units.M)
    self.assertTrue(np.all(a.values == (1,2,3)))

    a = a.as_readonly()
    self.assertTrue(a.readonly)
    self.assertRaises(ValueError, a.set_units, Units.KM)

    a.set_units(Units.KM, override=True)
    self.assertTrue(a.readonly)
    self.assertEqual(a.units, Units.KM)
    self.assertTrue(np.all(a.values == (1,2,3)))

    # Classes for which units are not allowed
    a = Matrix3([(1,0,0),(0,1,0),(0,0,1)])
    self.assertRaises(TypeError, a.set_units, Units.KM)

    a = Quaternion((1,0,0,0))
    self.assertRaises(TypeError, a.set_units, Units.KM)

    a = Boolean([True, False])
    self.assertRaises(TypeError, a.set_units, Units.KM)

    ############################################################################
    # without_units(self, recursive=True)
    ############################################################################

    a = Scalar((1.,2.,3.), units=Units.KM)
    b = a.without_units()
    self.assertEqual(a.units, Units.KM)

    self.assertEqual(b.units, None)
    self.assertTrue(np.all(a.values == b.values))

    self.assertEqual(a.readonly, False)
    self.assertEqual(b.readonly, False)

    a = a.as_readonly()
    self.assertEqual(a.readonly, True)

    b = a.without_units()
    self.assertEqual(b.readonly, True)
    self.assertEqual(b.units, None)
    self.assertTrue(np.all(b.values == (1,2,3)))

    ############################################################################
    # into_units(self, recursive=True)
    # from_units(self, recursive=True)
    ############################################################################

    a = Scalar((1.,2.,3.))
    self.assertEqual(a.units, None)
    self.assertTrue(np.all(a.values == (1,2,3)))

    a.set_units(Units.M)
    self.assertEqual(a.units, Units.M)
    self.assertTrue(np.all(a.values == (1,2,3)))

    b = a.into_units()
    self.assertTrue(np.all(b.values == (1000, 2000, 3000)))

    c = b.from_units()
    self.assertTrue(np.all(c.values == a.values))

    a = Scalar((1.,2.,3.), units=Units.M)
    da_dt = Scalar((4., 5., 6.), units=Units.CM/Units.S)
    a.insert_deriv('t', da_dt)

    b = a.into_units()
    self.assertTrue(np.all(b.values == (1000, 2000, 3000)))
    self.assertTrue(np.all(b.d_dt.values == (400000, 500000, 600000)))

    DEL = 1.e-14

    c = b.from_units()

    self.assertTrue(np.all(c.values == a.values))
    self.assertTrue((c-a).d_dt.rms().max() < DEL)
  #=============================================================================



#*******************************************************************************



############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
