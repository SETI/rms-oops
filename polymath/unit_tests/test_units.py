# UNIT TESTS
################################################################################

from __future__ import division
import unittest
import numpy as np

from polymath import Units

class Test_Units(unittest.TestCase):

  def runTest(self):

    self.assertEqual(str(Units.KM),                 "Units(km)")
    self.assertEqual(str(Units.KM*Units.KM),        "Units(km**2)")
    self.assertEqual(str(Units.KM**2),              "Units(km**2)")
    self.assertEqual(str(Units.KM**(-2)),           "Units(km**(-2))")
    self.assertEqual(str(Units.KM/Units.S),         "Units(km/s)")
    self.assertEqual(str((Units.KM/Units.S)**2),    "Units(km**2/s**2)")
    self.assertEqual(str((Units.KM/Units.S)**(-2)), "Units(s**2/km**2)")

    self.assertEqual((Units.KM/Units.S).exponents, (1,-1,0))
    self.assertEqual((Units.KM/Units.S/Units.S).exponents, (1,-2,0))

    self.assertEqual(Units.KM.convert(3.,Units.CM), 3.e5)
    self.assertTrue(np.all(Units.KM.convert(np.array([1.,2.,3.]), Units.CM) ==
                           [1.e5, 2.e5, 3.e5]))

    self.assertTrue(np.all(Units.DEGREES.convert(np.array([1.,2.,3.]),
                           Units.ARCSEC) == [3600., 7200., 10800.]))

    self.assertTrue(np.all((Units.DEG/Units.S).convert(np.array([1.,2.,3.]),
                            Units.ARCSEC/Units.S) == [3600., 7200., 10800.]))

    self.assertTrue(np.all((Units.DEG/Units.H).convert(np.array([1.,2.,3.]),
                            Units.ARCSEC/Units.S) == [1., 2., 3.]))

    self.assertTrue(np.all((Units.DEG*Units.S).convert(np.array([1.,2.,3.]),
                            Units.ARCSEC*Units.H) == [1., 2., 3.]))

    self.assertTrue(np.all((Units.DEG**2).convert(np.array([1.,2.,3.]),
                            Units.ARCMIN*Units.ARCSEC) ==
                            [3600*60, 3600*60*2, 3600*60*3]))

    eps = 1.e-15
    test = Units.DEG.from_this(np.array([1.,2.,3.]))
    self.assertTrue(np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps))
    self.assertTrue(np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps))

    test = Units.DEG.into_this(test)
    self.assertTrue(np.all(np.array([1., 2., 3.]) < test + eps))
    self.assertTrue(np.all(np.array([1., 2., 3.]) > test - eps))

    self.assertFalse(Units.CM == Units.M)
    self.assertTrue( Units.CM != Units.M)
    self.assertTrue( Units.M  != Units.SEC)
    self.assertEqual(Units.M.factor, Units.MRAD.factor)
    self.assertTrue( Units.CM, Units((1,0,0), (10., 1.e6, 0)))

    test = Units.ROTATION/Units.S
    self.assertEqual(test.get_name(), "rotation/s")

    units = Units.KM**3/Units.S*Units.RAD*Units.KM**(-2) / Units.RAD
    self.assertEqual(str(units), "Units(km/s)")

    units = (Units.KM**3/Units.S*Units.RAD*Units.KM**(-2) /
                         Units.MRAD*Units.MSEC/(Units.KM/Units.S) /
                         Units.S)
    units.name = None
    self.assertEqual(str(units), "Units()")

    self.assertEqual(str(Units.S * 60), "Units(min)")
    self.assertEqual(str(60 * Units.S), "Units(min)")

    self.assertEqual(str(Units.H/3600), "Units(s)")
    self.assertEqual(str((1000/Units.KM)**(-2)), "Units(m**2)")

    self.assertTrue( Units.can_match(None, None))
    self.assertTrue( Units.can_match(None, Units.UNITLESS))
    self.assertTrue( Units.can_match(None, Units.KM))
    self.assertTrue( Units.can_match(Units.KM, None))
    self.assertTrue( Units.can_match(Units.CM, Units.KM))
    self.assertFalse(Units.can_match(Units.S, Units.KM))
    self.assertFalse(Units.can_match(Units.S, Units.UNITLESS))

    self.assertTrue( Units.do_match(None, None))
    self.assertTrue( Units.do_match(None, Units.UNITLESS))
    self.assertFalse(Units.do_match(None, Units.KM))
    self.assertFalse(Units.do_match(Units.KM, None))
    self.assertTrue( Units.do_match(Units.CM, Units.KM))
    self.assertFalse(Units.do_match(Units.S, Units.KM))
    self.assertFalse(Units.do_match(Units.S, Units.UNITLESS))

    self.assertEqual(Units.KM, (Units.KM**2).sqrt())

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
