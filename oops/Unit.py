################################################################################
# Class Unit
#
# 1/24/12 (MRS): Initial version.
################################################################################

import numpy as np
import unittest

import oops

class Unit(object):
    """Unit is a class defining unit names and the methods for converting
    between values that include units."""

    OOPS_CLASS = "Unit"

    def __init__(self, name, exponents, numer=1, denom=1, pi_exponent=0):
        """Constructor for a Unit object.

        Input:
            exponents       a triple of integers defining the exponents on
                            distance, time and angle that are used for this
                            unit.
            numer           the numerator of a factor that converts from a value
                            in these units to a value in standard units of km,
                            seconds and/or radians.
            denom           the denominator of this same factor.
            pi_exponent     the exponent on pi that should multiply the
                            numerator of this factor.

        For example, a unit of degrees would have numer=1., denom=180., and
        pi_exponent=1. This defines a factor pi/180, which converts from
        degrees to radians.
        """

        self.name = name
        self.exponents = tuple(exponents)
        self.numer = numer
        self.denom = denom
        self.pi_exponent = pi_exponent

        self.factor     = (numer / float(denom)) * np.pi**pi_exponent
        self.factor_inv = (denom / float(numer)) / np.pi**pi_exponent

    def to_standard(self, value):
        """Converts a value with units to one in standard units of km, seconds
        and radians."""

        return self.factor * np.asarray(value)

    def to_unit(self, value):
        """Converts a value in standard units to one with the specified units.
        """

        return self.factor_inv * np.asarray(value)

    def convert(self, value, unit):
        """Converts a value in these units to a value in the given units.
        Conversions are exact whenever possible."""

        if self.exponents != unit.exponents:
            raise ValueError("cannot convert units " + self.name + " to " +
                                                       unit.name)

        return (float(self.numer * unit.denom) * np.asarray(value) /
                float(self.denom * unit.numer) *
                np.pi**(self.pi_exponent - unit.pi_exponent))

    def mul(self, arg, name=None):
        """Returns a Unit constructed as the product of two other Units."""

        if name == None:
            name = self.name + "*" + arg.name

        return Unit(name,
                    (self.exponents[0] + arg.exponents[0],
                     self.exponents[1] + arg.exponents[1],
                     self.exponents[2] + arg.exponents[2]),
                    self.numer * arg.numer,
                    self.denom * arg.denom,
                    self.pi_exponent + arg.pi_exponent)

    def div(self, arg, name=None):
        """Returns a Unit constructed as the ratio of two other Units."""

        if name == None:
            name = self.name + "/" + arg.name

        return Unit(name,
                    (self.exponents[0] - arg.exponents[0],
                     self.exponents[1] - arg.exponents[1],
                     self.exponents[2] - arg.exponents[2]),
                    self.numer * arg.denom,
                    self.denom * arg.numer,
                    self.pi_exponent - arg.pi_exponent)

    def pow(self, power, name=None):
        """Returns a Unit constructed as a power of another Unit."""

        if power > 0:
            if name == None:
                if self.exponents == (0,0,0):
                    name = ""
                elif "*" in self.name or "/" in self.name:
                    name = "(" + self.name + ")**" + str(power)
                else:
                    name = self.name + "**" + str(power)

            return Unit(name,
                        (power * self.exponents[0],
                         power * self.exponents[1],
                         power * self.exponents[2]),
                        self.numer**power,
                        self.denom**power,
                        power * self.pi_exponent)

        else:
            if name == None:
                if self.exponents == (0,0,0):
                    name = ""
                elif "*" in self.name or "/" in self.name:
                    name = "(" + self.name + ")**(" + str(power) + ")"
                else:
                    name = self.name + "**(" + str(power) + ")"

            return Unit(name,
                        (power * self.exponents[0],
                         power * self.exponents[1],
                         power * self.exponents[2]),
                        self.denom**(-power),
                        self.numer**(-power),
                        power * self.pi_exponent)

    def __mul__(self, arg):
        if isinstance(arg, Unit): return self.mul(arg)

        if isinstance(arg, oops.UnitScalar):
            return arg * self

        if isinstance(arg, oops.Scalar):
            return oops.UnitScalar(arg, self)

        return oops.UnitScalar(oops.Scalar.as_scalar(arg).vals, self)

    def __div__(self, arg):
        if isinstance(arg, Unit): return self.div(arg)

        if isinstance(arg, oops.UnitScalar):
            return oops.UnitScalar(1.,self) / arg

        return oops.UnitScalar(1./oops.Scalar.as_scalar(arg).vals, self)

    def __eq__(self, arg):
        return (self.exponents == arg.exponents and self.factor == arg.factor)

    def __ne__(self, arg):
        return (self.exponents != arg.exponents or self.factor != arg.factor)

    def __pow__(self, power):
        return self.pow(power)

    def __str__(self):
        return "Unit(" + self.name + ")"

    def __repr__(self):
        return str(self)

########################################
# Define the most common units
########################################

UNITLESS    = Unit("",             (0,0,0), 1, 1, 0)

KILOMETERS  = Unit("kilometers",   (1,0,0), 1, 1, 0)
KM          = Unit("km",           (1,0,0), 1, 1, 0)
METERS      = Unit("meters",       (1,0,0), 1, 1000, 0)
M           = Unit("m",            (1,0,0), 1, 1000, 0)
CENTIMETERS = Unit("centimeters",  (1,0,0), 1, 100000, 0)
CM          = Unit("cm",           (1,0,0), 1, 100000, 0)

SECONDS     = Unit("seconds",      (0,1,0),     1, 1, 0)
SEC         = Unit("sec",          (0,1,0),     1, 1, 0)
S           = Unit("s",            (0,1,0),     1, 1, 0)
MINUTES     = Unit("minutes",      (0,1,0),    60, 1, 0)
MIN         = Unit("min",          (0,1,0),    60, 1, 0)
HOURS       = Unit("hours",        (0,1,0),  3600, 1, 0)
H           = Unit("h",            (0,1,0),  3600, 1, 0)
DAYS        = Unit("days",         (0,1,0), 86400, 1, 0)
D           = Unit("d",            (0,1,0), 86400, 1, 0)

RADIANS     = Unit("radians",      (0,0,1), 1, 1, 0)
RAD         = Unit("rad",          (0,0,1), 1, 1, 0)
MILLIRAD    = Unit("millirad",     (0,0,1), 1, 1000, 0)
MRAD        = Unit("mrad",         (0,0,1), 1, 1000, 0)
DEGREES     = Unit("degrees",      (0,0,1), 1,      180, 1)
DEG         = Unit("deg",          (0,0,1), 1,      180, 1)
ARCHOURS    = Unit("archours",     (0,0,1), 1,       12, 1)
ARCMINUTES  = Unit("arcminutes",   (0,0,1), 1,   180*60, 1)
ARCMIN      = Unit("arcmin",       (0,0,1), 1,   180*60, 1)
ARCSECONDS  = Unit("arcseconds",   (0,0,1), 1, 180*3600, 1)
ARCSEC      = Unit("arcsec",       (0,0,1), 1, 180*3600, 1)
ROTATIONS   = Unit("rotations",    (0,0,1), 2, 1, 1)
CYCLES      = Unit("cycles",       (0,0,1), 2, 1, 1)

########################################
# UNIT TESTS
########################################

class Test_Unit(unittest.TestCase):

    def runTest(self):

        self.assertEqual(str(KM),           "Unit(km)")
        self.assertEqual(str(KM*KM),        "Unit(km*km)")
        self.assertEqual(str(KM**2),        "Unit(km**2)")
        self.assertEqual(str(KM**(-2)),     "Unit(km**(-2))")
        self.assertEqual(str(KM/S),         "Unit(km/s)")
        self.assertEqual(str((KM/S)**2),    "Unit((km/s)**2)")
        self.assertEqual(str((KM/S)**(-2)), "Unit((km/s)**(-2))")

        self.assertEqual((KM/S).exponents, (1,-1,0))
        self.assertEqual((KM/S/S).exponents, (1,-2,0))

        self.assertEquals(KM.convert(3.,CM), 3.e5)
        self.assertTrue(np.all(KM.convert([1.,2.,3.], CM) ==
                               [1.e5, 2.e5, 3.e5]))

        self.assertTrue(np.all(DEGREES.convert([1.,2.,3.], ARCSEC) ==
                               [3600., 7200., 10800.]))

        self.assertTrue(np.all((DEG/S).convert([1.,2.,3.], ARCSEC/S) ==
                               [3600., 7200., 10800.]))

        self.assertTrue(np.all((DEG/H).convert([1.,2.,3.], ARCSEC/S) ==
                               [1., 2., 3.]))

        self.assertTrue(np.all((DEG*S).convert([1.,2.,3.], ARCSEC*H) ==
                               [1., 2., 3.]))

        self.assertTrue(np.all((DEG**2).convert([1.,2.,3.], ARCMIN*ARCSEC) ==
                               [3600*60, 3600*60*2, 3600*60*3]))

        eps = 1.e-15
        test = DEG.to_standard([1.,2.,3.])
        self.assertTrue(np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps))
        self.assertTrue(np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps))

        test = DEG.to_unit(test)
        self.assertTrue(np.all([1., 2., 3.] < test + eps))
        self.assertTrue(np.all([1., 2., 3.] > test - eps))

        self.assertTrue(np.all((KM/[1.,2.,4]).vals == [1., 0.5, 0.25]))
        self.assertTrue((KM/[1.,2.,4]).unit == KM)

        self.assertTrue(np.all((CM/[1.,2.,4]).vals == [1., 0.5, 0.25]))
        self.assertTrue((CM/[1.,2.,4]).unit == CM)

        self.assertTrue(np.all((CM*[1.,2.,4]).vals == [1., 2., 4.]))
        self.assertTrue((CM*[1.,2.,4]).unit == CM)

        self.assertFalse(CM == M)
        self.assertTrue(CM != M)
        self.assertTrue(M != SEC)
        self.assertEqual(M.factor, MRAD.factor)
        self.assertTrue(CM, Unit("whatever", (1,0,0), 10., 1.e6, 0))

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
