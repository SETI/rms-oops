################################################################################
# Class Units
#
# 1/24/12 (MRS): Initial version.
# 2/2/12 (MRS): Creates betterunits units strings; performs more arithmetic
#   operations.
################################################################################

import numpy as np
import fractions
import numbers

class Units(object):
    """Units is a class defining units names and the methods for converting
    between values that include units."""

    UNNAMED = "UNNAMED"

    def __init__(self, exponents, triple, name=None):
        """Constructor for a Units object.

        Input:
            exponents       a tuple of integers defining the exponents on
                            distance, time and angle that are used for this
                            set of units.
            triple          a tuple containing:
                [0]         the numerator of a factor that converts from a value
                            in these units to a value in standard units of km,
                            seconds and/or radians.
                [1]         the denominator of this same factor.
                [2]         the exponent on pi that should multiply the
                            numerator of this factor.
            name            the name of the units (optional). If not specified,
                            the units will be checked against a list of
                            recognized units, and a name will be filled in if it
                            is recognized. Otherwise, the name will be set to
                            "UNNAMED".

        For example, units of degrees would have a triple (1,180,1). This
        defines a factor pi/180, which converts from degrees to radians.
        """

        self.exponents = tuple(exponents)

        gcd = fractions.gcd(triple[0], triple[1])
        self.numer = triple[0] / gcd
        self.denom = triple[1] / gcd
        self.pi_expo = triple[2]

        self.triple = (self.numer, self.denom, self.pi_expo)

        # Factor to convert from these units to standard units
        self.factor = (self.numer / float(self.denom)) * np.pi**self.pi_expo

        # Factor to convert from standard units to these units
        self.factor_inv = (self.denom / float(self.numer)) / np.pi**self.pi_expo

        # Attempt to fill in a valid name
        self.name = name
        if self.name is None: self.name = self.get_name()

    def get_name(self):
        """Returns a valid name for the units as best it can."""

        if self.name is not None and self.name != Units.UNNAMED:
            return self.name

        try:
            return Units.TUPLES_TO_UNIT[(self.exponents, self.triple)].name
        except KeyError:
            return Units.UNNAMED

    @staticmethod
    def as_units(arg):
        """Converts the given argument to a string. It can be an object of class
        Unit or one of the standard unit names. An argument of None returns
        None."""

        if arg is None:
            return None
        elif type(arg) == type(""):
            return Units.NAME_TO_UNIT[arg]
        elif type(arg) == Units:
            return arg
        else:
            raise ValueError("object is not a recognized unit: " + str(arg))

    @staticmethod
    def can_match(first, second):
        """Returns True if the units can match, meaning that either they have
        the same exponents or one or both are None."""

        if first is None or second is None: return True
        return first.exponents == second.exponents

    @staticmethod
    def do_match(first, second):
        """Returns True if the units match, meaning that they have the same
        exponents. Values of None are treated as equivalent to nnitless."""

        if first  is None: first  = Units.UNITLESS
        if second is None: second = Units.UNITLESS

        return first.exponents == second.exponents

    def to_standard(self, value):
        """Converts a scalar or numpy array with units to one in standard units
        of km, seconds and radians."""

        return self.factor * value

    def to_units(self, value):
        """Converts a scalar or numpy array in standard units to one with the
        specified units.
        """

        return self.factor_inv * value

    def convert(self, value, units):
        """Converts a scalar or numpy array in these units to a value in the
        given units. Conversions are exact whenever possible."""

        if units is None: units = Units.UNITLESS
        if self.exponents != units.exponents:
            raise ValueError("cannot convert units " + self.name + " to " +
                                                       units.name)

        # If the factor is unity, return the value without modification
        if (self.pi_expo == units.pi_expo and
            self.numer * units.denom == self.denom * units.numer):
                return value

        return (float(self.numer * units.denom) * value /
                float(self.denom * units.numer) *
                np.pi**(self.pi_expo - units.pi_expo))

    def mul_unit(self, arg, name=None):
        """Returns a Units object constructed as the product of two other Units.
        """

        result = Units((self.exponents[0] + arg.exponents[0],
                        self.exponents[1] + arg.exponents[1],
                        self.exponents[2] + arg.exponents[2]),
                       (self.numer * arg.numer,
                        self.denom * arg.denom,
                        self.pi_expo + arg.pi_expo), name)

        if (result.name == Units.UNNAMED and self.name != Units.UNNAMED
                                         and arg.name  != Units.UNNAMED):
            result.name = self.name + "*" + arg.name

        return result

    def div_unit(self, arg, name=None):
        """Returns a Units object constructed as the ratio of two other Units.
        """

        result = Units((self.exponents[0] - arg.exponents[0],
                        self.exponents[1] - arg.exponents[1],
                        self.exponents[2] - arg.exponents[2]),
                       (self.numer * arg.denom,
                        self.denom * arg.numer,
                        self.pi_expo - arg.pi_expo), name)

        if (result.name == Units.UNNAMED and self.name != Units.UNNAMED
                                         and arg.name  != Units.UNNAMED):
            result.name = self.name + "/" + arg.name

        return result

    def to_power(self, power, name=None):
        """Returns a Units object constructed as a power of another Units
        object."""

        if power > 0:
            result = Units((power * self.exponents[0],
                            power * self.exponents[1],
                            power * self.exponents[2]),
                           (self.numer**power,
                            self.denom**power,
                            power * self.pi_expo), name)
        else:
            result = Units((power * self.exponents[0],
                            power * self.exponents[1],
                            power * self.exponents[2]),
                           (self.denom**(-power),
                            self.numer**(-power),
                            power * self.pi_expo), name)

        if result.name == Units.UNNAMED and self.name != Units.UNNAMED:
            if power > 0:
                if "*" in self.name or "/" in self.name:
                    result.name = "(" + self.name + ")**" + str(power)
                else:
                    result.name = self.name + "**" + str(power)
            else:
                if "*" in self.name or "/" in self.name:
                    result.name = "(" + self.name + ")**(" + str(power) + ")"
                else:
                    result.name = self.name + "**(" + str(power) + ")"

        return result

    def __mul__(self, arg):
        if isinstance(arg, Units): return self.mul_unit(arg)

        if isinstance(arg, numbers.Real):
            return self.mul_unit(Units((0,0,0),(arg,1,0)))

        return  NotImplemented

    def __rmul__(self, arg):
        return self.__mul__(arg)

    def __div__(self, arg):
        if isinstance(arg, Units): return self.div_unit(arg)

        if isinstance(arg, numbers.Number):
            return self.mul_unit(Units((0,0,0),(1,arg,0)))

        return NotImplemented

    def __rdiv__(self, arg):
        if isinstance(arg, numbers.Number):
            return self.__div__(arg)**(-1)

    def __eq__(self, arg):
        if not isinstance(arg, Units): return False
        return (self.exponents == arg.exponents and self.factor == arg.factor)

    def __ne__(self, arg):
        if not isinstance(arg, Units): return True
        return (self.exponents != arg.exponents or self.factor != arg.factor)

    def __pow__(self, power):
        return self.to_power(power)

    def __str__(self):
        return "Units(" + self.name + ")"

    def __repr__(self):
        return str(self)

########################################
# Define the most common units
########################################

Units.UNITLESS    = Units((0,0,0), (1, 1, 0), "")

Units.KM          = Units((1,0,0), (1,          1, 0), "km")
Units.KILOMETER   = Units((1,0,0), (1,          1, 0), "kilometer")
Units.KILOMETERS  = Units((1,0,0), (1,          1, 0), "kilometers")
Units.M           = Units((1,0,0), (1,       1000, 0), "m")
Units.METER       = Units((1,0,0), (1,       1000, 0), "meter")
Units.METERS      = Units((1,0,0), (1,       1000, 0), "meters")
Units.CM          = Units((1,0,0), (1,     100000, 0), "cm")
Units.CENTIMETER  = Units((1,0,0), (1,     100000, 0), "centimeter")
Units.CENTIMETERS = Units((1,0,0), (1,     100000, 0), "centimeters")
Units.MM          = Units((1,0,0), (1,    1000000, 0), "mm")
Units.MILLIMETER  = Units((1,0,0), (1,    1000000, 0), "millimeter")
Units.MILLIMETERS = Units((1,0,0), (1,    1000000, 0), "millimeters")
Units.MICRON      = Units((1,0,0), (1, 1000000000, 0), "micron")
Units.MICRONS     = Units((1,0,0), (1, 1000000000, 0), "microns")

Units.S           = Units((0,1,0), (    1,    1, 0), "s")
Units.SEC         = Units((0,1,0), (    1,    1, 0), "sec")
Units.SECOND      = Units((0,1,0), (    1,    1, 0), "second ")
Units.SECONDS     = Units((0,1,0), (    1,    1, 0), "seconds")
Units.MIN         = Units((0,1,0), (   60,    1, 0), "min")
Units.MINUTE      = Units((0,1,0), (   60,    1, 0), "minute")
Units.MINUTES     = Units((0,1,0), (   60,    1, 0), "minutes")
Units.H           = Units((0,1,0), ( 3600,    1, 0), "h")
Units.HOUR        = Units((0,1,0), ( 3600,    1, 0), "hour")
Units.HOURS       = Units((0,1,0), ( 3600,    1, 0), "hours")
Units.D           = Units((0,1,0), (86400,    1, 0), "d")
Units.DAY         = Units((0,1,0), (86400,    1, 0), "day")
Units.DAYS        = Units((0,1,0), (86400,    1, 0), "days")
Units.MS          = Units((0,1,0), (    1, 1000, 0), "ms")
Units.MSEC        = Units((0,1,0), (    1, 1000, 0), "msec")

Units.RAD         = Units((0,0,1), (1,        1, 0), "rad")
Units.RADIAN      = Units((0,0,1), (1,        1, 0), "radian")
Units.RADIANS     = Units((0,0,1), (1,        1, 0), "radians")
Units.MRAD        = Units((0,0,1), (1,     1000, 0), "mrad")
Units.MILLIRAD    = Units((0,0,1), (1,     1000, 0), "millirad")
Units.DEG         = Units((0,0,1), (1,      180, 1), "deg")
Units.DEGREE      = Units((0,0,1), (1,      180, 1), "degree")
Units.DEGREES     = Units((0,0,1), (1,      180, 1), "degrees")
Units.ARCHOUR     = Units((0,0,1), (1,       12, 1), "archour")
Units.ARCHOURS    = Units((0,0,1), (1,       12, 1), "archours")
Units.ARCMIN      = Units((0,0,1), (1,   180*60, 1), "arcmin")
Units.ARCMINUTE   = Units((0,0,1), (1,   180*60, 1), "arcminute")
Units.ARCMINUTES  = Units((0,0,1), (1,   180*60, 1), "arcminutes")
Units.ARCSEC      = Units((0,0,1), (1, 180*3600, 1), "arcsec")
Units.ARCSECOND   = Units((0,0,1), (1, 180*3600, 1), "arcsecond")
Units.ARCSECONDS  = Units((0,0,1), (1, 180*3600, 1), "arcseconds")
Units.REV         = Units((0,0,1), (2,        1, 1), "rev")
Units.REVS        = Units((0,0,1), (2,        1, 1), "revs")
Units.ROTATION    = Units((0,0,1), (2,        1, 1), "rotation")
Units.ROTATIONS   = Units((0,0,1), (2,        1, 1), "rotations")
Units.CYCLE       = Units((0,0,1), (2,        1, 1), "cycle")
Units.CYCLES      = Units((0,0,1), (2,        1, 1), "cycles")

# Create dictionaries to convert between name and units
Units.NAME_TO_UNIT = {}
Units.TUPLES_TO_UNIT = {}

# Assemble a list of all the recognized units
Units.STANDARD_LIST = [Units.UNITLESS,
    Units.KM, Units.M, Units.CM, Units.MM, Units.MICRON,
    Units.S, Units.D, Units.H, Units.MIN, Units.MSEC,
    Units.RAD, Units.MRAD,
    Units.DEG, Units.ARCSEC, Units.ARCMIN, Units.ARCHOUR, Units.ROTATION,
    Units.KM/Units.S, Units.M/Units.S, Units.CM/Units.S, Units.MM/Units.S,
    Units.RAD/Units.S, Units.MRAD/Units.S,
    Units.DEG/Units.S, Units.DEG/Units.DAY, Units.DEG/Units.MIN,
        Units.DEG/Units.H,
    Units.ROTATION/Units.D, Units.ROTATION/Units.H, Units.ROTATION/Units.MIN,
        Units.ROTATION/Units.S,
    Units.KM**2, Units.M**2, Units.CM**2,
    Units.KM**3, Units.M**3, Units.CM**3,
    Units.KM/Units.S**2, Units.M/Units.S**2, Units.CM/Units.S**2,
        Units.MM/Units.S**2,
    Units.KM**3/Units.S**2]

# Fill in the dictionaries
for units in Units.STANDARD_LIST:
    Units.NAME_TO_UNIT[units.name] = units
    Units.TUPLES_TO_UNIT[(units.exponents, units.triple)] = units

########################################
# UNIT TESTS
########################################

import unittest

class Test_Units(unittest.TestCase):

    def runTest(self):

        self.assertEqual(str(Units.KM),                 "Units(km)")
        self.assertEqual(str(Units.KM*Units.KM),        "Units(km**2)")
        self.assertEqual(str(Units.KM**2),              "Units(km**2)")
        self.assertEqual(str(Units.KM**(-2)),           "Units(km**(-2))")
        self.assertEqual(str(Units.KM/Units.S),         "Units(km/s)")
        self.assertEqual(str((Units.KM/Units.S)**2),    "Units((km/s)**2)")
        self.assertEqual(str((Units.KM/Units.S)**(-2)), "Units((km/s)**(-2))")

        self.assertEqual((Units.KM/Units.S).exponents, (1,-1,0))
        self.assertEqual((Units.KM/Units.S/Units.S).exponents, (1,-2,0))

        self.assertEquals(Units.KM.convert(3.,Units.CM), 3.e5)
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
        test = Units.DEG.to_standard(np.array([1.,2.,3.]))
        self.assertTrue(np.all([np.pi/180., np.pi/90., np.pi/60.] < test + eps))
        self.assertTrue(np.all([np.pi/180., np.pi/90., np.pi/60.] > test - eps))

        test = Units.DEG.to_units(test)
        self.assertTrue(np.all(np.array([1., 2., 3.]) < test + eps))
        self.assertTrue(np.all(np.array([1., 2., 3.]) > test - eps))

        self.assertFalse(Units.CM == Units.M)
        self.assertTrue( Units.CM != Units.M)
        self.assertTrue( Units.M  != Units.SEC)
        self.assertEqual(Units.M.factor, Units.MRAD.factor)
        self.assertTrue( Units.CM, Units((1,0,0), (10., 1.e6, 0)))

        test = Units.ROTATION/Units.S
        test.name = None
        self.assertEqual(test.get_name(), "rotation/s")

        self.assertEqual(str(Units.KM**3/Units.S*Units.RAD*Units.KM**(-2) /
                             Units.RAD), "Units(km/s)")

        self.assertEqual(str(Units.KM**3/Units.S*Units.RAD*Units.KM**(-2) /
                             Units.MRAD*Units.MSEC/(Units.KM/Units.S) /
                             Units.S), "Units()")

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

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
