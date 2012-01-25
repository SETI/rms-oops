################################################################################
# class Latitude
#
# 1/24/12 (MRS) Added.
################################################################################

import oops

class Latitude(oops.Coordinate):
    """A Latitude is a Coordinate subclass used to describe an angle between -90
    and +90 degrees.
    """

    def __init__(self, unit=None, format=None, southward=False):
        """The constructor for a Latitude Coordinate

        Input:
            unit        the Unit object used by the coordinate.
            format      the default Format object used by the coordinate.
            southward   if True, latitudes are measured in the reverse
                        direction; default is False
        """

        if unit is None: unit = oops.Unit.DEG

        minimum = oops.UnitScalar(-90., oops.Unit.DEG).convert(unit).vals
        maximum = oops.UnitScalar( 90., oops.Unit.DEG).convert(unit).vals

        Coordinate.__init__(self, unit, format, minimum, maximum,
                            modulus   = None,
                            reference = 0.,
                            negated   = southward)

        if self.unit.exponents != (0,0,1):
            raise ValueError("illegal unit for a Latitude coordinate: " +
                             unit.name)

################################################################################
