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

    def __init__(self, unit=oops.Unit.DEG, format=None, reference=0.
                                           southward=False):
        """The constructor for a Latitude Coordinate

        Input:
            unit        the Unit object used by the coordinate.
            format      the default Format object used by the coordinate.
            reference   the reference location in standard coordinates
                        corresponding to a value of zero in this defined
                        coordinate system. Default is 0.
            southward   if True, latitudes are measured in the reverse
                        direction; default is False
        """

        Coordinate.__init__(self, unit, format,
                            minimum = oops.UnitScalar(-90., oops.Unit.DEG),
                        modulus   = None,
                        reference = reference,
                        negated   = southward)

        if self.exponents != (0,0,1):
            raise ValueError("illegal unit for a Latitude coordinate: " +
                             str(unit))

################################################################################
