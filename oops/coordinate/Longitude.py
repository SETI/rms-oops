################################################################################
# class Longitude
#
# 1/24/12 (MRS) Added.
################################################################################

import oops

class Longitude(oops.Coordinate):
    """A Longitude is a Coordinate used to describe a rotation angle that cycles
    through 360 degrees.
    """

    def __init__(self, unit=None, format=None, minimum=0., reference=0.,
                                               retrograde=False):
        """The constructor for a Longitude Coordinate

        Input:
            unit        the Unit object used by the coordinate.
            format      the default Format object used by the coordinate.
            minimum     the minimum value of the longitude, in degrees; default
                        = 0. The maximum value is always larger by 360 degrees.
            reference   the reference location in standard coordinates
                        corresponding to a value of zero in this defined
                        coordinate system. Default is 0.
            retrograde  if True, longitudes are measured in the reverse
                        direction; default is False
        """

        if unit is None: unit = oops.Unit.DEG

        modulus = oops.UnitScalar(360.,oops.Unit.DEG).convert(unit).vals

        Coordinate.__init__(self, unit, format, minimum, minimum + modulus,
                            modulus, reference,
                            negated = retrograde)

        if self.unit.exponents != (0,0,1):
            raise ValueError("illegal unit for a Longitude coordinate: " +
                             unit.name)

################################################################################
