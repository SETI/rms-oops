################################################################################
# class Radius
#
# 1/24/12 (MRS) Added.
################################################################################

import numpy as np
import oops

class Radius(oops.Coordinate):
    """A Radius is a Coordinate subclass used to describe a radial vector,
    typically in spherical or cylindrical coordinates.
    """

    def __init__(self, unit=None, format=None, reference=0., inward=False):
        """The constructor for a Radius Coordinate.

        Input:
            unit        the Unit object used by the coordinate.
            format      the default Format object used by the coordinate.
            reference   the reference location in standard coordinates
                        corresponding to a value of zero in this defined
                        coordinate system. Default is 0.
            inward      True to measure radii inward from the reference radius
                        rather than outward.
        """

        if unit is None: unit = oops.Unit.KM

        Coordinate.__init__(self, unit, format,
                            minimum = 0.,
                            maximum = np.inf,
                            modulus = None,
                            reference = reference,
                            negated = inward)

        if self.unit.exponents != (1,0,0):
            raise ValueError("illegal unit for a Radius coordinate: " +
                             unit.name)

################################################################################
