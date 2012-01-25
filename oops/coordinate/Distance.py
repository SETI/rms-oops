################################################################################
# class Distance
#
# 1/24/12 (MRS) Added.
################################################################################

import numpy as np
import oops

class Distance(oops.Coordinate):
    """A Distance is a Coordinate subclass used to describe one component of a
    position vector, typically in rectangular coordinates.
    """

    def __init__(self, unit=None, format=None, reference=0., downward=False):
        """The constructor for a Distance Coordinate.

        Input:
            unit        the Unit object used by the coordinate.
            format      the default Format object used by the coordinate.
            reference   the reference location in standard coordinates
                        corresponding to a value of zero in this defined
                        coordinate system. Default is 0.
            downward    True to measure positions "downward" from the reference
                        position, i.e., in a direction opposite to the direction
                        of increase of the standard coordinate value.
        """

        if unit is None: unit = oops.Unit.KM

        Coordinate.__init__(self, unit, format,
                            minimum = -np.inf,
                            maximum =  np.inf,
                            modulus = None,
                            reference = reference,
                            negated = downward)

        if self.unit.exponents != (1,0,0):
            raise ValueError("illegal unit for a Distance coordinate: " +
                             str(unit))

################################################################################
