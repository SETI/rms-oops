################################################################################
# Cylindrical
#
# Added 1/24/12 (MRS)
################################################################################

import numpy as np
import unittest

from oops.broadcastable.UnitScalar import UnitScalar
from oops.broadcastable.Vector3    import Vector3
import oops

class Cylindrical(System):
    """A Cylindrical System is one that defines locations using (radius,
    longitude, elevation) coordinates. Dimensions are (distance, angle,
    distance)."""

    def __init__(self, radius    = Coordinate.Radius(),
                       longitude = Coordinate.Longitude(),
                       elevation = Coordinate.Distance()):
        """Constructor for a Cylindrical System.

        Input:
            radius          a Coordinate.Radius object.
            longitude       a Coordinate.Longitude object.
            elevation       a Coordinate.Distance object.

        The three objects independently describe the properties of each
        coordinate."""

        self.models = (radius, longitude, elevation)
        self.longnames  = ["radius", "longitude", "elevation"]
        self.shortnames = ["r", "theta", "z"]

    def as_vector3(self, r, theta, z=Scalar(0.)):
        """Converts the given coordinates to a 3-vector."""

        # Normalize all coordinates
        r     = self.models[0].normal_value(r)
        theta = self.models[1].normal_value(theta)
        z     = self.models[2].normal_value(z)

        # Convert to vectors
        shape = Array.broadcast_shape((r, theta, z), [3])
        array = np.empty(shape)
        array[...,0] = np.cos(theta.vals) * r.vals
        array[...,1] = np.sin(theta.vals) * r.vals
        array[...,2] = z.vals

        return Vector3(array)

    def as_coordinates(self, vector3, axes=2):
        """Converts the specified Vector3 into a tuple of Scalar coordinates.
        """

        vector3 = Vector3(vector3)
        x = vector3.vals[...,0]
        y = vector3.vals[...,1]

        r     = self.models[0].coord_value(np.sqrt(x**2 + y**2))
        theta = self.models[1].coord_value(np.arctan(y,x))

        if axes > 2:
            z = self.models[2].coord_value(vector.vals[...,2])
            return (r, theta, z)
        else:
            return (r, theta)

########################################
# UNIT TESTS
########################################

class Test_Cylindrical(unittest.TestCase):

    def runTest(self):

        print "\n\nCoordinateSystem.Cylindrical does not yet have unit tests\n"

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
