################################################################################
# Rectangular
#
# Added 1/24/12 (MRS)
################################################################################

import numpy as np
import unittest

import      oops

class Rectangular(oops.System):
    """A CoordinateSystem that defines locations using linear (x,y,z)
    coordinates."""

    def __init__(self, x = Coordinate.Distance(),
                       y = Coordinate.Distance(),
                       z = Coordinate.Distance()):
        """Constructor for a Rectangular Coordinate System.

        Input:
            x           a Coordinate.Distance object.
            y           a Coordinate.Distance object.
            z           a Coordinate.Distance object.

        The three models independently describe the properties of each
        coordinate."""

        self.models = (x, y, z)
        self.longnames  = ["x", "y", "z"]
        self.shortnames = ["x", "y", "z"]

    def as_vector3(self, x, y, z=Scalar(0.)):
        """Converts the given coordinates to a 3-vector."""

        # Normalize all coordinates
        x = self.models[0].normal_value(x)
        y = self.models[1].normal_value(y)
        z = self.models[2].normal_value(z)

        # Convert to vectors
        shape = Array.broadcast_shape((x, y, z), [3])
        array = np.empty(shape)
        array[...,0] = x
        array[...,1] = y
        array[...,2] = z

        return Vector3(array)

    def as_coordinates(self, vector3, axes=3):
        """Converts the specified Vector3 into a tuple of Scalar coordinates.
        """

        vector3 = Vector3(vector3)
        x = vector3.vals[...,0]
        y = vector3.vals[...,1]
        z = vector3.vals[...,2]
        r = np.sqrt(x**2 + y**2 + z**2)

        x = self.models[0].coord_value(x)
        y = self.models[1].coord_value(y)

        if axes > 2:
            z = self.models[2].coord_value(z)
            return (x, y, z)
        else:
            return (x, y)

########################################
# UNIT TESTS
########################################

class Test_Rectangular(unittest.TestCase):

    def runTest(self):

        print "\n\nCoordinateSystem.Rectangular does not yet have unit tests\n"

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
