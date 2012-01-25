################################################################################
# Rectangular
#
# Added 1/24/12 (MRS)
################################################################################

import numpy as np
import unittest

import oops

class Rectangular(oops.System):
    """A System that defines locations using linear (x,y,z) coordinates."""

    def __init__(self, x=None, y=None, z=None):
        """Constructor for a Rectangular coordinate system.

        Input:
            x           a Distance object.
            y           a Distance object.
            z           a Distance object.

        The three models independently describe the properties of each
        coordinate."""

        if x is None: x = oops.Distance()
        if y is None: y = oops.Distance()
        if z is None: z = oops.Distance()

        self.coords     = (x, y, z)
        self.longnames  = ["x", "y", "z"]
        self.shortnames = ["x", "y", "z"]

    def as_vector3(self, x, y, z=0.):
        """Converts the given coordinates to a 3-vector."""

        # Normalize all coordinates
        x = self.coords[0].to_standard(x)
        y = self.coords[1].to_standard(y)
        z = self.coords[2].to_standard(z)

        # Convert to vectors
        shape = Array.broadcast_shape((x, y, z), [3])
        array = np.empty(shape)
        array[...,0] = x
        array[...,1] = y
        array[...,2] = z

        return Vector3(array)

    def as_coords(self, vector3, axes=3, units=False):
        """Converts the specified Vector3 into a tuple of Scalar coordinates.
        """

        vector3 = Vector3(vector3)
        x = vector3.vals[...,0]
        y = vector3.vals[...,1]
        z = vector3.vals[...,2]
        r = np.sqrt(x**2 + y**2 + z**2)

        x = self.coords[0].to_coord(x, units)
        y = self.coords[1].to_coord(y, units)

        if axes > 2:
            z = self.coords[2].to_coord(z, units)
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
