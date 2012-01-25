################################################################################
# Cylindrical
#
# Added 1/24/12 (MRS)
################################################################################

import numpy as np
import unittest

import oops

class Cylindrical(oops.System):
    """A Cylindrical System is one that defines locations using (radius,
    longitude, elevation) coordinates. Dimensions are (distance, angle,
    distance)."""

    def __init__(self, radius=None, longitude=None, elevation=None):
        """Constructor for a Cylindrical System.

        Input:
            radius          a Radius object.
            longitude       a Longitude object.
            elevation       a Distance object.

        The three objects independently describe the properties of each
        coordinate."""

        if radius    is None: radius = oops.Radius()
        if longitude is None: longitude = oops.Longitude()
        if elevation is None: elevation = oops.Distance()

        self.coords     = (radius, longitude, elevation)
        self.longnames  = ["radius", "longitude", "elevation"]
        self.shortnames = ["r", "theta", "z"]

    def as_vector3(self, r, theta, z=oops.Scalar(0.)):
        """Converts the given coordinates to a 3-vector."""

        # Normalize all coordinates
        r     = self.coords[0].to_standard(r)
        theta = self.coords[1].to_standard(theta)
        z     = self.coords[2].to_standard(z)

        # Convert to vectors
        shape = Array.broadcast_shape((r, theta, z), [3])
        array = np.empty(shape)
        array[...,0] = np.cos(theta.vals) * r.vals
        array[...,1] = np.sin(theta.vals) * r.vals
        array[...,2] = z.vals

        return Vector3(array)

    def as_coords(self, vector3, axes=2, units=False):
        """Converts the specified Vector3 into a tuple of Scalar coordinates.
        """

        vector3 = Vector3(vector3)
        x = vector3.vals[...,0]
        y = vector3.vals[...,1]

        r     = self.coords[0].to_coord(np.sqrt(x**2 + y**2), units)
        theta = self.coords[1].to_coord(np.arctan(y,x), units)

        if axes > 2:
            z = self.coords[2].to_coord(vector.vals[...,2], units)
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
