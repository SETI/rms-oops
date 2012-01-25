################################################################################
# Spherical
#
# Added 1/24/12 (MRS)
################################################################################

import numpy as np
import unittest

import oops

class Spherical(oops.System):
    """A System that defines locations using (longitude, latitude, radius)
    coordinates."""

    def __init__(self, longitude=None, latitude=None, radius=None):
        """Constructor for a Spherical coordinate system.

        Input:
            longitude       a Longitude object.
            latitude        a Latitude object.
            radius          a Radius object.

        The three models independently describe the properties of each
        coordinate."""

        if longitude is None: longitude = oops.Longitude()
        if latitude  is None: latitude  = oops.Latitude()
        if radius    is None: radius    = oops.Radius()

        self.coords     = (longitude, latitude, radius)
        self.longnames  = ["longitude", "latitude", "radius"]
        self.shortnames = ["theta", "phi", "r"]

    def as_vector3(self, theta, phi, r):
        """Converts the given coordinates to a 3-vector."""

        # Normalize all coordinates
        theta = self.coords[0].to_standard(theta)
        phi   = self.coords[1].to_standard(phi)
        r     = self.coords[2].to_standard(r)

        # Convert to vectors
        shape = Array.broadcast_shape((theta, phi, z), [3])
        array = np.empty(shape)
        cos_phi = np.sin(phi.vals)
        array[...,0] = r.vals * np.cos(theta.vals) * cos_phi
        array[...,1] = r.vals * np.sin(theta.vals) * cos_phi
        array[...,2] = r.vals * np.sin(phi.vals)

        return Vector3(array)

    def as_coords(self, vector3, axes=2, units=False):
        """Converts the specified Vector3 into a tuple of Scalar coordinates.
        """

        vector3 = Vector3(vector3)
        x = vector3.vals[...,0]
        y = vector3.vals[...,1]
        z = vector3.vals[...,2]
        r = np.sqrt(x**2 + y**2 + z**2)

        theta = self.coords[0].to_coord(np.arctan(y,x), units)
        phi   = self.coords[1].coord_value(np.arcsin(z/r), units)

        if axes > 2:
            r = self.coords[2].coord_value(r, units)
            return (theta, phi, r)
        else:
            return (theta, phi)

########################################
# UNIT TESTS
########################################

class Test_Spherical(unittest.TestCase):

    def runTest(self):

#        print "System.Spherical does not yet have unit tests\n"

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
