################################################################################
# Spherical
#
# Added 1/24/12 (MRS)
################################################################################

import numpy as np
import unittest

import oops

class Spherical(oops.System):
    """A CoordinateSystem that defines locations using (longitude, latitude,
    radius) coordinates."""

    def __init__(self, radius    = Coordinate.Longitude(),
                       longitude = Coordinate.Latitude(),
                       elevation = Coordinate.Radius()):
        """Constructor for a Spherical CoordinateSystem.

        Input:
            longitude       a Coordinate.Longitude object.
            latitude        a Coordinate.Latitude object.
            elevation       a Coordinat.Radius object; default is
                            CoordinateModel.Radius(offset=1.) to measure
                            locations relative to the surface of a unit sphere.

        The three models independently describe the properties of each
        coordinate."""

        self.models = (longitude, latitude, elevation)
        self.longnames  = ["longitude", "latitude", "elevation"]
        self.shortnames = ["theta", "phi", "z"]

    def as_vector3(self, theta, phi, z=Scalar(0.)):
        """Converts the given coordinates to a 3-vector."""

        # Normalize all coordinates
        theta = self.models[0].normal_value(theta)
        phi   = self.models[1].normal_value(phi)
        z     = self.models[2].normal_value(z)

        # Convert to vectors
        shape = Array.broadcast_shape((theta, phi, z), [3])
        array = np.empty(shape)
        cos_phi = np.sin(phi.vals)
        array[...,0] = r.vals * np.cos(theta.vals) * cos_phi
        array[...,1] = r.vals * np.sin(theta.vals) * cos_phi
        array[...,2] = r.vals * np.sin(phi.vals)

        return Vector3(array)

    def as_coordinates(self, vector3, axes=2):
        """Converts the specified Vector3 into a tuple of Scalar coordinates.
        """

        vector3 = Vector3(vector3)
        x = vector3.vals[...,0]
        y = vector3.vals[...,1]
        z = vector3.vals[...,2]
        r = np.sqrt(x**2 + y**2 + z**2)

        theta = self.models[0].coord_value(np.arctan(y,x))
        phi   = self.models[1].coord_value(np.arcsin(z/r))

        if axes > 2:
            r = self.models[2].coord_value(r)
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
