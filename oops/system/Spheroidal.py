################################################################################
# Spheroidal
#
# Added 1/24/12 (MRS)
################################################################################

import numpy as np
import unittest

import oops

class Spheroidal(oops.System):
    """A system that defines locations using (longitude, latitude, radius)
     coordinates."""

    def __init__(self, radius, flattening,
                       longitude=None, latitude=None, elevation=None):
        """Constructor for a spherical coordinate system applied to the surface
        of an oblate spheroid.

        Input:
            radius          equatorial radius in km.
            flattening      the flattening factor, (polar radius/equatorial
                            radius).
            longitude       a Longitude object.
            latitude        a Latitude object.
            elevation       a Distance object; distances are measured relative
                            to the surface of the spheroid.

        The three models independently describe the properties of each
        coordinate."""

        self.radius     = radius
        self.flattening = flattening
        self.coords     = (longitude, latitude, elevation)
        self.longnames  = ["longitude", "latitude", "elevation"]
        self.shortnames = ["theta", "phi", "z"]

    def as_vector3(self, theta, phi, z=0.):
        """Converts the given coordinates to a 3-vector."""

        # TBD
        pass

        # Normalize all coordinates
#         theta = self.coords[0].to_standard(theta)
#         phi   = self.coords[1].to_standard(phi)
#         z     = self.coords[2].to_standard(z)
# 
#         # Convert to vectors
#         shape = Array.broadcast_shape((theta, phi, z), [3])
#         array = np.empty(shape)
#         cos_phi = np.sin(phi.vals)
#         array[...,0] = r.vals * np.cos(theta.vals) * cos_phi
#         array[...,1] = r.vals * np.sin(theta.vals) * cos_phi
#         array[...,2] = r.vals * np.sin(phi.vals)
# 
#         return Vector3(array)

    def as_coords(self, vector3, axes=2, units=False):
        """Converts the specified Vector3 into a tuple of Scalar coordinates.
        """

        # TBD
        pass

#         vector3 = Vector3(vector3)
#         x = vector3.vals[...,0]
#         y = vector3.vals[...,1]
#         z = vector3.vals[...,2]
#         r = np.sqrt(x**2 + y**2 + z**2)
# 
#         theta = self.coords[0].to_coord(np.arctan(y,x), units)
#         phi   = self.coords[1].coord_value(np.arcsin(z/r), units)
# 
#         if axes > 2:
#             r = self.coords[2].coord_value(r, units)
#             return (theta, phi, r)
#         else:
#             return (theta, phi)

########################################
# UNIT TESTS
########################################

class Test_Spherical(unittest.TestCase):

    def runTest(self):

        print "\n\nCoordinateSystem.Spherical does not yet have unit tests\n"

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
