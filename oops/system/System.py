################################################################################
# class System
#
# 1/24/12 (MRS) - Drafted.
################################################################################

import oops

class System(object):
    """A System is an abstract class that describes method for using Scalars to
    specify a Vector3 object. Examples include cylindrical and spherical
    coordinates. The coordinate values can also have arbitrary units and print
    formats.
    """

    def __init__(self): pass

    def as_vector3(self, coord1, coord2, coord3=0.):
        """Converts the specified coordinates into a Vector3 object."""

        pass

    def as_coordinates(self, vector3, axes=3):
        """Converts the specified Vector3 into a tuple of either 2 or 3 Scalar
        coordinates."""

        pass

################################################################################
