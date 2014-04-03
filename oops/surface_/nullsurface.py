################################################################################
# oops/surface_/nullsurface.py: NullSurface subclass of class Surface
################################################################################

import numpy as np
from polymath import *

from oops.surface_.surface import Surface
from oops.path_.path       import Path
from oops.frame_.frame     import Frame

class NullSurface(Surface):
    """NullSurface is a subclass of Surface of describing an infinitesimal
    surface centered on the specified path, and using the specified coordinate
    frame."""

    COORDINATE_TYPE = "rectangular"

    def __init__(self, origin, frame):
        """Constructor for a NullSurface surface.

        Input:
            origin      a Path object or ID defining the motion of the center
                        of the ring plane.

            frame       a Frame object or ID in which the surface's "normal" is
                        defind by the z-axis.
            """

        self.origin = Path.as_waypoint(origin)
        self.frame  = Frame.as_wayframe(frame)

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer observer positions. Ignored for
                        solid surfaces but needed for virtual surfaces.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.
        """

        # Derivatives are not supported
        if np.any(derivs):
            raise ValueError('Derivatives are not supported for a NullSurface')

        # Simple rectangular coordinates
        pos = Vector3.as_vector3(pos)
        return pos.to_scalars(derivs)[:axes]

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the
                        coordinates.
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a unitless Vector3 of intercept points defined by the
                        coordinates.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
        """

        # Convert to Scalars and strip units, if any
        x = Scalar.as_scalar(coords[0], derivs)
        y = Scalar.as_scalar(coords[1], derivs)

        if len(coords) == 2:
            z = Scalar(0.)
        else:
            z = Scalar.as_scalar(coords[2], derivs)

        # Convert to a Vector3 and return
        return Vector3.from_scalars(x, y, z)

    def intercept(self, obs, los, derivs=False):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            t_guess     initial guess at the t array, optional.

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a unitless Scalar such that:
                            position = obs + t * los
        """

        obs = Vector3.as_scalar(obs, derivs)
        los = Vector3.as_scalar(los, derivs)
        shape = Qube.broadcasted_shape(obs, los, derivs=False)

        t = obs.to_scalar(0, derivs)
        t = t.all_masked(derivs)
        pos = obs.all_masked(derivs)

        return (pos, t)

    def normal(self, pos, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        pos = Vector3.as_vector3(pos, derivs)

        # Always the Z-axis
        return pos.all_constant((0.,0.,1.), derivs)

    def velocity(self, pos):
        """The local velocity vector at a point within the surface.

        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface.

        Return:         a unitless Vector3 of velocities, in units of km/s.
        """

        pos = Vector3.as_vector3(pos, False)

        # Always zero
        return pos.all_constant((0.,0.,0.), False)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_NullSurface(unittest.TestCase):

    pass        # TBD

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
