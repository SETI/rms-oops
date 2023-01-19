################################################################################
# oops/surface/nullsurface.py: NullSurface subclass of class Surface
################################################################################

from polymath     import Scalar, Vector3
from oops.frame   import Frame
from oops.path    import Path
from oops.surface import Surface

class NullSurface(Surface):
    """A subclass of Surface of describing an infinitesimal surface centered on
    the specified path, and using the specified coordinate frame.
    """

    COORDINATE_TYPE = 'rectangular'

    #===========================================================================
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

        self.unmasked = self

        # Unique key for intercept calculations
        self.intercept_key = ('null', self.origin.waypoint,
                                      self.frame.wayframe)

    def __getstate__(self):
        return (Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame))

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False):
        """Convert positions in the internal frame to surface coordinates.

        Input:
            pos         a Vector3 of positions at or near the surface.
            obs         a Vector3 of observer positions. Ignored for solid
                        surfaces but needed for virtual surfaces.
            time        a Scalar time at which to evaluate the surface; ignored.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.
        """

        if axes not in (2, 3):
            raise ValueError('Surface.coords_from_vector3 ' +
                             'axes values must equal 2 or 3')

        # Simple rectangular coordinates
        pos = Vector3.as_vector3(pos, derivs)
        return pos.to_scalars(derivs)[:axes]

    #===========================================================================
    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Convert surface coordinates to positions in the internal frame.

        Input:
            coords      a tuple of two or three Scalars defining the
                        coordinates.
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of intercept points defined by the
                        coordinates.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
        """

        if len(coords) not in (2, 3):
            raise ValueError('Surface.vector3_from_coords requires 2 or 3 '
                             'coords')

        # Convert to Scalars and strip units, if any
        x = Scalar.as_scalar(coords[0], derivs)
        y = Scalar.as_scalar(coords[1], derivs)

        if len(coords) == 2:
            z = Scalar(0.)
        else:
            z = Scalar.as_scalar(coords[2], derivs)

        # Convert to a Vector3 and return
        return Vector3.from_scalars(x, y, z)

    #===========================================================================
    def intercept(self, obs, los, time=None, derivs=False, guess=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3.
            los         line of sight as a Vector3.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       unused.

        Return:         a tuple (pos, t) where
            pos         a Vector3 of intercept points on the surface, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
        """

        # This is a quick way to create a position vector of the correct shape,
        # and with the correct set of derivatives, even though it will be
        # entirely masked.

        pos = Vector3.as_vector(obs, derivs) + Vector3.as_vector(los, derivs)
        t = pos.to_scalar(0, derivs)

        pos = pos.as_all_constant(1.).as_all_masked()
        t = t.as_all_constant(0.).as_all_masked()

        return (pos, t)

    #===========================================================================
    def normal(self, pos, time=None, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface.
            time        a Scalar time at which to evaluate the surface; ignored.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.
        """

        # Always the Z-axis
        return Vector3.ZAXIS

    #===========================================================================
    def velocity(self, pos, time=None):
        """The local velocity vector at a point within the surface.

        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface.
            time        a Scalar time at which to evaluate the surface; ignored.

        Return:         a Vector3 of velocities, in units of km/s.
        """

        # Always zero
        return Vector3.ZERO

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
