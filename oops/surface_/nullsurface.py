################################################################################
# oops/surface_/nullsurface.py: NullSurface subclass of class Surface
#
# 7/24/13 MRS - Created to serve as the surface object for bodies that have no
#   surfaces, such as barycenters.
################################################################################

import numpy as np

from oops.surface_.surface import Surface
from oops.array_ import *
import oops.registry as registry

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

        self.origin_id = registry.as_path_id(origin)
        self.frame_id  = registry.as_frame_id(frame)

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False):
        """Converts from position vectors in the internal frame to the surface
        coordinate system.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            obs         ignored.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate with respect to
                        surface position and observer position are returned as
                        well. Using a tuple, you can indicate whether to return
                        partial derivatives on an coordinate-by-coordinate
                        basis.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.

                        If derivs is True, then the coordinate has extra
                        attributes "d_dpos" and "d_dobs", which contain the
                        partial derivatives with respect to the surface position
                        and the observer position, represented as a MatrixN
                        objects with item shape [1,3].
        """

        # Derivatives are not supported
        if np.any(derivs):
            raise ValueError('Derivatives are not supported for a NullSurface')

        # Convert to a Vector3 and strip units, if any
        pos = Vector3.as_standard(pos)
        mask = pos.mask

        # Generate cylindrical coordinates
        x = pos.vals[...,0]
        y = pos.vals[...,1]
        z = pos.vals[...,2]

        # Derivatives are not supported
        if np.any(derivs):
            raise ValueError('Derivatives are not supported for a NullSurface')

        if axes > 2:
            return (Scalar(x,mask), Scalar(y,mask), Scalar(z,mask))
        else:
            return (Scalar(x,mask), Scalar(y,mask))

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars defining the coordinates
            obs         position of the observer in the surface frame; ignored.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to observer and to the coordinates.

        Return:         a unitless Vector3 of intercept points defined by the
                        coordinates.

                        If derivs is True, then pos is returned with subfields
                        "d_dobs" and "d_dcoords", where the former contains the
                        MatrixN of partial derivatives with respect to obs and
                        the latter is the MatrixN of partial derivatives with
                        respect to the coordinates. The MatrixN item shapes are
                        [3,3].
        """

        # Derivatives are not supported
        if np.any(derivs):
            raise ValueError('Derivatives are not supported for a NullSurface')

        # Convert to Scalars and strip units, if any
        x = Scalar.as_standard(coords[0])
        y = Scalar.as_standard(coords[1])

        if len(coords) == 2:
            z = Scalar(0.)
        else:
            z = Scalar.as_standard(coords[2])

        # Convert to vectors
        shape = Array.broadcast_shape((x, y, z))
        vals = np.empty(shape + [3])

        vals[...,0] = x.vals
        vals[...,1] = y.vals
        vals[...,2] = z.vals

        pos = Vector3(vals, x.mask | y.mask | z.mask)

        return pos

    def intercept(self, obs, los, derivs=False):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.

        Return:         (pos, t)
            pos         a unitless Vector3 of intercept points on the surface,
                        in km.
            t           a unitless Scalar of scale factors t such that:
                            position = obs + t * los

                        If derivs is True, then pos and t are returned with
                        subfields "d_dobs" and "d_dlos", where the former
                        contains the MatrixN of partial derivatives with respect
                        to obs and the latter is the MatrixN of partial
                        derivatives with respect to los. The MatrixN item shapes
                        are [3,3] for the derivatives of pos, and [1,3] for the
                        derivatives of t. For purposes of differentiation, los
                        is assumed to have unit length.
        """

        # Derivatives are not supported
        if np.any(derivs):
            raise ValueError('Derivatives are not supported for a NullSurface')

        # Standardize
        obs = Vector3.as_standard(obs)
        los = Vector3.as_standard(los)

        # Fill the buffers
        shape = Array.broadcast_shape((obs, los))
        pos = Vector3(np.zeros(shape + [3]), (obs == -los))
        t = Scalar(np.ones(shape), pos.mask)

        return (pos, t)

    def normal(self, pos, derivs=False):
        """Returns the normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            derivs      True to include a matrix of partial derivatives.

        Return:         a unitless Vector3 containing directions normal to the
                        surface that pass through the position. Lengths are
                        arbitrary.

                        If derivs is True, then the normal vectors returned have
                        a subfield "d_dpos", which contains the partial
                        derivatives with respect to components of the given
                        position vector, as a MatrixN object with item shape
                        [3,3].
        """

        # Derivatives are not supported
        if np.any(derivs):
            raise ValueError('Derivatives are not supported for a NullSurface')

        pos = Vector3.as_vector3(pos)
        mask = pos.mask

        # The normal is always the z-axis
        vals = np.zeros(pos.vals.shape)
        vals[...,2] = 1.
        perp = Vector3(vals, mask)

        return perp

    def velocity(self, pos):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 of velocities, in units of km/s.
        """

        pos = Vector3.as_vector3(pos)
        mask = pos.mask

        return Vector3(np.zeros(pos.shape + [3]), mask)

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
