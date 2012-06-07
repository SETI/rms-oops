################################################################################
# oops_/surface/ellipsoid.py: Ellipsoid subclass of class Surface
#
# 2/17/12 Created (MRS)
# 3/4/12 MRS: cleaned up comments, added NotImplementedErrors for features still
#   TBD.
################################################################################

import numpy as np

from surface_ import Surface
from oops_.array.all import *
import oops_.registry as registry

class Ellipsoid(Surface):
    """Ellipsoid defines a ellipsoidal surface centered on the given path and
    fixed with respect to the given frame. The short radius of the ellipsoid is
    oriented along the Z-axis of the frame and the long radius is along the
    X-axis.

    The coordinates defining the surface grid are (longitude, latitude), based
    on the assumption that a spherical body has been "squashed" along the Y- and
    Z-axes. The latitudes and longitudes defined in this manner are neither
    planetocentric nor planetographic; functions are provided to perform the
    conversion to either choice. Longitudes are measured in a right-handed
    manner, increasing toward the east. Values range from -pi to pi.

    Elevations are defined by "unsquashing" the radial vectors and then
    subtracting off the equatorial radius of the body. Thus, the surface is
    defined as the locus of points where elevation equals zero. However, the
    direction of increasing elevation is not exactly normal to the surface.
    """

    COORDINATE_TYPE = "spherical"

    # Class constants to override where derivs are undefined
    coords_from_vector3_DERIVS_ARE_IMPLEMENTED = False
    vector3_from_coords_DERIVS_ARE_IMPLEMENTED = False
    intercept_DERIVS_ARE_IMPLEMENTED = False
    normal_DERIVS_ARE_IMPLEMENTED = False
    intercept_with_normal_DERIVS_ARE_IMPLEMENTED = False
    intercept_normal_to_DERIVS_ARE_IMPLEMENTED = False

    def __init__(self, origin, frame, radii):
        """Constructor for an Ellipsoid object.
            
        Input:
            origin      a Path object or ID defining the motion of the center
                        of the ring plane.

            frame       a Frame object or ID in which the ring plane is the
                        (x,y) plane (where z = 0).

            radii       a tuple (a,b,c) containing the radii from longest to
                        shortest, in km.
        """

        self.origin_id = registry.as_path_id(origin)
        self.frame_id  = registry.as_frame_id(frame)

        self.radii  = np.asfarray(radii)
        self.req    = radii[0]
        self.req_sq = self.req**2

        self.squash_y   = radii[1] / radii[0]
        self.unsquash_y = radii[0] / radii[1]

        self.squash_z   = radii[2] / radii[0]
        self.unsquash_z = radii[0] / radii[2]

        self.squash   = Vector3((1., self.squash_y,   self.squash_z))
        self.unsquash = Vector3((1., self.unsquash_y, self.unsquash_z))
        self.unsquash_sq = self.unsquash**2

    def coords_from_vector3(self, position, axes=2):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            obs         a Vector3 of observer positions. In some cases, a
                        surface is defined in part by the position of the
                        observer. In the case of a RingPlane, this argument is
                        ignored and can be omitted.
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

        unsquashed = Vector3.as_standard(position) * self.unsquash

        r = unsquashed.norm()
        (x,y,z) = unsquashed.as_scalars()
        lat = (z/r).arcsin()
        lon = y.arctan2(x)

        if derivs:
            raise NotImplementedError("ellipsoid coordinate derivatives are " +
                                      "not yet supported")

        if axes == 2:
            return (lon, lat)
        else:
            return (lon, lat, r - self.req)

    def vector3_from_coords(self, coords, obs=None, derivs=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars defining the coordinates
                lon     longitude in radians.
                lat     latitude in radians
                elev    a rough measure of distance from the surface, in km;
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

        # Convert to Scalars in standard units
        lon = Scalar.as_standard(coords[0])
        lat = Scalar.as_standard(coords[1])

        if len(coords) == 2:
            r = Scalar(0.)
        else:
            r = Scalar.as_standard(elev) + self.req

        r_coslat = r * lat.cos()
        x = r_coslat * lon.cos()
        y = r_coslat * lon.sin() * self.squash_y
        z = r * lat.sin() * self.squash_z

        pos = Vector3.from_scalars(x,y,z)

        if derivs:
            raise NotImplementedError("ellipsoid position derivatives are " +
                                      "not yet supported")

        return pos

    def intercept(self, obs, los, derivs=False, t_guess=None):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.
            t_guess     initial guess at the t array, optional.

        Return:         a tuple (pos, t) where
            pos         a unitless Vector3 of intercept points on the surface,
                        in km.
            t           a unitless Scalar such that:
                            position = obs + t * los

                        If derivs is True, then pos and t are returned with
                        subfields "d_dobs" and "d_dlos", where the former
                        contains the MatrixN of partial derivatives with respect
                        to obs and the latter is the MatrixN of partial
                        derivatives with respect to los. The MatrixN item shapes
                        are [3,3] for the derivatives of pos, and [1,3] for the
                        derivatives of t.
        """

        # Convert to standard units and un-squash
        obs = Vector3.as_standard(obs)
        los = Vector3.as_standard(los)

        obs_unsquashed = Vector3.as_standard(obs) * self.unsquash
        los_unsquashed = Vector3.as_standard(los) * self.unsquash

        # Solve for the intercept distance, masking lines of sight that miss
        a = los_unsquashed.dot(los_unsquashed)
        b = los_unsquashed.dot(obs_unsquashed) * 2.
        c = obs_unsquashed.dot(obs_unsquashed) - self.req_sq
        d = b**2 - 4. * a * c

        d_sqrt = d.sqrt()
        t = (d_sqrt - b) / (2. * a)
        pos = obs + t*los

        if derivs:
            raise NotImplementedError("ellipsoid intercept derivatives are " +
                                      "not yet supported")

        return (pos, t)

    def normal(self, position):
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

        perp = Vector3.as_standard(pos) * self.unsquash_sq

        if derivs:
            raise NotImplementedError("ellipsoid normal derivatives are " +
                                      "not supported")

        return perp

    def intercept_with_normal(self, normal):
        """Constructs the intercept point on the surface where the normal vector
        is parallel to the given vector.

        Input:
            normal      a Vector3 of normal vectors, with optional units.
            derivs      true to return a matrix of partial derivatives.

        Return:         a unitless Vector3 of surface intercept points, in km.
                        Where no solution exists, the components of the returned
                        vector should be masked.

                        If derivs is True, then the returned intercept points
                        have a subfield "d_dperp", which contains the partial
                        derivatives with respect to components of the normal
                        vector, as a MatrixN object with item shape [3,3].
        """

        # TBD
        pass

    def intercept_normal_to(self, position):
        """Constructs the intercept point on the surface where a normal vector
        passes through a given position.

        Input:
            pos         a Vector3 of positions near the surface, with optional
                        units.

        Return:         a unitless vector3 of surface intercept points. Where no
                        solution exists, the returned vector should be masked.

                        If derivs is True, then the returned intercept points
                        have a subfield "d_dpos", which contains the partial
                        derivatives with respect to components of the given
                        position vector, as a MatrixN object with item shape
                        [3,3].
        """

        # TBD
        pass

    def velocity(self, position):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 of velocities, in units of km/s.
        """

        return Vector3((0,0,0))

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Ellipsoid(unittest.TestCase):
    
    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
