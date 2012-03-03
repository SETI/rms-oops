################################################################################
# oops_/surface/ellipsoid.py: Ellipsoid subclass of class Surface
#
# 2/17/12 Created (MRS)
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

    def as_coords(self, position, axes=2):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            position        a Vector3 of positions at or near the surface.
            axes            2 or 3, indicating whether to return a tuple of two
                            or 3 Scalar objects.

        Return:             coordinate values packaged as a tuple containing
                            two or three Scalars, one for each coordinate. If
                            axes=2, then the tuple is (longitude, latitude); if
                            axes=3, the tuple is (longitude, latitude,
                            elevation).
        """

        unsquashed = Vector3.as_standard(position) * self.unsquash

        r = unsquashed.norm()
        (x,y,z) = unsquashed.as_scalars()
        lat = (z/r).arcsin()
        lon = y.arctan2(x)

        if axes == 2:
            return (lon, lat)
        else:
            return (lon, lat, r - self.req)

    def as_vector3(self, lon, lat, elevation=0.):
        """Converts coordinates in the surface's internal coordinate system into
        position vectors at or near the surface.

        Input:
            lon             longitude in radians.
            lat             latitude in radians
            elevation       a rough measure of distance from the surface, in km.

        Return:             the corresponding Vector3 of (unsquashed) positions,
                            in km.
        """

        # Convert to Scalars in standard units
        lon = Scalar.as_standard(lon)
        lat = Scalar.as_standard(lat)
        r = Scalar.as_standard(elevation) + self.req

        r_coslat = r * lat.cos()
        x = r_coslat * lon.cos()
        y = r_coslat * lon.sin() * self.squash_y
        z = r * lat.sin() * self.squash_z

        return Vector3.from_scalars(x,y,z)

    def intercept(self, obs, los, derivs=False):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.

        Return:         a tuple (position, factor) if derivs is False; a tuple
                        (position, factor, dpos_dobs, dpos_dlos) if derivs is
                        True.
            position    a unitless Vector3 of intercept points on the surface,
                        in km.
            factor      a unitless Scalar of factors such that:
                            position = obs + factor * los
            dpos_dobs   the partial derivatives of the position vector with
                        respect to the observer position, as a Matrix3.
            dpos_dlos   the partial derivatives of the position vector with
                        respect to the line of sight, as a Matrix3.
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

        t = (d.sqrt() - b) / (2. * a)

        return (obs + t*los, t)

    def normal(self, position):
        """Returns the normal vector at a position at or near a surface.

        Input:
            position        a Vector3 of positions at or near the surface.

        Return:             a Vector3 containing directions normal to the
                            surface that pass through the position. Lengths are
                            arbitrary.
        """

        return Vector3.as_standard(position) * self.unsquash_sq

    def gradient(self, position, axis=0):
        """Returns the gradient vector at a specified position at or near the
        surface. The gradient of surface coordinate c is defined as a vector
            (dc/dx,dc/dy,dc/dz)
        It has the property that it points in the direction of the most rapid
        change in value of the coordinate, and its magnitude is the rate of
        change in that direction.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.
            axis        0, 1 or 2, identifying the coordinate axis for which the
                        gradient is sought.
        """

        # TBD
        pass
    
    def velocity(self, position):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            position        a Vector3 of positions at or near the surface.
        """

        # An internal wind field is not implemented
        return Vector3((0,0,0))
    
    def intercept_with_normal(self, normal):
        """Constructs the intercept point on the surface where the normal vector
            is parallel to the given vector.
            
            Input:
            normal          a Vector3 of normal vectors.
            
            Return:             a Vector3 of surface intercept points. Where no
            solution exists, the components of the returned
            vector should be np.nan.
            """

        # TBD
        pass

    def intercept_normal_to(self, position):
        """Constructs the intercept point on the surface where a normal vector
        passes through a given position.

        Input:
            position        a Vector3 of positions near the surface.

        Return:             a Vector3 of surface intercept points. Where no
                            solution exists, the components of the returned
                            vector should be np.nan.
        """

        # TBD
        pass

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
