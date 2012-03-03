################################################################################
# oops_/surface/spheroid.py: Spheroid subclass of class Surface
#
# 2/15/12 Checked in (BSW)
# 2/17/12 Modified (MRS) - Inserted coordinate definitions; added use of trig
#   functions and sqrt() defined in Scalar class to enable cleaner algorithms.
#   Unit tests added.
################################################################################

import numpy as np

from surface_ import Surface
from oops_.array.all import *
import oops_.registry as registry

class Spheroid(Surface):
    """Spheroid defines a spheroidal surface centered on the given path and
    fixed with respect to the given frame. The short radius of the spheroid is
    oriented along the Z-axis of the frame.

    The coordinates defining the surface grid are (longitude, latitude), based
    on the assumption that a spherical body has been "squashed" along the
    Z-axis. The latitude defined in this manner is neither planetocentric nor
    planetographic; functions are provided to perform the conversion to either
    choice. Longitudes are measured in a right-handed manner, increasing toward
    the east. Values range from -pi to pi.

    Elevations are defined by "unsquashing" the radial vectors and then
    subtracting off the equatorial radius of the body. Thus, the surface is
    defined as the locus of points where elevation equals zero. However, the
    direction of increasing elevation is not exactly normal to the surface.
    """
    
    def __init__(self, origin, frame, radii):
        """Constructor for a Spheroid object.
            
        Input:
            origin      a Path object or ID defining the motion of the center
                        of the ring plane.

            frame       a Frame object or ID in which the ring plane is the
                        (x,y) plane (where z = 0).

            radii       a tuple (equatorial_radius, polar_radius) in km.
        """

        self.origin_id = registry.as_path_id(origin)
        self.frame_id  = registry.as_frame_id(frame)

        self.radii  = np.array((radii[0], radii[0], radii[1]))
        self.req    = radii[0]
        self.req_sq = self.req**2

        self.squash_z   = radii[1] / radii[0]
        self.unsquash_z = radii[0] / radii[1]

        self.squash   = Vector3((1., 1., self.squash_z))
        self.unsquash = Vector3((1., 1., self.unsquash_z))
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
        y = r_coslat * lon.sin()
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

        Return:         a unitless Vector3 of the gradients sought. Values are
                        always in standard units.
        """

        pass

        #gradient for a spheroid is simply <2x/a, 2y/a, 2z/c>

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

        # For a sphere, this is just the point at r * n, where r is the radius
        # of the sphere.  For a spheroid, this is just the same point scaled up
#         np_scale_array = np.array( [[self.r0, 0, 0], [0, self.r0, 0],
#                                     [0, 0, self.r2]])
#         expand_matrix = Matrix3(np_scale_array)
#         pos = expand_matrix * normal

        return pos

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

    def lat_to_centric(self, lat):
        """Converts a latitude value given in internal spheroid coordinates to
        its planetocentric equivalent.
        """

        return (lat.tan() * self.squash_z).arctan()

    def lat_to_graphic(self, lat):
        """Converts a latitude value given in internal spheroid coordinates to
        its planetographic equivalent.
        """

        return (lat.tan() * self.unsquash_z).arctan()

    def lat_from_centric(self, lat):
        """Converts a latitude value given in planetocentric coordinates to its
        equivalent value in internal spheroid coordinates.
        """

        return (lat.tan() * self.unsquash_z).arctan()

    def lat_from_graphic(self, lat):
        """Converts a latitude value given in planetographic coordinates to its
        equivalent value in internal spheroid coordinates.
        """

        return (lat.tan() * self.squash_z).arctan()

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Spheroid(unittest.TestCase):

    def runTest(self):

        REQ  = 60268.
        RPOL = 54364.
        planet = Spheroid("SSB", "J2000", (REQ, RPOL))

        # Coordinate/vector conversions
        NPTS = 10000
        obs = (2 * np.random.rand(NPTS,3) - 1.) * REQ

        (lon,lat,elev) = planet.as_coords(obs,axes=3)
        test = planet.as_vector3(lon,lat,elev)
        self.assertTrue(abs(test - obs) < 3.e-9)

        # Spheroid intercepts & normals
        obs[...,0] = np.abs(obs[...,0])
        obs[...,0] += REQ

        los = (2 * np.random.rand(NPTS,3) - 1.)
        los[...,0] = -np.abs(los[...,0])

        (pts, t) = planet.intercept(obs, los)
        test = t * Vector3(los) + Vector3(obs)
        self.assertTrue(abs(test - pts) < 1.e-9)

        self.assertTrue(np.all(t.mask == pts.mask))
        self.assertTrue(np.all(pts.mask[t.vals < 0.]))

        normals = planet.normal(pts)

        pts.vals[...,2] *= REQ/RPOL
        self.assertTrue(abs(pts.norm()[~pts.mask] - REQ) < 1.e-8)

        normals.vals[...,2] *= RPOL/REQ
        self.assertTrue(abs(normals.unit() - pts.unit()) < 1.e-14)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
