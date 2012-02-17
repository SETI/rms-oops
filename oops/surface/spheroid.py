################################################################################
# oops/surface/spheroid.py: Spheroid subclass of class Surface
#
# 2/15/12 Checked in (BSW)
# 2/16/12 Modified (MRS) - Inserted coordinate definitions; added use of trig
#   functions and sqrt() defined in Scalar class to enable cleaner algorithms.
################################################################################

import numpy as np

from baseclass import Surface
from oops.xarray.all import *
import oops.frame.registry as frame_registry
import oops.path.registry as path_registry

class Spheroid(Surface):
    """Spheroid defines a spheroidal surface centered on the given path and
    fixed with respect to the given frame. The short radius of the spheroid is
    oriented along the Z-axis of the frame."""
    
    def __init__(self, origin, frame, radii, centric=True):
        """Constructor for a Spheroid object.
            
        Input:
            origin      a Path object or ID defining the motion of the center
                        of the ring plane.

            frame       a Frame object or ID in which the ring plane is the
                        (x,y) plane (where z = 0).

            radii       a tuple (equatorial_radius, polar_radius) in km.

            centric     True to use planetocentric latitudes; False to use
                        planetographic latitudes.
        """

        self.origin_id = path_registry.as_id(origin)
        self.frame_id  = frame_registry.as_id(frame)

        self.equatorial = radii[0]
        self.polar      = radii[1]
        self.squash     = self.polar / self.equatorial
        self.unsquash   = self.equatorial / self.polar

        self.centric = centric
        if self.centric:
            self.tan_factor = self.squash
        else:
            self.tan_factor = self.unsquash

        self.v_squash   = Vector3((1., 1., self.squash))
        self.v_unsquash = Vector3((1., 1., self.unsquash))
        self.v_unsquash_sq = self.v_unsquash**2

    def to_squashed_coords(self, position):
        """Converts from position vectors in the internal frame into "squashed"
        surface coordinates. This behaves like a spherical coordinate system,
        except all Z coordinates have been scaled by a constant factor to
        accommodate the flattening. Radii are measured from the center of the
        frame.

        Input:
            position        a Vector3 of positions at or near the surface.
            axes            2 or 3, indicating whether to return a tuple of two
                            or 3 Scalar objects.

        Return:             coordinate values packaged as a tuple containing
                            (lon, lat, r) squashed coordinates as Scalars.
        """

        unsquashed = Vector3.as_standard(position) * self.v_unsquash

        r = unflat.norm()
        (x,y,z) = unflat.as_scalars()
        lat = (z/r).arcsin()
        lon = y.arctan2(x)

    def from_squashed_coords(self, lon, lat, r):
        """Converts coordinates in the surface's internal "squashed" coordinate
        system into position vectors at or near the surface. Squashed
        coordinates behave like normal spherical coordinates except that all
        Z-coordinates have been scaled by a constant factor to accommodate the
        flattening.

        Input:
            lon             longitude in radians.
            lat             squashed latitude in radians
            r               squashed radial distance from the origin in km.

        Return:             the corresponding Vector3 of (unsquashed) positions.
        """

        lon = Scalar.as_scalar(lon)
        lat = Scalar.as_scalar(lat)
        r   = Scalar.as_scalar(r)

        r_coslat = r * lat.cos()
        x = r_coslat * lon.cos()
        y = r_coslat * lon.sin()
        z = r * lat.sin() * self.flat[2]

        return Vector3.from_scalars((x,y,z))

    def as_coords(self, position, axes=2):
        """Converts from position vectors in the internal frame into the surface
        coordinate system. We use spherical coordinates (longitude, latitude,
        elevation), where latitude is either planetocentric or planetographic,
        as specified in the constructor. Elevation values are NOT distances
        normal to the surface, because these are difficult to derive. Instead,
            elevation == (r_unsquashed - equatorial_radius)
        With this definition, the spheroid's surface is defined by the locus of
        points where elevation = 0.

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

        (lon, lat, r) = self.to_squashed_coords(position)

        # Convert the latitude to planetocentric or planetographic
        lat = (lat.tan() * self.tan_factor).arctan()

        if axes == 2:
            return (lon, lat)
        else:
            return (lon, lat, r - self.equatorial)

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
        elevation = Scalar.as_standard(elevation)
        
        # Convert to squashed coordinates and return unsquashed position
        return self.from_squashed_coords(lon,
                                         (lat.tan()/self.tan_factor).arctan(),
                                         elevation + self.equatorial)

    def intercept(self, obs, los):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs             observer position as a Vector3.
            los             line of sight as a Vector3.

        Return:             a tuple (position, factor)
            position        a Vector3 of intercept points on the surface.
            factor          a Scalar of factors such that
                                position = obs + factor * los
        """

        # Convert to standard units and un-squash
        obs = Vector3.as_standard(obs)
        los = Vector3.as_standard(los)

        obs_unsquashed = Vector3.as_standard(obs) * self.v_unsquash
        los_unsquashed = Vector3.as_standard(los) * self.v_unsquash

        # Solve for the intercept distance, masking lines of sight that miss
        a = np.sum(los_unsquashed**2, axis=-1)
        b = 2. * np.sum(los_unsquashed * obs_unsquashed, axis=-1)
        c = np.sum(obs_unsquashed**2, axis=-1) - 1.
        d = b**2 - 4. * a * c

        t = (d.sqrt() - b) / (2. * a)

        return (obs + t*los, t)

#     def intercept_using_scale(self, obs, los):
#         """Returns the position where a specified line of sight intercepts the
#             surface.
#         
#             Input:
#             obs             observer position as a Vector3.
#             los             line of sight as a Vector3.
#         
#             Return:             a tuple (position, factor)
#             position        a Vector3 of intercept points on the surface.
#             factor          a Scalar of factors such that
#             position = obs + factor * los
#             """
#     
#         # Solve for obs + t * los for scalar t, such that the z-component
#         # equals zero.
#         obs = Vector3.as_vector3(obs).vals
#         los = Vector3.as_vector3(los).vals
#         
# 
#         #flattening = self.r2 / self.r0
#         one_over_r = 1. / self.radii
#         np_scale_array = np.array( [[one_over_r[0], 0, 0], [0, one_over_r[1], 0],
#                                     [0, 0, one_over_r[2]]])
#         flattening_matrix = Matrix3(np_scale_array)
#         fl_obs = (flattening_matrix * obs).vals
#         fl_los = (flattening_matrix * los).vals
# 
#         a = np.sum(fl_los**2, axis=-1)
#         b = 2. * np.sum(fl_los*fl_obs, axis=-1)
#         c = np.sum(fl_obs**2, axis=-1) - 1.
#         d = b * b - 4. * a * c
#         
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             d_sqrt = np.sqrt(d)
#         
#         mask = d < 0.
#         #d_sqrt[mask] = np.nan
#             #if mask:
#         #d_sqrt = b
#     
#         t = ((d_sqrt - b) / (2. * a))
#         #mean_t = t.mean()
#         #for the moment, just set it to the first t
#         #mean_t = t[0][0]
#             #        if np.shape(t) == ():
#             #print "setting t == nan to mean_t as array"
#         t[np.isnan(t)] = 0.
#         #np.set_printoptions(threshold=np.nan)
#             #else:
#             #print "setting t == 0 as float value"
#             #t = 0.  # fix with something else
# 
#         array = obs + t[..., np.newaxis] * los
#         #np.nan_to_num(array)
#     
#         return (Vector3(array, mask), Scalar(t, mask))
    
    def normal(self, position):
        """Returns the normal vector at a position at or near a surface.

        Input:
            position        a Vector3 of positions at or near the surface.

        Return:             a Vector3 containing directions normal to the
                            surface that pass through the position. Lengths are
                            arbitrary.
        """

        return Vector3.as_standard(position) * self.v_unsquash_sq

        # n = <df/dx, df,dy, df,dz>
#         nn = position.vals / self.radii_sq
#         n = Vector3(nn).unit()
#         return n    # This does not inherit the given vector's
#                     # shape, but should broadcast properly

    def gradient_at_position(self, position, axis=0, projected=True):
        """Returns the gradient vector at a specified position at or near the
        surface. The gradient is defined as the vector pointing in the direction
        of most rapid change in the value of a particular surface coordinate.

        The magnitude of the gradient vector is the rate of change of the
        coordinate value when starting from this point and moving in this
        direction.

        Input:
            position        a Vector3 of positions at or near the surface.

            axis            0, 1 or 2, identifying the coordinate axis for which
                            the gradient is sought.

            projected       True to project the gradient into the surface if
                            necessary.
        """

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
        np_scale_array = np.array( [[self.r0, 0, 0], [0, self.r0, 0],
                                    [0, 0, self.r2]])
        expand_matrix = Matrix3(np_scale_array)
        pos = expand_matrix * normal

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

########################################
# UNIT TESTS
########################################

import unittest
import oops.frame.all as frame_
import oops.tools as tools

class Test_Spheroid(unittest.TestCase):
    
    def runTest(self):
        
        planet = Spheroid("SSB", "J2000", 60268., 54364.)
        
        # Coordinate/vector conversions
        obs = np.random.rand(2,4,3,3)
        
        (r,theta,z) = planet.as_coords(obs,axes=3)
        self.assertTrue(theta >= 0.)
        self.assertTrue(theta < 2.*np.pi)
        self.assertTrue(r >= 0.)
        
        test = planet.as_vector3(r,theta,z)
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-15))
        
        # Spheroid intercepts
        """los = np.random.rand(2,4,3,3)
        obs[...,2] =  np.abs(obs[...,2])
        los[...,2] = -np.abs(los[...,2])
        
        (pts, factors) = planet.intercept(obs, los)
        self.assertTrue(pts.as_scalar(2) == 0.)
        
        angles = pts - obs
        self.assertTrue(angles.sep(los) > -1.e-12)
        self.assertTrue(angles.sep(los) <  1.e-12)
        
        # Intercepts that point away from the spheroid
        self.assertTrue(np.all(factors.vals > 0.))"""

        # test normals
        obsn = np.array([[100000., 0., 40000.], [100000., 0., 0.]])
        losn = np.array([[-1., 0., 0.], [-1., 0., 0.]])
        obs = Vector3(obsn)
        los = Vector3(losn)
        saturn = Spheroid("SATURN", "IAU_SATURN", 60268., 54364.)
        (pts, pts_mask) = saturn.intercept(obs, los)
        print "pts:"
        print pts
        normals = saturn.normal(pts)
        print "normals:"
        print normals

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
