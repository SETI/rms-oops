import numpy as np
import warnings
import gravity

from baseclass import Surface
from oops.xarray.all import *
import oops.frame.registry as frame_registry
import oops.path.registry as path_registry

############################################
# Spheroid
############################################

class Spheroid(Surface):
    """A spheroid surface in the (x,y) plane, in which the points are optionally
        undergoing circular Keplerian motion about the center point."""
    
    def __init__(self, origin, frame, r0=60269., r2=54364.):
        
        """Constructor for a Spheroid object.
            
            Input:
            origin      a Path object or ID defining the motion of the center
            of the ring plane.
            
            frame       a Frame object or ID in which the ring plane is the
            (x,y) plane (where z = 0).
            
            gravity     an optional Gravity object, used to define the orbital
            velocities within the plane.
            """
        
        self.origin_id = path_registry.as_id(origin)
        self.frame_id  = frame_registry.as_id(frame)
        self.radii = np.array((r0, r0, r2))
        self.radii_sq = self.radii**2
    
    def as_coords(self, position, axes=2):
        """Converts from position vectors in the internal frame into the surface
            coordinate system.
            
            Input:
            position        a Vector3 of positions at or near the surface.
            axes            2 or 3, indicating whether to return a tuple of two
            or 3 Scalar objects.
            
            Return:             coordinate values packaged as a tuple containing
            two or three Scalars, one for each coordinate.
            """
        
        position = Vector3.as_standard(position)
        mask = position.mask
        x = position.vals[...,0]
        y = position.vals[...,1]
        
        r     = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y,x) % (2.*np.pi)
        
        if axes > 2:
            z = position.vals[...,2]
            return (Scalar(r, mask), Scalar(theta, mask), Scalar(z, mask))
        else:
            return (Scalar(r, mask), Scalar(theta, mask))
    
    def as_vector3(self, r, theta, z=Scalar(0.)):
        """Converts coordinates in the surface's internal coordinate system into
            position vectors at or near the surface.
            
            Input:
            r               Scalar of radius values.
            theta           Scalar of longitude values.
            z               Optional Scalar of elevation values.
            
            Return:             the corresponding Vector3 of positions.
            """
        
        # Convert to Scalars
        r     = Scalar.as_standard(r)
        theta = Scalar.as_standard(theta)
        z     = Scalar.as_standard(z)
        
        # Convert to vectors
        shape = Array.broadcast_shape((r, theta, z), [3])
        array = np.empty(shape)
        array[...,0] = np.cos(theta.vals) * r.vals
        array[...,1] = np.sin(theta.vals) * r.vals
        array[...,2] = z.vals
        
        return Vector3(array, r.mask | theta.mask | z.mask)
    
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
        
        # Solve for obs + t * los for scalar t, such that the z-component
        # equals zero.
        return self.intercept_using_scale(obs, los)

    def intercept_no_scale(self, obs, los):
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
    
        # Solve for obs + t * los for scalar t, such that the z-component
        # equals zero.
        obs = Vector3.as_standard(obs).vals
        los = Vector3.as_standard(los).vals
        
        a = np.sum(fl_los**2/self.radii_sq, axis=-1)
        b = np.sum(fl_obs*fl_los/self.radii_sq, axis=-1)
        c = np.sum(fl_obs**2/self.radii_sq, axis=-1)
        #a = ((los[...,0] * los[...,0]) + (los[...,1] * los[...,1])) / r0_sq + (los[...,2] * los[...,2]) / r2_sq
        #b = 2. * ((obs[...,0] * los[...,0] + obs[...,1] * los[...,1]) / r0_sq + (obs[...,2] * los[...,2]) / r2_sq)
        #c = ((obs[...,0] * obs[...,0]) + (obs[...,1] * obs[...,1])) / r0_sq + (obs[...,2] * obs[...,2]) / r2_sq - 1.
        d = b * b - 4. * a * c
        
        mask = d < 0.
        d_sqrt = np.sqrt(d)
        #d_sqrt[mask] = np.nan
        
        t = ((d_sqrt - b) / (2. * a))
        array = obs + t[..., np.newaxis] * los

        mask = obs.mask | los.mask | (los.vals[...,2] == 0) | (t < 0.)
        
        return (Vector3(array, mask), Scalar(t, mask))

    def intercept_using_scale(self, obs, los):
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
    
        # Solve for obs + t * los for scalar t, such that the z-component
        # equals zero.
        obs = Vector3.as_vector3(obs).vals
        los = Vector3.as_vector3(los).vals
        

        #flattening = self.r2 / self.r0
        one_over_r = 1. / self.radii
        np_scale_array = np.array( [[one_over_r[0], 0, 0], [0, one_over_r[1], 0],
                                    [0, 0, one_over_r[2]]])
        flattening_matrix = Matrix3(np_scale_array)
        fl_obs = (flattening_matrix * obs).vals
        fl_los = (flattening_matrix * los).vals

        a = np.sum(fl_los**2, axis=-1)
        b = 2. * np.sum(fl_los*fl_obs, axis=-1)
        c = np.sum(fl_obs**2, axis=-1) - 1.
        d = b * b - 4. * a * c
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d_sqrt = np.sqrt(d)
        
        mask = d < 0.
        #d_sqrt[mask] = np.nan
            #if mask:
        #d_sqrt = b
    
        t = ((d_sqrt - b) / (2. * a))
        #mean_t = t.mean()
        #for the moment, just set it to the first t
        #mean_t = t[0][0]
            #        if np.shape(t) == ():
            #print "setting t == nan to mean_t as array"
        t[np.isnan(t)] = 0.
        #np.set_printoptions(threshold=np.nan)
            #else:
            #print "setting t == 0 as float value"
            #t = 0.  # fix with something else

        array = obs + t[..., np.newaxis] * los
        #np.nan_to_num(array)
    
        return (Vector3(array, mask), Scalar(t, mask))
    
    def normal(self, position):
        """Returns the normal vector at a position at or near a surface.
            
            Input:
            position        a Vector3 of positions at or near the surface.
            
            Return:             a Vector3 containing directions normal to the
            surface that pass through the position. Lengths are
            arbitrary.
            """
        # n = <df/dx, df,dy, df,dz>
        nn = position.vals / self.radii_sq
        n = Vector3(nn).unit()
        return n    # This does not inherit the given vector's
                    # shape, but should broadcast properly

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
            necessary. This has no effect on a Spheroid.
            """
        #gradient for a spheroid is simply <2x/a, 2y/a, 2z/c>
        if axis == 3: return Vector3([0,0,1])
        
        radii = position.copy()
        radii.vals[...,2] = 0.
        
        if axis == 0:
            return radii.unit()
        
        if axis == 1:
            vectors = Vector3(np.zeros(position.vals.shape))
            vectors.vals[...,0] =  position.vals[...,1]
            vectors.vals[...,1] = -position.vals[...,0]
            
            return vectors / radii.norm()**2
        
        raise ValueError("illegal axis value")
    
    def velocity(self, position):
        """Returns the local velocity vector at a point within the surface.
            This can be used to describe the orbital motion of ring particles or
            local wind speeds on a planet.
            
            Input:
            position        a Vector3 of positions at or near the surface.
            """
        
        
        # TBD: We need a vector form of the gravity library!
        
        position = Vector3.as_vector3(position)
        radius = Vector3(position).norm()
        # n = gravity.n(radius.vals)        # !!! Not yet implemented
        n = 0.
        
        return Vector3(position.cross((0,0,-1)) * n)
    
    def intercept_with_normal(self, normal):
        """Constructs the intercept point on the surface where the normal vector
            is parallel to the given vector.
            
            Input:
            normal          a Vector3 of normal vectors.
            
            Return:             a Vector3 of surface intercept points. Where no
            solution exists, the components of the returned
            vector should be np.nan.
            """
        
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
        
        intercept = Vector3.as_vector3(position).copy()
        intercept.vals[...,2] = 0.
        
        return intercept

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
