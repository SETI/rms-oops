################################################################################
# oops_/surface/spheroid.py: Spheroid subclass of class Surface
#
# 2/27/12 Checked in (BSW)
################################################################################

import numpy as np

from baseclass import Surface
from oops_.array_.all import *
import oops_.registry as registry

class Ansa(Surface):
    """Ansa defines a surface centered on the given path and
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
    
    def __init__(self, origin, frame):
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
        
    
    def as_coords(self, position, obs, axes=2, derivs=False):
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
        obs_xy = obs.vals[...,0:2]
        obs_xy_modulus = np.sqrt((obs_xy*obs_xy).sum())
        obs_x = obs.vals[...,0]
        obs_y = obs.vals[...,1]
        obs_alpha = np.arctan2(obs_x, -obs_y)
        
        x = position.vals[...,0]
        y = position.vals[...,1]
        rvals = np.sqrt(x**2 + y**2)
        r = Scalar.as_standard(rvals)
        z = Scalar.as_standard(position.vals[...,2])
            
        theta = 0.
        
        # now get theta
        if axes==3:
            xy = np.column_stack((x,y))
            los_xy = xy - obs_xy
            
            los_modulus = np.sqrt((los_xy*los_xy).sum(axis=1))
            cos_alpha1 = (-obs_xy*los_xy).sum(axis=1)/(obs_xy_modulus*los_modulus)
            alpha1 = np.arccos(cos_alpha1)
            theta1 = alpha1 - obs_alpha
            
            theta2 = np.arccos(x/rvals)
            
            theta = theta1 + theta2
        
        if derivs:
            raise NotImplementedError("ansa as_coords() " +
                                      "derivatives are not implemented")

        return Vector3.from_scalars(r,z,theta)
    
    def as_vector3(self, obs, r, z, theta=0., derivs=False):
        """Converts coordinates in the surface's internal coordinate system into
            position vectors at or near the surface.
            
            Input:
            lon             longitude in radians.
            lat             latitude in radians
            elevation       a rough measure of distance from the surface, in km.
            
            Return:             the corresponding Vector3 of (unsquashed) positions,
            in km.
            """
        obs_x = obs.vals[...,0]
        obs_y = obs.vals[...,1]
        obs_x2 = obs_x**2
        obs_y2 = obs_y**2
        
        # first find a point on the surface that has the same los as the point
        # we are looking for
        # we have x**2 + y**2 = r**2 for point on surface
        # also have obsx * x + obs_y * y = r**2
        rvals = r.vals
        r1_vals = rvals * np.cos(theta.vals) #r to surface
        r1_vals2 = r1_vals**2
        
        a = obs_x2 + obs_y2
        """b = -2. * r1_vals2 * obs_x
        c = r1_vals2 * (r1_vals2 - obs_y2)
        discr_sq = b**2 - 4.*a*c
        discr = np.sqrt(b**2 - 4*a*c)
        two_a = a + a
        x_surf1 = (-b + discr)/two_a
        x_surf2 = (-b - discr)/two_a
        x_surf = x_surf1
        x_surf[r1_vals<0.] = x_surf2
        y_surf = (r1_vals2 - obs_x * x_surf) / obs_y"""
        
        r_obs_x = r1_vals * obs_x
        discr = np.sqrt(r_obs_x**2 - a * (r1_vals2 - obs_y2))
        r1_vals_over_a = r1_vals / a
        x_surf1 = r1_vals_over_a * (r_obs_x + discr)
        x_surf2 = r1_vals_over_a * (r_obs_x - discr)
        x_surf = np.where(r1_vals>0.,
                          np.maximum(x_surf1, x_surf2),
                          np.minimum(x_surf1, x_surf2))
        y_surf = (r1_vals2 - obs_x * x_surf) / obs_y
        
        # if theta = 0. (pt on surface), we are done
        if theta == 0.:
            x = x_surf
            y = y_surf
        else:
            # now find theta2 where theta = theta1 + theta2 and
            # theta1 = arccos(x_surf/r)
            theta1 = np.arccos(x_surf/r1_vals)
            theta2 = theta.vals - theta1
            x = rvals * np.cos(theta2)
            y = rvals * np.sin(theta2)
        
        if derivs:
            raise NotImplementedError("ansa as_vector3() " +
                                      "derivatives are not implemented")
        
        return Vector3.from_scalars(x, y, z.vals)

    def intercept(self, obs, los, derivs=False):
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
        # <dx,dy> = <los_xy>
        # vector of r from origin is <-dy,dx> or <dy,-dx>
        # obs_xy + u * <dx,dy> = w * <-dy,dx>
        # u = -(obs_x + obs_y*dy)/(dy**2 + dx)
        obs_x = obs.vals[...,0]
        obs_y = obs.vals[...,1]
        dx = los.vals[...,0]
        dy = los.vals[...,1]
        u = -(obs_x + obs_y*dy) / (dy**2 + dx)
        pos = obs + u * los
        
        if derivs:
            raise NotImplementedError("ansa intercept() " +
                                      "derivatives are not implemented")
        
        return pos
    
    def normal(self, position, obs, derivs=False):
        """Returns the normal vector at a position at or near a surface.
            
            Input:
            position        a Vector3 of positions at or near the surface.
            
            Return:             a Vector3 containing directions normal to the
            surface that pass through the position. Lengths are
            arbitrary.
            """
        # normal is simply the position - obs
        n = position - obs
        
        if derivs:
            raise NotImplementedError("ansa normal() " +
                                      "derivatives are not implemented")
        
        return n
    
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
    
    def intercept_with_normal(self, normal, obs, derivs=False):
        """Constructs the intercept point on the surface where the normal vector
            is parallel to the given vector.
            
            Input:
            normal          a Vector3 of normal vectors.
            
            Return:             a Vector3 of surface intercept points. Where no
            solution exists, the components of the returned
            vector should be np.nan.
            """
        
        obs_x = obs.vals[...,0]
        obs_y = obs.vals[...,1]
        dx = normal.vals[...,0]
        dy = normal.vals[...,1]
        u = (obs_x*dy - obs_y*dx) / (dx**2 + dy)
        pos = obs + u * los
        
        if derivs:
            raise NotImplementedError("ansa intercept_with_normal() " +
                                      "derivatives are not implemented")
        
        return pos
    
    def intercept_normal_to(self, position, obs, derivs=False):
        """Constructs the intercept point on the surface where a normal vector
            passes through a given position.
            
            Input:
            position        a Vector3 of positions near the surface.
            
            Return:             a Vector3 of surface intercept points. Where no
            solution exists, the components of the returned
            vector should be np.nan.
            """
        los = position - obs
        return self.intercept(obs, los, derivs)
    
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
        
        obs_r = np.random.random((1,1))*100000. + 400000.
        obs_theta = np.random.random((1,1))*np.pi/4 + (np.pi * 1.375)
        obs_z = np.random.random((1,1))*10000. + 10000.
        ob_x = obs_r * np.cos(obs_theta)
        ob_y = obs_r * np.sin(obs_theta)
        obs_arr = np.column_stack((ob_x,ob_y,obs_z))
        #obs_arr = np.random.random((10,3))*100000. + 400000.
        #obs_arr = np.random.random((1,3))*100000. + 400000.
        obs = Vector3(obs_arr)
        surf = Ansa("SSB", "J2000")
        
        orig_pos = Vector3(np.random.random((10,3))*100000. + 70000.)
        coord = surf.as_coords(orig_pos, obs, 3)
        r = Scalar(coord.vals[...,0])
        z = Scalar(coord.vals[...,1])
        theta = Scalar(coord.vals[...,2])
        pos = surf.as_vector3(obs,r,z,theta)
        x_o = orig_pos.vals[...,0]
        y_o = orig_pos.vals[...,1]
        r_o = np.sqrt(x_o**2 + y_o**2)
        x_p = orig_pos.vals[...,0]
        y_p = orig_pos.vals[...,1]
        r_p = np.sqrt(x_p**2 + y_p**2)
        self.assertTrue(abs(orig_pos - pos) < 1.e-9)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
