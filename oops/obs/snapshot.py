################################################################################
# oops/obs/snapshot.py: Subclass Snapshot of class Observation
################################################################################

import numpy as np
import numpy.ma as ma
import pylab
import unittest

import oops
import julian
import solar
import sys
import math

from oops.obs.baseclass import Observation
from oops.xarray.all import *
from oops.event import Event

import oops.frame.all 
import oops.path.all 
import oops.surface.all 

dist_tolerance = 1000.

class Snapshot(Observation):
    """A Snapshot is an Observation consisting of a 2-D image made up of pixels
    all exposed at the same time."""

    def __init__(self, data, mask, axes, time, fov, path_id, frame_id,
                       calibration):

        self.data = data
        self.mask = mask
        self.path_id = path_id
        self.frame_id = frame_id
        self.t0 = time[0]
        self.t1 = time[1]
        self.fov = fov
        self.calibration = calibration

        self.u_axis = axes.index("u")
        self.v_axis = axes.index("v")
        self.t_axis = None
    
    def ring_shadow_back_plane(self, min_range=0., max_range=sys.float_info.max):
        """gets the ring's backplane and returns whether in shadow or not.
            
            Input:
            min_range       minimum range for rings from planet's center
            max_range       maximum range for rings from planet's center
            
            Return:         2D array of 0 for not in rings, 1 for rings in
                            shadow, and 2 for rings not in shadow
            """

        ignore = oops.frame.RingFrame("IAU_SATURN")
        surface = oops.surface.RingPlane("SATURN", "IAU_SATURN_DESPUN")
        planet_surface = oops.surface.Spheroid("SATURN", "IAU_SATURN")

        #(surface_event, rel_surf_evt) = self.back_plane_setup(surface)
        (surface_event, rel_surf_evt,
         dist_from_instrument, obs_mask) = self.back_plane_setup(surface,
                                                                 planet_surface)
        
        #determine what is visible
        """dist_from_instrument = np.sqrt(rel_surf_evt.pos.vals[...,0]**2 +
                                       rel_surf_evt.pos.vals[...,1]**2 +
                                       rel_surf_evt.pos.vals[...,2]**2)
        rel_blk_evt = planet_surface.photon_to_event(image_event, 1)[1]
        block_dist = np.sqrt(rel_blk_evt.pos.vals[...,0]**2 +
                             rel_blk_evt.pos.vals[...,1]**2 +
                             rel_blk_evt.pos.vals[...,2]**2)
        obs_mask = block_dist < (dist_from_instrument - dist_tolerance)"""


        dist_from_center = np.sqrt(surface_event.pos.vals[...,0]**2 +
                                   surface_event.pos.vals[...,1]**2)
        
        dist_from_center[np.isnan(dist_from_center)] = 0.
        dist_from_center[dist_from_center < min_range] = 0.
        dist_from_center[dist_from_center > max_range] = 0.
        
        #now, for each point on surface that is in rings, check if in shadow
        shadow_mask = self.get_shadow_mask("SUN", "SATURN", planet_surface,
                                           surface_event)

        ring_mask = dist_from_center == 0.
        bp_masked = ma.array(dist_from_center,
                             mask=(obs_mask | ring_mask | shadow_mask))
        
        return bp_masked
    
    def get_shadow_mask(self, light_path_id, target_path_id, target_surface,
                        obj_evt):
        """Check to see if target blocks object as seen from light source.
            
            Input:
            light_path_id   path id for the light source
            target_path_id  path id for the target of the light to see if it is
                            between the light source and checked object along
                            los rays.
            target_surface  the target's surface, such as a Spheroid or
                            RingPlane.
            obj_evt         the image event for the object that we are checking
                            whether is in shadow or not.
            
            Return:         mask returning true for those pixels on object in
                            shadow of target for this light source.
            """
        
        #get the observer path wrt the target we are checking is blocking los
        light_wrt_target = oops.path.connect(light_path_id, target_path_id)
        
        # get the relative event of the photon leaving the light source
        light_dep_event = light_wrt_target.photon_to_event(obj_evt, 1)[1]
        #light from target is light from obj plus obj from target
        origin = obj_evt.pos + light_dep_event.pos
        dir = -light_dep_event.pos   #ray to check from observer to pixel

        #intercept = target_surface.intercept(obs_dep_event.pos, dir)[0]
        intercept = target_surface.intercept(origin, dir)[0]
        rel_pos = light_dep_event.pos + intercept
        pos = rel_pos.vals
        dist = np.sqrt(pos[...,0]**2 + pos[...,1]**2 + pos[...,2]**2)
        opos = light_dep_event.pos.vals
        obj_dist = np.sqrt(opos[...,0]**2 + opos[...,1]**2 + opos[...,2]**2)

        farther_mask = dist < (obj_dist + dist_tolerance)
        return farther_mask


    def radius_back_plane(self, min_range=0., max_range=sys.float_info.max):
        """the backplane for radius values for the given snapshot. currently
            just for Cassini to rings... needs to be updated for all spacecraft
            and all targets.
            
            Return:     2D array of radius values
            """
        ignore = oops.frame.RingFrame("IAU_SATURN")
        surface = oops.surface.RingPlane("SATURN", "IAU_SATURN_DESPUN")

        (surface_event, rel_surf_evt) = self.back_plane_setup(surface)

        bp_data = np.sqrt(surface_event.pos.vals[...,0]**2 +
                          surface_event.pos.vals[...,1]**2)
        bp_data[np.isnan(bp_data)] = 0.
        bp_data[bp_data < min_range] = 0.
        bp_data[bp_data > max_range] = 0.
        mask = bp_data == 0.
        bp_masked = ma.array(bp_data, mask=mask)

        return bp_masked

    def polar_back_plane(self, path_id, frame_id, min_range=0.,
                         max_range=sys.float_info.max):
        """the backplane for polar theta value for the given snapshot. currently
            just for Cassini to rings... needs to be updated for all spacecraft
            and all targets.
        
            Return:     2D array of radius values
            """
        ignore = oops.frame.RingFrame(frame_id)
        surface = oops.surface.Spheroid(path_id, frame_id)
    
        (surface_event, rel_surf_evt) = self.back_plane_setup(surface)
        
        x = surface_event.pos.vals[...,0]
        y = surface_event.pos.vals[...,1]
        
        theta = np.arctan2(y,x) % (2.*np.pi)
        mask = x == np.nan
        theta_mask = ma.array(theta, mask=mask)
        
        return theta_mask

    def latitude_c_back_plane(self, path_id, frame_id, min_range=0.,
                              max_range=sys.float_info.max):
        """the backplane for polar theta value for the given snapshot. currently
            just for Cassini to rings... needs to be updated for all spacecraft
            and all targets.
        
            Return:     2D array of radius values
            """
        ignore = oops.frame.RingFrame(frame_id)
        surface = oops.surface.Spheroid(path_id, frame_id)
    
        (surface_event, rel_surf_evt) = self.back_plane_setup(surface)
    
        x = surface_event.pos.vals[...,0]
        y = surface_event.pos.vals[...,1]
        z = surface_event.pos.vals[...,2]
    
        theta = ( np.arctan2(z,np.sqrt(x*x+y*y)) + 0.5 * np.pi) % (np.pi)
        mask = x == np.nan
        theta_mask = ma.array(theta, mask=mask)
        
        return theta_mask

    def back_plane_setup(self, surface, blocking_surface=None):
        """Currently this is set up purely for ring planes.  Must be generalized
            in the future.
            
            Return:     surface_event - the event where the photons left the
                        surface.
            """
        #first get the tdb time when instrument took measurement
        #get the average between the start and stop time to get the mid time
        tdb = (self.t0 + self.t1) * 0.5
        
        #construct a buffer from which we will create rays through these pixels
        uv_shape = self.fov.uv_shape.vals
        buffer = np.empty((uv_shape[0], uv_shape[1], 2))
        buffer[:,:,1] = np.arange(uv_shape[1]).reshape(uv_shape[1],1)
        buffer[:,:,0] = np.arange(uv_shape[0])
        indices = Pair(buffer + 0.5)
        
        rays = self.fov.los_from_uv(indices)

        #reverse so that they are arrival rays
        arrivals = -rays
        #arrivals = -rays[568][956]
        image_event = Event(tdb, (0,0,0), (0,0,0), "CASSINI",
                                 self.frame_id, Empty(), arrivals)
        
        (surface_event, rel_surf_evt) = surface.photon_to_event(image_event, 1)
        
        dist_from_instrument = np.sqrt(rel_surf_evt.pos.vals[...,0]**2 +
                                       rel_surf_evt.pos.vals[...,1]**2 +
                                       rel_surf_evt.pos.vals[...,2]**2)
        
        if blocking_surface != None:
            rel_blk_evt = blocking_surface.photon_to_event(image_event, 1)[1]
            block_dist = np.sqrt(rel_blk_evt.pos.vals[...,0]**2 +
                                 rel_blk_evt.pos.vals[...,1]**2 +
                                 rel_blk_evt.pos.vals[...,2]**2)
            mask = block_dist < (dist_from_instrument - dist_tolerance)
        else:
            mask = dist_from_instrument == np.nan

        return (surface_event, rel_surf_evt, dist_from_instrument, mask)
        #return (surface_event, rel_surf_evt)



########################################
# UNIT TESTS
########################################

class Test_Snapshot(unittest.TestCase):

    def runTest(self):

        pass

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
