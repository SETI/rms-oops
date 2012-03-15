################################################################################
# oops_/obs/snapshot.py: Subclass Snapshot of class Observation
################################################################################

import numpy as np
import numpy.ma as ma
import pylab
import unittest

import julian
import solar
import sys
import math

from oops_.obs.observation_ import Observation
from oops_.array.all import *
from oops_.event import Event

import oops_.frame.all as frame_
import oops_.path.all  as path_
import oops_.surface.all as surface_

dist_tolerance = 1000.

class Snapshot(Observation):
    """A Snapshot is an Observation consisting of a 2-D image made up of pixels
    all exposed at the same time."""

    def __init__(self, time, fov, path_id, frame_id, **subfields):
        """Constructor for a Snapshot.

        Input:
            time        a tuple or Pair defining the start time and end time of
                        the observation overall, in seconds TDB.
            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y).
            path_id     the registered ID of a path co-located with the
                        instrument.
            frame_id    the registered ID of a coordinate frame fixed to the
                        optics of the instrument. This frame should have its
                        Z-axis pointing outward near the center of the line of
                        sight, with the X-axis pointing rightward and the y-axis
                        pointing downward.
            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
        """

        self.time = time
        self.fov = fov
        self.path_id = path_id
        self.frame_id = frame_id

        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        # Attributes specific to a Snapshot
        self.midtime = (self.time[0] + self.time[1]) / 2.
        self.texp = self.time[1] - self.time[0]

    def ring_shadow_back_plane(self, min_range=0., max_range=sys.float_info.max):
        """gets the ring's backplane and returns whether in shadow or not.
            
            Input:
            min_range       minimum range for rings from planet's center
            max_range       maximum range for rings from planet's center
            
            Return:         2D array of 0 for not in rings, 1 for rings in
                            shadow, and 2 for rings not in shadow
            """

        ignore = frame_.RingFrame("IAU_SATURN")
        surface = surface_.RingPlane("SATURN", "IAU_SATURN_DESPUN")
        planet_surface = surface_.Spheroid("SATURN", "IAU_SATURN", 60268.,
                                           54364.)

        #(surface_event, rel_surf_evt) = self.back_plane_setup(surface)
        (surface_event, rel_surf_evt,
         dist_from_instrument, obs_mask,
         blk_evt) = self.back_plane_setup(surface, planet_surface)
        
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
        (shadow_mask, light_evt) = self.get_shadow_mask("SUN", "SATURN",
                                                        planet_surface,
                                                        surface_event)
        

        ring_mask = dist_from_center == 0.
        bp_masked = ma.array(dist_from_center,
                             mask=(obs_mask | ring_mask | shadow_mask))
            #bp_masked = ma.array(dist_from_center,
        #                mask=(obs_mask | ring_mask))
        
        return bp_masked

    def incidence_angle_back_plane(self, min_range=0., max_range=sys.float_info.max):
        """gets the ring's backplane and returns whether in shadow or not.
        
            Input:
            min_range       minimum range for rings from planet's center
            max_range       maximum range for rings from planet's center
        
            Return:         2D array of 0 for not in rings, 1 for rings in
            shadow, and 2 for rings not in shadow
            """
    
        ignore = frame_.RingFrame("IAU_SATURN")
        surface = surface_.RingPlane("SATURN", "IAU_SATURN_DESPUN")
        planet_surface = surface_.Spheroid("SATURN", "IAU_SATURN", 60268.,
                                       54364.)
    
        #(surface_event, rel_surf_evt) = self.back_plane_setup(surface)
        (surface_event, rel_surf_evt,
         dist_from_instrument, obs_mask,
         blk_evt) = self.back_plane_setup(planet_surface)
        
        print "surface_event.perp:"
        print surface_event.perp
    
        #now get the sun
        sun_wrt_target = path_.connect("SUN", "SATURN")
    
        # get the relative event of the photon leaving the light source
        sun_dep_event = sun_wrt_target.photon_to_event(surface_event, 1)[0]
    
        surface_event.arr = surface_event.pos - sun_dep_event.pos
        print "surface_event.arr:"
        print surface_event.arr
    
        incidence_angle = surface_event.incidence_angle()
        print "incidence_angle:"
        print incidence_angle
    
        return incidence_angle

    def phase_angle_back_plane(self, min_range=0., max_range=sys.float_info.max):
        """gets the ring's backplane and returns whether in shadow or not.
        
            Input:
            min_range       minimum range for rings from planet's center
            max_range       maximum range for rings from planet's center
        
            Return:         2D array of 0 for not in rings, 1 for rings in
            shadow, and 2 for rings not in shadow
        """
    
        ignore = frame_.RingFrame("IAU_SATURN")
        surface = surface_.RingPlane("SATURN", "IAU_SATURN_DESPUN")
        planet_surface = surface_.Spheroid("SATURN", "IAU_SATURN", 60268.,
                                           54364.)
    
        #(surface_event, rel_surf_evt) = self.back_plane_setup(surface)
        (surface_event, rel_surf_evt,
         dist_from_instrument, obs_mask,
         blk_evt) = self.back_plane_setup(planet_surface)    
    
        #now get the sun
        sun_wrt_target = path_.connect("SUN", "SATURN")
        
        # get the relative event of the photon leaving the light source
        sun_dep_event = sun_wrt_target.photon_to_event(surface_event, 1)[0]
    
        surface_event.arr = surface_event.pos - sun_dep_event.pos
        
        phase_angle = surface_event.phase_angle()
        print "phase_angle:"
        print phase_angle
        
        return phase_angle
    
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
        light_wrt_target = path_.connect(light_path_id, target_path_id)
        
        # get the relative event of the photon leaving the light source
        light_dep_event = light_wrt_target.photon_to_event(obj_evt, 1)[1]
        print "light_dep_event.perp:"
        print light_dep_event.perp
        print "light_dep_event.arr:"
        print light_dep_event.arr
        print "light_dep_event.dep:"
        print light_dep_event.dep
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
        return (farther_mask, light_dep_event)


    def radius_back_plane(self, min_range=0., max_range=sys.float_info.max):
        """the backplane for radius values for the given snapshot. currently
            just for Cassini to rings... needs to be updated for all spacecraft
            and all targets.
            
            Return:     2D array of radius values
            """
        ignore = frame_.RingFrame("IAU_SATURN")
        surface = surface_.RingPlane("SATURN", "IAU_SATURN_DESPUN")

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
        ignore = frame_.RingFrame(frame_id)
        surface = surface_.Spheroid(path_id, frame_id, 60268., 54364.)
    
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
        ignore = frame_.RingFrame(frame_id)
        surface = surface_.Spheroid(path_id, frame_id, 60268., 54364.)
    
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
        
        if blocking_surface is not None:
            rel_blk_evt = blocking_surface.photon_to_event(image_event, 1)[1]
            """print "rel_blk_evt.perp:"
            print rel_blk_evt.perp
            print "rel_blk_evt.arr:"
            print rel_blk_evt.arr
            print "rel_blk_evt.dep:"
            print rel_blk_evt.dep
            rel_blk_evt.pos[rel_blk_evt.pos.mask] = rel_blk_evt.pos[0][0]
            rel_blk_evt.dep[rel_blk_evt.dep.mask] = rel_blk_evt.dep[0][0]
            rel_blk_evt.time[rel_blk_evt.time.mask] = rel_blk_evt.time[0][0]
            saturn_light_evt = rel_blk_evt.wrt_path("SUN")
            print "after wrt_path() called:"
            print "rel_blk_evt.perp:"
            print rel_blk_evt.perp
            print "rel_blk_evt.arr:"
            print rel_blk_evt.arr
            print "rel_blk_evt.dep:"
            print rel_blk_evt.dep
            print "saturn_light_evt.perp:"
            print saturn_light_evt.perp
            print "saturn_light_evt.arr:"
            print saturn_light_evt.arr
            print "saturn_light_evt.dep:"
            print saturn_light_evt.dep"""
            block_dist = np.sqrt(rel_blk_evt.pos.vals[...,0]**2 +
                                 rel_blk_evt.pos.vals[...,1]**2 +
                                 rel_blk_evt.pos.vals[...,2]**2)
            #make sure we don't compare values that did not intercept blocker
            block_dist[rel_blk_evt.pos.mask] = sys.float_info.max
            mask = block_dist < (dist_from_instrument - dist_tolerance)
        else:
            rel_blk_evt = None
            mask = dist_from_instrument == np.nan

        return (surface_event, rel_surf_evt, dist_from_instrument, mask, rel_blk_evt)
        #return (surface_event, rel_surf_evt)



################################################################################
# UNIT TESTS
################################################################################

class Test_Snapshot(unittest.TestCase):

    def runTest(self):

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
