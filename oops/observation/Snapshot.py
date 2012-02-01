import numpy as np
import numpy.ma as ma
import pylab
import unittest

import oops
import julian
import solar
import sys
import math

################################################################################
# Snapshot Class
################################################################################

class Snapshot(oops.Observation):
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

        ignore = oops.RingFrame("IAU_SATURN")
        surface = oops.RingPlane("SATURN", "IAU_SATURN_DESPUN")

        surface_event = self.back_plane_setup(surface)

        bp_data = np.sqrt(surface_event.pos.vals[...,0]**2 +
                          surface_event.pos.vals[...,1]**2)
        
        bp_data[np.isnan(bp_data)] = 0.
        bp_data[bp_data < min_range] = 0.
        bp_data[bp_data > max_range] = 0.

        #now, for each point on surface that is in rings, check if in shadow
        planet_surface = oops.Spheroid("SATURN", "IAU_SATURN")
        sun_wrt_saturn = oops.Path.connect("SUN", "SATURN")
        sun_dep_event = sun_wrt_saturn.photon_to_event(surface_event, 1)[1]
        dir = surface_event.pos - sun_dep_event.pos


        intercept = planet_surface.intercept(sun_dep_event.pos, dir)[0]
        pos = intercept.vals
        dist = np.sqrt(pos[...,0]**2 + pos[...,1]**2 + pos[...,2]**2)
        dist = np.nan_to_num(dist)

        #for y in range(1024):
            #for x in range(1024):
                #if (bp_data[y][x] != 0.) and (dist[y][x] == 0.):
                    #bp_data[y][x] += max_range
        bmask = bp_data == 0.
        dmask = dist == 0.
        mask = bmask | dmask
        bp_masked = ma.array(bp_data, mask=mask)
        
        return bp_masked


    def radius_back_plane(self, min_range=0., max_range=sys.float_info.max):
        """the backplane for radius values for the given snapshot. currently
            just for Cassini to rings... needs to be updated for all spacecraft
            and all targets.
            
            Return:     2D array of radius values
            """
        ignore = oops.RingFrame("IAU_SATURN")
        surface = oops.RingPlane("SATURN", "IAU_SATURN_DESPUN")

        surface_event = self.back_plane_setup(surface)

        bp_data = np.sqrt(surface_event.pos.vals[...,0]**2 +
                          surface_event.pos.vals[...,1]**2)
        bp_data[np.isnan(bp_data)] = 0.
        bp_data[bp_data < min_range] = 0.
        bp_data[bp_data > max_range] = 0.
        mask = bp_data == 0.
        bp_masked = ma.array(bp_data, mask=mask)

        return bp_masked

    def polar_back_plane(self, min_range=0., max_range=sys.float_info.max):
        """the backplane for polar theta value for the given snapshot. currently
            just for Cassini to rings... needs to be updated for all spacecraft
            and all targets.
        
            Return:     2D array of radius values
            """
        ignore = oops.RingFrame("IAU_SATURN")
        surface = oops.Spheroid("SATURN", "IAU_SATURN")
    
        surface_event = self.back_plane_setup(surface)
        
        x = surface_event.pos.vals[...,0]
        y = surface_event.pos.vals[...,1]
        
        theta = np.arctan2(y,x) % (2.*np.pi)
        mask = x == np.nan
        theta_mask = ma.array(theta, mask=mask)
        
        return theta_mask

    def latitude_c_back_plane(self, min_range=0., max_range=sys.float_info.max):
        """the backplane for polar theta value for the given snapshot. currently
            just for Cassini to rings... needs to be updated for all spacecraft
            and all targets.
        
            Return:     2D array of radius values
            """
        ignore = oops.RingFrame("IAU_SATURN")
        surface = oops.Spheroid("SATURN", "IAU_SATURN")
    
        surface_event = self.back_plane_setup(surface)
    
        x = surface_event.pos.vals[...,0]
        y = surface_event.pos.vals[...,1]
        z = surface_event.pos.vals[...,2]
    
        theta = ( np.arctan2(z,np.sqrt(x*x+y*y)) + 0.5 * np.pi) % (np.pi)
        mask = x == np.nan
        theta_mask = ma.array(theta, mask=mask)
        
        return theta_mask

    def back_plane_setup(self, surface):
        """Currently this is set up purely for ring planes.  Must be generalized
            in the future.
            
            Return:     surface_event - the event where the photons left the
                        surface.
            """
        #first get the tdb time when instrument took measurement
        #get the average between the start and stop time to get the mid time
        tdb = (self.t0 + self.t1) * 0.5
        
        #construct a buffer from which we will create rays through these pixels
        #buffer = np.empty((self.fov.uv_shape[0].vals, self.fov.uv_shape[1].vals,2))
        #buffer[:,:,0] = np.arange(uv_shape[0].vals).reshape(uv_shape[0].vals,1)
        #buffer[:,:,1] = np.arange(uv_shape[1].vals)
        #figure out the proper syntax... for initial testing, just use 1024x1024
        uv_shape = self.fov.uv_shape.vals
        buffer = np.empty((uv_shape[0], uv_shape[1], 2))
        buffer[:,:,1] = np.arange(uv_shape[1]).reshape(uv_shape[1],1)
        buffer[:,:,0] = np.arange(uv_shape[0])
        
        #buffer = np.empty((1024, 1024,2))
        #buffer[:,:,1] = np.arange(1024).reshape(1024,1)
        #buffer[:,:,0] = np.arange(1024)
        indices = oops.Pair(buffer + 0.5)
        
        rays = self.fov.los_from_uv(indices)

        #reverse so that they are arrival rays
        arrivals = -rays
        #arrivals = -rays[445][527]
        #arrivals = -rays[474][820]
        image_event = oops.Event(tdb, (0,0,0), (0,0,0), "CASSINI",
                                 self.frame_id, oops.Empty(), arrivals)
        
        surface_event = surface.photon_to_event(image_event, 1)[0]

        return surface_event


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
