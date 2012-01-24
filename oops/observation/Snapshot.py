import numpy as np
import pylab
import unittest

import oops
import julian
import solar
import sys

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

    def radius_back_plane(self, min_range=0., max_range=sys.float_info.max):
        """the backplane for radius values for the given snapshot. currently
            just for Cassini to rings... needs to be updated for all spacecraft
            and all targets.
            
            Return:     2D array of radius values
            """
        #test some values relative to the LBL table file
        avg_time = (self.t0 + self.t1) * 0.5
        event_at_cassini = oops.Event(avg_time, (0,0,0), (0,0,0), "CASSINI")
        saturn_wrt_cassini = oops.Path.connect("SATURN", "CASSINI", "J2000")
        (abs_ev, rel_ev) = saturn_wrt_cassini.photon_to_event(event_at_cassini)


        surface_event = self.back_plane_setup()

        bp_data = np.sqrt(surface_event.pos.vals[...,0]**2 +
                          surface_event.pos.vals[...,1]**2)
        bp_data[np.isnan(bp_data)] = 0.
        bp_data[bp_data < min_range] = 0.
        bp_data[bp_data > max_range] = 0.

        return bp_data

    def back_plane_setup(self):
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
        buffer = np.empty((1024, 1024,2))
        buffer[:,:,1] = np.arange(1024).reshape(1024,1)
        buffer[:,:,0] = np.arange(1024)
        indices = oops.Pair(buffer + 0.5)
        
        rays = self.fov.los_from_uv(indices)

        #reverse so that they are arrival rays
        arrivals = -rays
        image_event = oops.Event(tdb, (0,0,0), (0,0,0), "CASSINI",
                                 self.frame_id, oops.Empty(), arrivals)
        
        ignore = oops.RingFrame("IAU_SATURN")
        surface = oops.RingPlane("SATURN", "IAU_SATURN_DESPUN")
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
