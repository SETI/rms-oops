import numpy as np
import pylab
import unittest

import oops
import julian
import solar

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

    def radius_back_plane(self):
        """the backplane for radius values for the given snapshot. currently
            just for Cassini to rings... needs to be updated for all spacecraft
            and all targets.
            
            Return:     2D array of radius values
            """
        #first get the tdb time when instrument took measurement
        #get the average between the start and stop time to get the mid time
        tdb = (self.t0 + self.t1) * 0.5
        
        #construct a buffer from which we will create rays through these pixels
        #buffer = np.empty((self.fov.uv_shape[0], self.fov.uv_shape[1],2))
        #buffer[:,:,0] = np.arange(uv_shape[0]).reshape(uv_shape[0],1)
        #buffer[:,:,1] = np.arange(uv_shape[1])
        #figure out the proper syntax... for initial testing, just use 1024x1024
        buffer = np.empty((1024, 1024,2))
        buffer[:,:,0] = np.arange(1024).reshape(1024,1)
        buffer[:,:,1] = np.arange(1024)
        indices = oops.Pair(buffer + 0.5)
        
        rays = self.fov.los_from_uv(indices)
        
        print 'frame id:'
        print self.frame_id
        frame = oops.Frame.connect("J2000", self.frame_id)
        transform = frame.transform_at_time(tdb)
        arrivals = transform.rotate(-rays)
        image_event = oops.Event(tdb, (0,0,0), (0,0,0), "CASSINI",
                                 self.frame_id, oops.Empty(), arrivals)
        
        #print oops.FRAME_REGISTRY['CASSINI']
        ignore = oops.RingFrame("SATURN")
        surface = oops.RingPlane("SATURN", "SATURN_DESPUN")
        #even the following commented out line gives a FRAME_REGISTRY error
        #surface = oops.RingPlane("CASSINI", "CASSINI")
        ring_event = surface.photon_to_event(image_event)[0]
        
        bp_data = np.sqrt(ring_event.pos[0]**2 + ring_event.pos[1]**2)
        #for i in range(self.fov.uv_shape[0]):
            #for j in range(self.fov.uv_shape[1]):
        #figure out the proper syntax... for initial testing, just use 1024x1024
        for i in range(1024):
            for j in range(1024):
                if bp_data[i][j] < 60298.:
                    bp_data[i][j] = 0.
                elif bp_data[i][j] > 483000.:   #include just for testing purposes
                    bp_data[i][j] = 0.
        return bp_data


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
