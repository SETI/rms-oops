################################################################################
# oops_/obs/snapshot.py: Subclass Snapshot of class Observation
################################################################################

import numpy as np

from oops_.obs.observation_ import Observation
from oops_.array.all import *
from oops_.config import QUICK

class Snapshot(Observation):
    """A Snapshot is an Observation consisting of a 2-D image made up of pixels
    all exposed at the same time."""

    ZERO_PAIR = Pair((0,0))

    def __init__(self, time, fov, path_id, frame_id, **subfields):
        """Constructor for a Snapshot.

        Input:
            time        a tuple defining the start time and end time of the
                        observation overall, in seconds TDB.
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

    def times_at_uv(self, uv_pair, extras=()):
        """Returns the start time and stop time associated with the selected
        spatial pixel index (u,v).

        Input:
            uv_pair     a Pair of spatial indices (u,v).
            extras      Scalars of any extra index values needed to define the
                        timing of array elements.
        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.

                        If derivs is True, then each time has a subfield "d_duv"
                        defining the change in time associated with a 1-pixel
                        step along the u and v axes. This is represented by a
                        MatrixN with item shape [1,2].
        """

        return self.time

    def sweep_duv_dt(self, uv_pair, extras=()):
        """Returns the mean local sweep speed of the instrument in the (u,v)
        directions.

        Input:
            uv_pair     a Pair of spatial indices (u,v).
            extras      Scalars of any extra index values needed to define the
                        timing of array elements.

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """

        return Snapshot.ZERO_PAIR

    def uv_from_path(self, path, quick=QUICK, derivs=False):
        """Solves for the (u,v) indices of an object in the field of view, given
        its path.

        Input:
            path        a Path object.
            quick       defines how to use QuickPaths and QuickFrames.
            derivs      True to include derivatives.

        Return:
            uv_pair     the (u,v) indices of the pixel in which the point was
                        found. The path is evaluated at the mid-time of this
                        pixel.

        For paths that fall outside the field of view, the returned values of
        time and index are masked.
        """

        # Snapshots are easy and require zero iterations
        return Observation.uv_from_path(self, path, (), quick, derivs, iters=0)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Snapshot(unittest.TestCase):

    def runTest(self):

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
