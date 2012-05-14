################################################################################
# oops_/obs/baseclass.py: Abstract class Observation
#
# 2/11/12 Modified (MRS) - updated for style
# 3/9/12 MRS - new methods fleshed out in preparation for additional observation
#   classes such as pushbrooms and raster scanners.
# 5/14/12 MRS: addeded gridless_event() method.
################################################################################

import numpy as np

from oops_.array.all import *
from oops_.config import *
from oops_.event import Event
from oops_.meshgrid import Meshgrid

class Observation(object):
    """An Observation is an abstract class that defines the timing and pointing
    of the samples that comprise a data product.

    An observation is always regarded as having at least three axes (t,u,v),
    where t increases with time during the observation, u is a spatial axis
    intended to increase rightward when displayed, and v is a spatial axis
    intended to increase upward or downward when displayed. Additional axes
    (such as wavelength bands), may appear after these first three. The order of
    the axes is not specified.

    When indices have non-integer values, the integer part identifies one
    "corner" of the sample, and the fractional part locates a point within the
    sample, i.e., part way from the start time to the end time of an
    integration, or a location inside the boundaries of a spatial pixel.
    Half-integer indices falls at the midpoint of each sample.

    At minimum, attributes are used to describe the observation:
        target          the registered name of the target body.
        time            a tuple or Pair defining the start time and end time of
                        the observation overall, in seconds TDB.
        fov             a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y).
        path_id         the registered ID of a path co-located with the
                        instrument.
        frame_id        the registered ID of a coordinate frame fixed to the
                        optics of the instrument. This frame should have its
                        Z-axis pointing outward near the center of the line of
                        sight, with the X-axis pointing rightward and the y-axis
                        pointing downward.
        subfields       a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
    """

    ####################################################
    # Methods to be defined for each subclass
    ####################################################

    def __init__(self):
        """A constructor."""

        pass

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

        pass

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

        pass

    ####################################################
    # Subarray support methods
    ####################################################

    def insert_subfield(self, key, value):
        """Adds a given subfield to the Event."""

        self.subfields[key] = value
        self.__dict__[key] = value      # This makes it an attribute as well

    def delete_subfield(self, key):
        """Deletes a subfield, but not arr or dep."""

        if key in ("arr","dep"):
            self.subfields[key] = Empty()
            self.__dict__[key] = self.subfields[key]
        elif key in self.subfields.keys():
            del self.subfields[key]
            del self.__dict__[key]

    def delete_subfields(self):
        """Deletes all subfields."""

        for key in self.subfields.keys():
            if key not in ("arr","dep"):
                del self.subfields[key]
                del self.__dict__[key]

    ####################################################
    # Methods probably not requiring overrides
    ####################################################

    def midtime_at_uv(self, uv, extras=()):
        """Returns the mid-time for the selected spatial pixel (u,v)."""

        (time0, time1) = self.times_at_uv(uv, extras=())
        return (time0 + time1) / 2.

    def event_at_grid(self, meshgrid, t=Scalar(0.5)):
        """Returns an event object describing the arrival of a photon at a set
        of locations defined by the given meshgrid.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view.
            t           a Scalar of fractional time offsets into each exposure;
                        default is 0.5. The shape of this Scalar will broadcast
                        with the shape of the meshgrid.

        Return:         the corresponding event.
        """

        (time0, time1) = self.times_at_uv(meshgrid.uv, meshgrid.extras)
        times = t * time1 + (1-t) * time0

        event = Event(times, Vector3((0,0,0)), Vector3((0,0,0)),
                             self.path_id, self.frame_id)

        # Insert the arrival directions
        event.insert_subfield("arr", -meshgrid.los)

        return event

    def gridless_event(self, t=Scalar(0.5)):
        """Returns an event object describing the arrival of a photon at an
        instrument, irrespective of the direction.

        Input:
            t           a Scalar of fractional time offsets into each exposure;
                        default is 0.5.

        Return:         the corresponding event.
        """

        (time0, time1) = self.time
        times = t * time1 + (1-t) * time0

        event = Event(times, Vector3((0,0,0)), Vector3((0,0,0)),
                             self.path_id, self.frame_id)

        return event

    ####################################################
    # Photon methods
    ####################################################

    # This procedure assumes that movement along a path is very limited during
    # the exposure time of an individual pixel. It could fail to converge if
    # there is a large gap in timing between adjacent pixels at a time when the
    # object is crossing that gap. However, even then, it should select roughly
    # the correct location. It could also fail to converge during a fast slew.
    #
    # It is safe to call the function with iters=0 for a Snapshot observation.

    def uv_from_path(self, path, extras=(), quick=None, derivs=False, iters=3):
        """Solves for the (u,v) indices of an object in the field of view, given
        its path.

        Input:
            path        a Path object.
            extras      a tuple of Scalar index values defining any extra
                        indices into the observation's array, should these
                        values be relevant.
            quick       defines how to use QuickPaths and QuickFrames.
            derivs      True to include derivatives d(u,v)/dt, neglecting any
                        sweep motion within the observation.
            iters       maximum number of iterations

        Return:
            uv_pair     the (u,v) indices of the pixel in which the point was
                        found. The path is evaluated at the mid-time of this
                        pixel.
        """

        # Iterate until convergence or until the limit is reached...
        zero = Scalar((0,0,0))
        path_event_guesses = None

        # Broadcast the times across the paths if necessary
        times = Scalar(tcube).reshape([2] + [1]*len(path.shape))

        max_dt = np.inf
        for iter in range(iters):

            # Locate the object in the field of view at two times
            event = Event(times, zero, zero, self.origin_id, self.frame_id)
            path_event = path.photon_to_event(event, quick=quick,
                                              guess=path_event_guesses)
            path_event_guesses = path_event.time
            uv = self.fov.uv_from_los(-event.arr)

            # Update the times based on the locations
            times = self.midtime_at_uv(uv)

            # Test for convergence
            prev_max_dt = max_dt
            max_dt = abs(times[0] - times[1]).max()

            if LOGGING.observation_iterations:
                print LOGGING.prefix, "Observation.uv_from_path", iter, max_dt

            if max_dt <= precision or max_dt >= prev_max_dt: break

        # Return the results at the best mid-time
        event = Event((times[0] + times[1])/2., zero, zero,
                       self.origin_id, self.frame_id)
        ignore = path.photon_to_event(event, quick, derivs=derivs)
        # If derivs is True, then event.arr.d_dt is defined, unnormalized

        uv = self.fov.uv_from_los(-event.arr)
        # If derivs is True, then uv.d_dlos is defined

        # Combine the derivatives if necessary
        if derivs:
            duv_dt = event.arr.d_dt/event.arr.plain().norm() * uv.d_dlos
            uv.insert_subfield("d_dt", duv_dt)

        return uv

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Observation(unittest.TestCase):

    def runTest(self):

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
