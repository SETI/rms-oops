################################################################################
# oops/frame_/spicetype1frame.py: Subclass SpiceType1Frame of Frame
################################################################################

import numpy as np

import cspice
from polymath import *

from oops.frame_.frame import Frame
from oops.config       import QUICK
from oops.transform    import Transform
import oops.spice_support as spice

class SpiceType1Frame(Frame):
    """A SpiceType1Frame is a Frame object defined within the SPICE toolkit as a
    Type 1 (discrete) C kernel."""

    def __init__(self, spice_frame, spice_host, tick_tolerance,
                       spice_reference="J2000", id=None):
        """Constructor for a SpiceType1Frame.

        Input:
            spice_frame     the name or integer ID of the destination frame or
                            of the central body as used in the SPICE toolkit.

            spice_host      the name or integer ID of the spacecraft. This is
                            needed to evaluate the spacecraft clock ticks.

            tick_tolerance  a number or string defining the error tolerance in
                            spacecraft clock ticks for the frame returned.

            spice_reference the name or integer ID of the reference frame as
                            used in the SPICE toolkit; "J2000" by default.

            id              the name or ID under which the frame will be
                            registered. By default, this will be the value of
                            spice_id if that is given as a string; otherwise
                            it will be the name as used by the SPICE toolkit.
        """

        # Interpret the SPICE frame and reference IDs
        (self.spice_frame_id,
         self.spice_frame_name) = spice.frame_id_and_name(spice_frame)

        (self.spice_reference_id,
         self.spice_reference_name) = spice.frame_id_and_name(spice_reference)

        (self.spice_body_id,
         self.spice_body_name) = spice.body_id_and_name(spice_host)

        # Fill in the time tolerances
        if type(tick_tolerance) == type(''):
            self.tick_tolerance = cspice.sctiks(spice_body_id, tick_tolerance)
        else:
            self.tick_tolerance = tick_tolerance

        self.time_tolerance = None      # filled in on first use

        # Fill in the Frame ID and save it in the SPICE global dictionary
        self.frame_id = id or self.spice_frame_name
        spice.FRAME_TRANSLATION[self.spice_frame_id]   = self.frame_id
        spice.FRAME_TRANSLATION[self.spice_frame_name] = self.frame_id

        # Fill in the reference wayframe
        reference_id = spice.FRAME_TRANSLATION[self.spice_reference_id]
        self.reference = Frame.as_wayframe(reference_id)

        # Fill in the origin waypoint
        self.spice_origin_id   = cspice.frinfo(self.spice_frame_id)[0]
        self.spice_origin_name = cspice.bodc2n(self.spice_origin_id)

        try:
            self.origin = Path.as_waypoint(origin_id)
        except KeyError:
            # If the origin path was never defined, define it now
            origin_path = SpicePath(origin_id)
            self.origin = origin_path.waypoint

        # No shape, no keys
        self.shape = ()
        self.keys = set()

        # Always register a SpiceType1Frame
        # This also fills in the waypoint
        self.register()

        # Initialize cache
        self.cached_shape = None
        self.cached_time = None
        self.cached_transform = None

    ########################################

    def transform_at_time(self, time, quick={}):
        """A Transform that rotates from the reference frame into this frame.

        Input:
            time            a Scalar time.

        Return:             the corresponding Tranform applicable at the
                            specified time(s).
        """

        # Fill in the time tolerance in seconds
        if self.time_tolerance is None:
            time = Scalar.as_scalar(time)
            ticks = cspice.sce2c(self.spice_body_id, time.vals)
            ticks_per_sec = cspice.sce2c(self.spice_body_id,
                                         time.vals + 1.) - ticks
            self.time_tolerance = self.tick_tolerance / ticks_per_sec

        # Check to see if the cached transform is adequate
        time = Scalar.as_scalar(time)
        if np.shape(time.vals) == self.cached_shape:
            diff = np.abs(time.vals - self.cached_time)
            if np.all(diff < self.time_tolerance):
                return self.cached_transform

        # A single input time can be handled quickly
        if time.shape == ():
            ticks = cspice.sce2c(self.spice_body_id, time.vals)
            (matrix3, true_ticks) = cspice.ckgp(self.spice_frame_id, ticks,
                                                self.tick_tolerance,
                                                self.spice_reference_name)

            self.cached_shape = time.shape
            self.cached_time = cspice.sct2e(self.spice_body_id, true_ticks)
            self.cached_transform = Transform(matrix3, Vector3.ZERO,
                                              self.frame_id, self.reference_id)
            return self.cached_transform

        # Create the buffers
        matrix = np.empty(time.shape + (3,3))
        omega  = np.zeros(time.shape + (3,))
        true_times = np.empty(time.shape)

        # If all the times are close, we can return more quickly
        time_min = time.vals.min()
        time_max = time.vals.max()
        if (time_max - time_min) < self.time_tolerance:
            ticks = cspice.sce2c(self.spice_body_id, (time_min + time_max)/2.)
            (matrix3, true_ticks) = cspice.ckgp(self.spice_frame_id, ticks,
                                                self.tick_tolerance,
                                                self.spice_reference_name)

            matrix[...] = matrix
            true_times[...] = cspice.sct2e(self.spice_body_id, true_ticks)

            self.cached_shape = time.shape
            self.cached_time = true_times
            self.cached_transform = Transform(matrix, omega,
                                              self.frame_id, self.reference_id)
            return self.cached_transform
        
        # Otherwise, iterate through the array...
        for (i,t) in np.ndenumerate(time.vals):
            ticks = cspice.sce2c(self.spice_body_id, t)
            (matrix3, true_ticks) = cspice.ckgp(self.spice_frame_id, ticks,
                                                self.tick_tolerance,
                                                self.spice_reference_name)
            matrix[i] = matrix3
            true_times[i] = cspice.sct2e(self.spice_body_id, true_ticks)

        self.cached_shape = time.shape
        self.cached_time = true_times
        self.cached_transform = Transform(matrix, omega,
                                          self.frame_id, self.reference_id)
        return self.cached_transform

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_SpiceType1Frame(unittest.TestCase):

    def runTest(self):
        pass                # TBD

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
