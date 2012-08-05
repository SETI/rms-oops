################################################################################
# oops_/cadence/metronome.py: Metronome subclass of class Cadence
#
# 7/2/12 MRS - created
################################################################################

import numpy as np
from oops_.array.all import *
from oops_.cadence.cadence_ import Cadence

class Metronome(Cadence):
    """Metronome is a Cadence subclass in which time steps occur at uniform
    intervals."""

    def __init__(self, tstart, tstride, texp, steps):
        """Constructor for a Metronome.

        Input:
            tstart      the start time of the observation in seconds TDB.
            tstride     the interval in seconds from the start of one time step
                        to the start of the next.
            texp        the exposure time in seconds associated with each step.
                        This may be shorter than tstride due to readout times,
                        etc.
            steps       the number of time steps.
        """

        self.tstart = float(tstart)
        self.tstride = float(tstride)
        self.texp = float(texp)
        self.steps = steps

        self.is_continuous = (self.texp == self.tstride)
        self.tscale = self.tstride / self.texp

        self.lasttime = self.tstart + self.tstride * (self.steps-1)
        self.time = (self.tstart, self.lasttime + self.texp)
        self.midtime = (self.time[0] + self.time[1]) * 0.5

        self.shape = (self.steps,)

        return

    def time_at_tstep(self, tstep, mask=False):
        """Returns the time associated with the given time step. This method
        supports non-integer step values.

        Input:
            tstep       a Scalar time step index or a Pair or Tuple of indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep)

        if self.is_continuous:
            time = self.time[0] + self.tstride * tstep
        else:
            tstep_int = tstep.int()
            time = (self.time[0] + self.tstride * tstep_int
                                 + (tstep - tstep_int) * self.texp)
            # The maximum tstep is still inside the domain
            max_mask = (tstep.vals == self.steps)
            time.replace(max_mask, self.time[1], newmask=False)

        if mask:
            is_inside = (tstep.vals >= 0) & (tstep.vals <= self.steps)
            if not np.all(is_inside):
                time.mask = time.mask | np.logical_not(is_inside)

        return time

    def time_range_at_tstep(self, tstep, mask=False):
        """Returns the range of time associated with the given integer time
        step index.

        Input:
            indices     a Scalar time step index or a Pair or Tuple of indices.
            mask        True to mask values outside the time limits.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        tstep = Scalar.as_int(tstep)

        time_min = self.time[0] + tstep * self.tstride
        time_max = time_min + self.texp

        if mask:
            is_inside = (tstep.vals >= 0) & (tstep.vals <= self.steps)
            if not np.all(is_inside):
                time_mask = tstep.mask | np.logical_not(is_inside)
                time_min.mask = time_mask
                time_max.mask = time_mask

        return (time_min, time_max)

    def tstep_at_time(self, time, mask=False):
        """Returns a the Scalar time step index or a Pair or Tuple of indices
        associated with a time in seconds TDB.

        Input:
            time        a Scalar of times in seconds TDB.
            mask        True to mask time values not sampled within the cadence.

        Return:         a Scalar, Pair or Tuple of time step indices.
        """

        time = Scalar.as_scalar(time)

        tstep = (time - self.time[0]) / self.tstride

        if not self.is_continuous:
            tstep_int = tstep.int()
            tstep_frac = (tstep - tstep_int) * self.tscale
            is_invalid = (tstep_frac >= 1.)
            if np.any(is_invalid):
                tstep_frac = frac.clip(0,1)
                tstep_frac.mask = tstep_frac.mask | is_invalid

            tstep = tstep_int + tstep_frac

        if mask:
            is_inside = (tstep.vals >= 0) & (tstep.vals <= self.steps)
            if not np.all(is_inside):
                tstep.mask = tstep.mask | np.logical_not(is_inside)

        return tstep

    def time_is_inside(self, time, inclusive=True):
        """Returns a boolean Numpy array indicating which elements in a given
        Scalar of times fall inside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to include the end moment of a time interval;
                        False to exclude.

        Return:         a boolean Numpy array indicating which time values are
                        sampled by the cadence.
        """

        time = Scalar.as_scalar(time)

        if self.is_continuous:
            return Cadence.time_is_inside(self, time, inclusive=inclusive)
        else:
            time_mod_vals = (time.vals - self.time[0]) % self.tstride

            if inclusive:
                return ((time_mod_vals <= self.texp) &
                        (time.vals >= self.time[0]) &
                        (time.vals <= self.time[1]))
            else:
                return ((time_mod_vals < self.texp) &
                        (time.vals >= self.time[0]) &
                        (time.vals <  self.time[1]))

    def time_shift(self, secs):
        """Returns a duplicate of the given cadence, with all times shifted by
        a specified number of seconds."

        Input:
            secs        the number of seconds to shift the time later.
        """

        return Metronome(self.tstart + secs,
                         self.tstride, self.texp, self.steps)

    def as_continuous(self):
        """Returns a shallow copy of the given cadence, with equivalent strides
        but with the property that the cadence is continuous.
        """

        return Metronome(self.tstart, self.tstride, self.tstride, self.steps)

################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy.random as random

class Test_Metronome(unittest.TestCase):

    def runTest(self):

        # Continuous case
        cadence = Metronome(100., 10., 10., 4)
        self.assertTrue(cadence.is_continuous)

        # time_at_tstep()
        self.assertEqual(cadence.time_at_tstep(0), 100.)
        self.assertEqual(cadence.time_at_tstep(1), 110.)
        self.assertEqual(cadence.time_at_tstep(4), 140.)
        self.assertEqual(cadence.time_at_tstep((3,4)), (130.,140.))

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,140])
        self.assertEqual(cadence.time_at_tstep(tstep), time)

        tstep = ([0,1],[2,3],[4,5])
        test = cadence.time_at_tstep(tstep, mask=True)
        self.assertTrue(np.all(test.mask ==
                               [[False,False],[False,False],[False,True]]))

        # time_is_inside()
        time  = ([100,110],[120,130],[140,150])
        self.assertTrue(np.all(cadence.time_is_inside(time) ==
                               [[True,True],[True,True],[True,False]]))
        self.assertTrue(np.all(cadence.time_is_inside(time, inclusive=False) ==
                               [[True,True],[True,True],[False,False]]))

        # tstep_at_time()
        tstep = Scalar(7*random.rand(100,100) - 1.)
        time = cadence.time_at_tstep(tstep)
        test = cadence.tstep_at_time(time)
        self.assertTrue(abs(tstep - test) < 1.e-14)

        # mask testing
        mask = (tstep.vals < 0) | (test.vals > cadence.steps)
        test = cadence.tstep_at_time(time, mask=True)
        self.assertTrue(abs(tstep - test.vals) < 1.e-14)
        self.assertTrue(np.all(test.mask == mask))
        self.assertTrue(np.all(cadence.time_is_inside(time) ==
                               np.logical_not(mask)))

        # time_range_at_tstep()
        (time0, time1) = cadence.time_range_at_tstep(tstep)
        self.assertTrue(np.all(time0.vals == 10*((time0/10).int()).vals))
        self.assertTrue(np.all(time1.vals == 10*((time1/10).int()).vals))

        self.assertTrue(np.all(np.abs(time1.vals - time0.vals - 10.) < 1.e-14))

        unmasked = np.logical_not(mask)
        self.assertTrue(np.all(time0.vals[unmasked] >= cadence.time[0]))
        self.assertTrue(np.all(time1.vals[unmasked] >= cadence.time[0]))
        self.assertTrue(np.all(time0.vals[unmasked] <= cadence.time[1]))
        self.assertTrue(np.all(time1.vals[unmasked] <= cadence.time[1]))

        self.assertTrue(np.all(time0.vals[unmasked] <= time.vals[unmasked]))
        self.assertTrue(np.all(time1.vals[unmasked] >= time.vals[unmasked]))

        # time_shift()
        shifted = cadence.time_shift(1.)
        time_shifted = shifted.time_at_tstep(tstep)

        self.assertTrue(np.all(np.abs(time_shifted.vals
                                      - time.vals - 1.) < 1.e-13))

        ####################################
        # Discontinuous case

        texp = 8.
        cadence = Metronome(100., 10., texp, 4)
        self.assertFalse(cadence.is_continuous)

        # time_at_tstep()
        self.assertEqual(cadence.time_at_tstep(0), 100.)
        self.assertEqual(cadence.time_at_tstep(1), 110.)
        self.assertEqual(cadence.time_at_tstep(4), 138.)
        self.assertEqual(cadence.time_at_tstep((3,4)), (130.,138.))

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,138])
        self.assertEqual(cadence.time_at_tstep(tstep), time)

        tstep = ([0,1],[2,3],[4,5])
        test = cadence.time_at_tstep(tstep, mask=True)
        self.assertTrue(np.all(test.mask ==
                               [[False,False],[False,False],[False,True]]))

        # time_is_inside()
        time  = ([100,110],[120,130],[138,150])
        self.assertTrue(np.all(cadence.time_is_inside(time) ==
                                [[True,True],[True,True],[True,False]]))
        self.assertTrue(np.all(cadence.time_is_inside(time, inclusive=False) ==
                                [[True,True],[True,True],[False,False]]))
        self.assertTrue(cadence.time_is_inside(138., inclusive=True))
        self.assertFalse(cadence.time_is_inside(138., inclusive=False))

        # tstep_at_time()
        tstep = Scalar(7*random.rand(100,100) - 1.)
        time = cadence.time_at_tstep(tstep)
        test = cadence.tstep_at_time(time)
        self.assertTrue(abs(tstep - test) < 1.e-14)

        # mask testing
        mask = (tstep.vals < 0) | (test.vals > cadence.steps)
        test = cadence.tstep_at_time(time, mask=True)
        self.assertTrue(abs(tstep - test.vals) < 1.e-14)
        self.assertTrue(np.all(test.mask == mask))
        self.assertTrue(np.all(cadence.time_is_inside(time) ==
                               np.logical_not(mask)))

        # time_range_at_tstep()
        (time0, time1) = cadence.time_range_at_tstep(tstep)
        self.assertTrue(np.all(time0.vals == 10*((time0/10).int()).vals))
        self.assertTrue(np.all(time1.vals == time0.vals + texp))

        unmasked = np.logical_not(mask)
        self.assertTrue(np.all(time0.vals[unmasked] >= cadence.time[0]))
        self.assertTrue(np.all(time1.vals[unmasked] >= cadence.time[0]))
        self.assertTrue(np.all(time0.vals[unmasked] <= cadence.time[1]))
        self.assertTrue(np.all(time1.vals[unmasked] <= cadence.time[1]))

        self.assertTrue(np.all(time0.vals[unmasked] <= time.vals[unmasked]))
        self.assertTrue(np.all(time1.vals[unmasked] >= time.vals[unmasked]))

        # time_shift()
        shifted = cadence.time_shift(1.)
        time_shifted = shifted.time_at_tstep(tstep)

        self.assertTrue(np.all(np.abs(time_shifted.vals
                                      - time.vals - 1.) < 1.e-13))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

