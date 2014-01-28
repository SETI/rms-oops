################################################################################
# oops/cadence_/sequence.py: Sequence subclass of class Cadence
#
# 7/2/12 MRS - created
################################################################################

import numpy as np
from oops.array_ import *
from oops.cadence_.cadence import Cadence

class Sequence(Cadence):
    """Sequence is a Cadence subclass in which time steps are defined by a
    list."""

    def __init__(self, times, texp):
        """Constructor for a Sequence.

        Input:
            times       a list or 1-D array of times in seconds TDB.
            texp        the exposure time in seconds associated with each step.
                        This can be shorter than the time interval due to
                        readout times, etc. This can be:
                            - a positive constant, indicating that exposure
                              times are fixed.
                            - a list or 1-D array, listing the exposure time
                              associated with each time step.
                            - zero, indicating that each exposure duration lasts
                              up to the start of the next time step. In this
                              case, the last tabulated time is assumed to be
                              the end time rather than the start of the final
                              time step; the number of time steps is
                              len(times)-1 rather than len(times).
        """

        self.tlist = np.array(times)
        assert len(self.tlist.shape) == 1

        # Used for the inverse conversion; filled in only if needed
        self.indices = None
        self.tstrides = None

        if len(np.shape(texp)) > 0:
            self.texp = np.array(texp)
            self.tstrides = np.diff(self.tlist)
            assert self.texp.shape == self.tlist.shape
            self.is_continuous = (self.texp[:-1] == np.diff(self.tlist))
        elif texp == 0:
            self.texp = np.diff(self.tlist)
            self.tstrides = self.texp
            self.tlist = self.tlist[:-1]
            self.is_continuous = True
        else:
            (ignore,self.texp) = np.broadcast_arrays(self.tlist,float(texp))
            self.tstrides = np.diff(self.tlist)
            self.is_continuous = np.all(np.diff(self.tlist) == texp)

        self.steps = self.tlist.size

        self.lasttime = self.tlist[-1]
        self.time = (self.tlist[0], self.tlist[-1] + self.texp[-1])
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
        tstep_int = tstep.int().clip(0,self.steps-1)

        time = ((tstep - tstep_int) * self.texp[tstep_int.vals] +
                self.tlist[tstep_int.vals])     # Scalar + ndarray

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
        tstep_clipped = tstep.clip(0,self.steps-1)

        time_min = Scalar(self.tlist[tstep_clipped.vals])
        time_max = time_min + self.texp[tstep_clipped.vals]

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

        # Fill in the internals if they are still empty
        if self.indices  is None: self.indices  = np.arange(self.steps)
        if self.tstrides is None: self.tstrides = np.diff(self.tlist)

        time = Scalar.as_scalar(time)
        tstep = Scalar(np.interp(time.vals, self.tlist, self.indices))
        tstep_int = tstep.int().clip(0,self.steps-1)
        tstep_frac = ((time - self.tlist[tstep_int.vals]) /
                      self.texp[tstep_int.vals])
        tstep = tstep_frac + np.maximum(tstep_int,1)
                                # Write as Scalar + ndarray, never the reverse!

        new_mask = (tstep_frac.vals >= 1) & (time.vals != self.time[1])
                                # Second term to include upper limit
        if mask:
            new_mask |= (time.vals < self.time[0]) | (time.vals > self.time[1])

        if np.any(new_mask):
            tstep.mask = tstep.mask | new_mask

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

        # Fill in the internals if they are still empty
        if self.indices  is None: self.indices  = np.arange(self.steps)
        if self.tstrides is None: self.tstrides = np.diff(self.tlist)

        time = Scalar.as_scalar(time)
        tstep = Scalar(np.interp(time.vals, self.tlist, self.indices))
        tstep_int = tstep.int().clip(0,self.steps-1)
        time_frac = time - self.tlist[tstep_int.vals]

        if inclusive:
            return ((time_frac.vals >= 0) &
                    (time_frac.vals <= self.texp[tstep_int.vals]))
        else:
            return ((time_frac.vals >= 0) &
                    (time_frac.vals <  self.texp[tstep_int.vals]))

    def time_shift(self, secs):
        """Returns a duplicate of the given cadence, with all times shifted by
        a specified number of seconds."

        Input:
            secs        the number of seconds to shift the time later.
        """

        result = Sequence(self.tlist + secs, self.texp)

        result.indices = self.indices
        result.tstrides = self.tstrides

        return result

    def as_continuous(self):
        """Returns a shallow copy of the given cadence, with equivalent strides
        but with the property that the cadence is continuous.
        """

        if self.is_continuous: return self

        texp = np.empty(self.tlist.shape)
        texp[:-1] = np.diff(self.tlist)
        texp[ -1] = self.time[1]

        result = Sequence(self.tlist, texp)
        result.is_continuous = True
        return result

################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy.random as random

class Test_Sequence(unittest.TestCase):

    def runTest(self):

        # These are the tests for subclas Metronome. We define the Sequence so
        # that behavior should be identical

        # Continuous case
        # cadence = Metronome(100., 10., 10., 4)

        cadence = Sequence([100.,110.,120.,130.,140.], 0.)
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
        # texp = 8.
        # cadence = Metronome(100., 10., texp, 4)

        texp = 8.
        cadence = Sequence([100.,110.,120.,130.], texp)
        self.assertFalse(cadence.is_continuous)

        # time_at_tstep()
        self.assertEqual(cadence.time_at_tstep(0), 100.)
        self.assertEqual(cadence.time_at_tstep(1), 110.)
        self.assertEqual(cadence.time_at_tstep(4), 138.)
        self.assertEqual(cadence.time_at_tstep((3,4)), (130,138.))

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

