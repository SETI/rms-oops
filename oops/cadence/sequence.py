################################################################################
# oops/cadence/sequence.py: Sequence subclass of class Cadence
################################################################################

import numpy as np
from polymath import Boolean, Scalar, Qube

from oops.cadence import Cadence

class Sequence(Cadence):
    """Cadence subclass in which time steps are defined by a list."""

    def __init__(self, tlist, texp):
        """Constructor for a Sequence.

        Input:
            tlist       a Scalar, list or 1-D array of times in seconds TDB.
            texp        the exposure time in seconds associated with each step.
                        This can be shorter than the time interval due to
                        readout times, etc. It could also potentially be longer.
                        The input value of texp can be:
                        - a positive constant, indicating that exposure times
                          are fixed.
                        - a list or 1-D array, listing the exposure time
                          associated with each time step.
                        - zero, indicating that each exposure duration lasts up
                          to the start of the next time step. In this case, the
                          last tabulated time is assumed to be the end time of
                          the previous exposure rather than the start of a final
                          time step; the number of time steps is therefore
                          len(tlist)-1 rather than len(tlist).
        """

        # Work with Numpy arrays initially
        if isinstance(tlist, Scalar):
            assert not np.any(tlist.mask)
            tlist = tlist.vals

        if isinstance(texp, Scalar):
            assert not np.any(texp.mask)
            texp = texp.vals

        tlist = np.asfarray(tlist)
        assert np.ndim(tlist) == 1
        assert tlist.size > 1

        tstrides = np.diff(tlist)

        self._state_texp = texp

        # Interpret texp
        if np.shape(texp):          # texp is an array
            texp = np.asfarray(texp)
            assert texp.shape == tlist.shape
            assert np.all(texp > 0.)

            self.min_tstride = np.min(tstrides)
            self.max_tstride = np.max(tstrides)
            self.is_continuous = np.all(texp[:-1] >= tstrides)
            self.is_unique = np.all(texp[:-1] <= tstrides)

            tstop = tlist + texp
            self._tstop_is_ordered = np.any(np.diff(tstop) < 0.)

        elif texp:                  # texp is a nonzero constant
            assert texp > 0.
            self.min_tstride = np.min(tstrides)
            self.max_tstride = np.max(tstrides)
            self.is_continuous = (texp >= self.max_tstride)
            self.is_unique = (texp <= self.min_tstride)

            # Create a filled array in place of the single value
            saved_texp = texp
            texp = np.empty(tlist.shape)
            texp.fill(saved_texp)

            tstop = tlist + texp
            self._tstop_is_ordered = True

        else:                       # use diffs to define texp
            texp = tstrides
            tstop = tlist[1:]
            tlist = tlist[:-1]      # last time is not a time step
            assert np.all(texp > 0.)

            tstrides = tstrides[:-1]
            self.min_tstride = np.min(tstrides)
            self.max_tstride = np.max(tstrides)
            self.is_continuous = True
            self.is_unique = True
            self._tstop_is_ordered = True

        # Convert back to Scalar and save
        # as_readonly() ensures that these inputs cannot be modified by
        # something external to the object.
        self.tlist  = Scalar(tlist).as_readonly()
        self.texp   = Scalar(texp).as_readonly()
        self._tstop = Scalar(tstop).as_readonly()

        self.steps = self.tlist.size
        self._max_tstep = self.steps - 1

        # Used for the inverse conversion
        self._interp_y = np.arange(self.steps, dtype='float')
        self._is_gapless = self.is_continuous and self.is_unique

        # Fill in required attributes
        self.lasttime = self.tlist.vals[-1]
        self.time = (self.tlist.vals[0],
                     self.tlist.vals[-1] + self.texp.vals[-1])
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.shape = self.tlist.shape

        return

    def __getstate__(self):
        return (self.tlist, self._state_texp)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

        Input:
            tstep       a Pair of time step index values.
            remask      True to mask values outside the time limits.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep, recursive=derivs)
        tstep_int = tstep.int(top=self.steps, remask=remask, clip=True,
                              inclusive=inclusive)
        tstep_frac = (tstep - tstep_int).clip(0, 1, remask=remask,
                                                    inclusive=inclusive)

        time = (self.tlist[tstep_int.vals] + tstep_frac *
                                             self.texp[tstep_int.vals])
        return time

    #===========================================================================
    def time_range_at_tstep(self, tstep, remask=False, inclusive=True):
        """The range of times for the given time step.

        Input:
            tstep       a Pair of time step index values.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        tstep = Scalar.as_scalar(tstep, recursive=False)
        tstep_int = tstep.int(top=self.steps, remask=remask, clip=True,
                              inclusive=inclusive)

        time_min = Scalar(self.tlist[tstep_int.vals], tstep_int.mask)

        return (time_min, time_min + self.texp[tstep_int.vals])

    #===========================================================================
    def tstep_at_time(self, time, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            derivs      True to include derivatives of time in the returned
                        tstep.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Scalar of time step indices.
        """

        time = Scalar.as_scalar(time, recursive=derivs)

        # np.interp converts each time to a float whose integer part is the
        # index of the time step at or below this time. Times outside the valid
        # range get mapped to the nearest valid index. As a result, any time
        # before the start time gets mapped to 0 and any time during or after
        # the last time step returns the last index, self.steps-1.
        #
        # Note that, if the Sequence integration times overlap and therefore
        # tstep_at_time does not have a unique solution, this will return the
        # last tstep that contains the time, which is probably what we want.

        interp = np.interp(time.vals, self.tlist.vals, self._interp_y)
        tstep_int = interp.astype('int')

        # tstep_frac is 0 at the beginning of each integration and 1 and the
        # end. It is negative before the first time step and > 1 after the end
        # of the last. We clip it (0 inclusive,1 exclusive) before adding it
        # back to the integer part.

        tstep_frac_unclipped = ((time - self.tlist[tstep_int])
                                / self.texp[tstep_int])
        tstep_frac_clipped = tstep_frac_unclipped.clip(0, 1, remask=remask,
                                                             inclusive=False)

        tstep = tstep_int + tstep_frac_clipped

        # The end time might require special handling, because it should be
        # unmasked if inclusive=True, whereas the end times of intermediate
        # time steps are not included.

        if inclusive:
            mask = Boolean.as_boolean(time == self.time[1])
            if mask.any():
                tstep[mask] = (Scalar.as_scalar(tstep_int)[mask]
                               + tstep_frac_unclipped[mask])

        return tstep

    #===========================================================================
    def tstep_range_at_time(self, time, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it. If this is cadence is not
                        continuous, this also defines whether the end moment of
                        each individual interval is included in that interval.

        Return:         (tstep_min, tstep_max)
            tstep_min   minimum Scalar time step containing the given time.
            tstep_max   maximum Scalar time step after the given time.

        Returned tstep_min will always be in the allowed range for the cadence,
        inclusive, regardless of masking. If the time is not inside the cadence,
        tstep_max == tstep_min.
        """

        if not self._tstop_is_ordered:
            raise RuntimeError('tstep_range_at_time failure in Sequence; '
                               'stop times are not strictly ordered')

        time = Scalar.as_scalar(time, recursive=False)

        # Locate the first stop time before and the last start time after
        tstep0 = np.interp(time.vals, self._tstop.vals, self._interp_y)
        tstep_min = Scalar(tstep0.astype('int'))        # last stop <= time

        temp_mask = (time.vals >= self._tstop[0]) & (time.vals < self.time[1])
        tstep_min[temp_mask] += 1                       # first stop > time

        tstep1 = np.interp(time.vals, self.tlist.vals, self._interp_y)
        tstep_max = Scalar(tstep1.astype('int')) + 1    # last start <= time + 1

        # Identify points outside the range for adjustment and masking
        # For all points outside range, tstep_max == tstep_min.
        # This also applies to times between time steps for discontinuous
        # cadences.
        if inclusive:
            mask = (time.vals < self.time[0]) | (time.vals > self.time[1])
            if not self.is_continuous:
                k = (tstep1.astype('int') if isinstance(tstep1, np.ndarray)
                                          else int(tstep1))
                mask |= ((time.vals - self.tlist.vals[k] >= self.texp.vals[k])
                         & (time.vals < self.time[1]))
        else:
            mask = (time.vals < self.time[0]) | (time.vals >= self.time[1])
            if not self.is_continuous:
                k = (tstep1.astype('int') if isinstance(tstep1, np.ndarray)
                                          else int(tstep1))
                mask |= (time.vals - self.tlist.vals[k] >= self.texp.vals[k])

        tstep_max[mask] = tstep_min[mask]

        # Update the mask
        if remask:
            if np.any(mask):
                new_mask = Qube.or_(time.mask, mask)
            else:
                new_mask = time.mask

            tstep_min = tstep_min.remask(new_mask)
            tstep_max = tstep_max.remask(new_mask)

        else:
            tstep_min = tstep_min.remask(time.mask)
            tstep_max = tstep_max.remask(time.mask)

        return (tstep_min, tstep_max)

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Boolean array indicating which time values are not
                        sampled by the cadence.
        """

        if self.is_continuous:
            return Cadence.time_is_outside(self, time, inclusive=inclusive)

        # See tstep_at_time above for explanation...
        time = Scalar.as_scalar(time, recursive=False)
        interp = np.interp(time.vals, self.tlist.vals, self._interp_y)

        # Convert to int, carefully...
        if np.isscalar(interp):
            tstep_int = int(interp)
        else:
            tstep_int = interp.astype('int')

        # Compare times, using TVL comparisons to retain the mask on time_diff
        time_diff = time - self.tlist.vals[tstep_int]
        if inclusive:
            is_outside = (time_diff.tvl_lt(0.) |
                          time_diff.tvl_gt(self.texp[tstep_int]))
        else:
            is_outside = (time_diff.tvl_lt(0.) |
                          time_diff.tvl_ge(self.texp[tstep_int]))

        return is_outside

    #===========================================================================
    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given
        amount.

        Input:
            secs        the number of seconds to shift the time later.
        """

        return Sequence(self.tlist + secs, self.texp)

    #===========================================================================
    def as_continuous(self):
        """A shallow copy of this cadence, forced to be continuous.

        For Sequence, this is accomplished by forcing the exposure times to be
        greater than or equal to the stride for each step.
        """

        if self.is_continuous:
            return self

        texp = np.empty(self.tlist.shape)
        texp[:-1] = np.maximum(self.texp.vals[:-1], np.diff(self.tlist.vals))
        texp[ -1] = self.texp[-1].vals

        result = Sequence(self.tlist, texp)
        result.is_continuous = True  # forced, in case of roundoff error in texp
        return result

################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.cadence.metronome import (case_continuous, case_discontinuous,
                                    case_non_unique, case_partial_overlap)

class Test_Sequence(unittest.TestCase):

    def runTest(self):

        import numpy.random as random

        random.seed(5995)

        # These are the tests for subclass Metronome. We define Sequences so
        # that behavior should be identical, except in the out-of-bound cases

        ############################################
        # Tests for continuous case
        # 100-110, 110-120, 120-130, 130-140
        ############################################

        # cadence = Metronome(100., 10., 10., 4)
        cadence = Sequence([100.,110.,120.,130.,140.], 0.)
        case_continuous(self, cadence)

        ############################################
        # Discontinuous case
        # 100-107.5, 110-117.5, 120-127.5, 130-137.5
        ############################################

        # cadence = Metronome(100., 10., 7.5, 4)
        cadence = Sequence([100.,110.,120.,130.], 7.5)
        case_discontinuous(self, cadence)

        ############################################
        # Non-unique case
        # 100-140, 110-150, 120-160, 130-170
        ############################################

        # cadence = Metronome(100., 10., 40., 4)
        cadence = Sequence([100.,110.,120.,130.], 40.)
        case_non_unique(self, cadence)

        ############################################
        # Partial overlap case
        # 100-140, 130-170, 160-200, 190-230
        ############################################

        # cadence = Metronome(100., 30., 40., 4)
        cadence = Sequence([100.,130.,160.,190.], 40.)
        case_partial_overlap(self, cadence)

        ############################################
        # Other cases
        ############################################

        cadence = Sequence([100.,110.,120.,130.], [10.,10.,5.,10.])
        self.assertFalse(cadence.is_continuous)
        cadence = Sequence([100.,110.,125.,130.], [10.,15.,5.,10.])
        self.assertTrue(cadence.is_continuous)

        self.assertEqual(cadence.tstep_at_time(105., remask=True), 0.5)
        self.assertEqual(cadence.tstep_at_time(115., remask=True), 4./3.)
        self.assertEqual(cadence.tstep_at_time(127., remask=True), 2.4)
        self.assertEqual(cadence.time_at_tstep(0.5 , remask=True), 105.)
        self.assertEqual(cadence.time_at_tstep(4./3., remask=True), 115.)
        self.assertEqual(cadence.time_at_tstep(2.4 , remask=True), 127.)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
