################################################################################
# oops/cadence/metronome.py: Metronome subclass of class Cadence
################################################################################

from polymath     import Scalar, Qube
from oops.cadence import Cadence

class Metronome(Cadence):
    """A Cadence subclass where time steps occur at uniform intervals."""

    def __init__(self, tstart, tstride, texp, steps, clip=True):
        """Constructor for a Metronome.

        Input:
            tstart      the start time of the observation in seconds TDB.
            tstride     the interval in seconds from the start of one time step
                        to the start of the next.
            texp        the exposure time in seconds associated with each step.
                        This may be shorter than tstride due to readout times,
                        etc. It may also be longer.
            steps       the number of time steps.
            clip        if True (the default), times and index values are always
                        clipped into the valid range.
        """

        self.tstart = float(tstart)
        self.tstride = float(tstride)
        self.texp = float(texp)
        self.steps = int(steps)
        self.clip = bool(clip)

        if self.steps == 1:
            self.tstride = self.texp

        # Required attributes
        self.lasttime = self.tstart + self.tstride * (self.steps - 1)
        self.time = (self.tstart, self.lasttime + self.texp)
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.shape = (self.steps,)
        self.is_continuous = (self.texp >= self.tstride)
        self.is_unique = (self.texp <= self.tstride)
        self.min_tstride = self.tstride
        self.max_tstride = self.tstride

        self._gapless = (self.texp == self.tstride)
        self._tscale = self.tstride / self.texp
        self._tspan = self.texp / self.tstride
        self._tspan1 = self._tspan - 1
        self._max_step = self.steps - 1

    def __getstate__(self):
        return (self.tstart, self.tstride, self.texp, self.steps,
                             self.clip)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values via interpolation.

        Input:
            tstep       a Scalar of time step index values.
            remask      True to mask values outside the time limits.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep, recursive=derivs)

        # One case is especially easy
        if not remask and not self.clip and self._gapless:
            return self.time[0] + self.tstride * tstep

        # Other cases
        tstep_int = tstep.int(top=self.steps, remask=remask,
                              inclusive=inclusive, clip=self.clip)
        tstep_frac = (tstep - tstep_int).clip(0, 1, remask=remask,
                                                    inclusive=False)
            # inclusive is False because the end moments of discontinuous time
            # steps are never included, except for the end of the final time
            # step, which is included when inclusive=True.

        # End moment might require special handling
        if inclusive and (remask or derivs):
            mask = (tstep == self.steps)
            tstep_frac[mask] = tstep[mask] - self._max_step
                # this sets the value to 1 but preserves derivatives

        return (self.time[0] + tstep_int * self.tstride
                             + tstep_frac * self.texp)

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
        tstep_int = tstep.int(top=self.steps, remask=remask,
                              inclusive=inclusive, clip=self.clip)
        time_min = self.time[0] + tstep_int * self.tstride

        return (time_min, time_min + self.texp)

    #===========================================================================
    def tstep_at_time(self, time, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps via interpolation.

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
        tstep = (time - self.time[0]) / self.tstride

        if self._gapless:
            if self.clip:
                tstep = tstep.clip(0, self.steps, remask=remask,
                                   inclusive=inclusive)
            elif remask:
                tstep = tstep.mask_where_outside(0, self.steps, remask=True,
                                                 mask_endpoints=(False,
                                                                 not inclusive))

        elif self.is_unique:
            tstep_int = tstep.int(top=self.steps, remask=remask,
                                  inclusive=inclusive, clip=self.clip)
            tstep_diff = tstep - tstep_int
                # Regardless of self.clip, at the top...
                # If inclusive, tstep_int = self.steps-1 and tstep_diff = texp
                # Otherwise, tstep_int = self.steps and tstep_diff = 0.

            # If self.clip is True, then tstep_diff < 0. before the start time.
            # Otherwise, tstep_diff cannot be negative.
            if self.clip:
                tstep_diff[tstep_diff.vals < 0.] = Scalar(0., remask)

            # Don't let an interior fractional part match or exceed tspan, which
            # happens in the gaps between tsteps. However, if inclusive is True,
            # then the fractional part is allowed to equal tspan at the end
            # time.
            if inclusive:
                mask = ((tstep_diff.vals >= self._tspan)
                        & (time.vals != self.time[1]))
            else:
                mask = (tstep_diff.vals >= self._tspan)

            tstep_diff[mask] = Scalar(self._tspan, remask)

            # Now we can add the integer and fractional parts
            tstep = tstep_int + tstep_diff * self._tscale

        else:
            # Because time steps can overlap, avoid remask for now
            tstep_int = tstep.int(top=self.steps, remask=False,
                                  inclusive=False, clip=False)

            # Handle the last, extended time step
            is_last = Qube.is_inside(time.vals, self.lasttime, self.time[1],
                                     inclusive=inclusive)
            tstep_int[is_last] = self.steps - 1

            # Combine with fractional part
            tstep = tstep_int + (tstep - tstep_int) * self._tscale

            # Clip and remask necessary
            if self.clip:
                tstep = tstep.clip(0, self.steps,
                                   remask=remask, inclusive=inclusive)
            elif remask:
                endpoints = (False, not inclusive)
                tstep = tstep.mask_where_outside(0, self.steps,
                                                 mask_endpoints=endpoints)

        return tstep

    #===========================================================================
    def tstep_range_at_time(self, time, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         (tstep_min, tstep_max)
            tstep_min   minimum Scalar time step containing the given time.
            tstep_max   maximum Scalar time step after the given time.

        Returned tstep_min will always be in the allowed range for the cadence,
        inclusive, regardless of masking. If the time is not inside the cadence,
        tstep_max == tstep_min.
        """

        time = Scalar.as_scalar(time, recursive=False)
        tstep = (time - self.time[0]) / self.tstride

        # Set mask=True here; restore mask later if remask is False
        tstep_min = tstep.int(top=self.steps, remask=True,
                              inclusive=inclusive, clip=True)
        new_mask = tstep_min.mask       # Note: not a copy so modify cautiously

        # For discontinuous or gapless cases...
        if self.is_unique:
            tstep_max = tstep_min + 1

            # Expand mask for discontinuous cadences
            if not self.is_continuous:
                # Determine active time within each time step
                time_frac = (time.vals - self.time[0]
                                       - self.tstride * tstep_min.vals)

                # Mask times when integration is not happening
                if inclusive:       # extra care needed at end time
                    not_integrating = ((time_frac >= self.texp) &
                                       (time.vals != self.time[1]))
                else:
                    not_integrating = (time_frac >= self.texp)

                new_mask = Qube.or_(new_mask, not_integrating)

        else:
            # For overlapping cases...
            tstep_max = tstep_min + 1
            tstep_min = (tstep - self._tspan1).int(top=self.steps, remask=True,
                                                   inclusive=inclusive,
                                                   clip=True)
            # The new mask only applies if _both_ min and max are masked;
            # Otherwise, it is just a time near the beginning or end, and is
            # associated with fewer time steps, not no time steps.
            new_mask = Qube.and_(new_mask, tstep_min.mask)

        # Masked tstep ranges must have zero length
        tstep_max[new_mask] = tstep_min[new_mask]

        # Make sure both endpoints share a common mask
        if remask:
            tstep_min = tstep_min.remask(new_mask)
            tstep_max = tstep_max.remask(new_mask)
        else:
            # Without remasking, revert to the original mask
            tstep_min = tstep_min.remask(time.mask)
            tstep_max = tstep_max.remask(time.mask)

        return (tstep_min, tstep_max)

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Masked time values return masked results.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Boolean array indicating which time values are not
                        sampled by the cadence.
        """

        if self.is_continuous:
            return Cadence.time_is_outside(self, time, inclusive=inclusive)

        time = Scalar.as_scalar(time, recursive=False)
        time_mod = (time - self.time[0]) % self.tstride

        # Use TVL comparison to propagate the mask of time_mod
        if inclusive:
            return (time_mod.tvl_gt(self.texp) | time.tvl_lt(self.time[0])
                                               | time.tvl_gt(self.time[1]))
        else:
            return (time_mod.tvl_gt(self.texp) | time.tvl_lt(self.time[0])
                                               | time.tvl_ge(self.time[1]))

    #===========================================================================
    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given
        amount.

        Input:
            secs        the number of seconds to shift the time later.
        """

        return Metronome(self.tstart + secs,
                         self.tstride, self.texp, self.steps)

    #===========================================================================
    def as_continuous(self):
        """Construct a shallow copy of this Cadence, forced to be continuous.

        For Metronome this is accomplished by forcing the exposure times to
        be equal to the stride.
        """

        return Metronome(self.tstart, self.tstride, self.tstride, self.steps)

    #===========================================================================
    def tstride_at_tstep(self, tstep, sign=1, remask=False):
        """The time interval(s) between the times of adjacent time steps.

        Input:
            tstep       a Scalar or Pair time step index, which need not be
                        integral.
            sign        +1 for the time interval to the next time step;
                        -1 for the time interval since the previous time step.
            remask      True to mask time tsteps that are out of range.

        Return:         a Scalar or Pair of strides in seconds.
        """

        tstep = Scalar.as_scalar(tstep, recursive=False)

        if remask:
            tstep = tstep.clip(0, self.steps, remask=remask)
            if np.any(tstep.mask):
                return Scalar.filled(tstep.shape, self.tstride, mask=tstep.mask)

        if np.shape(tstep.mask):
            return Scalar.filled(tstep.shape, self.tstride, mask=tstep.mask)

        return Scalar(self.tstride)

    #===========================================================================
    @staticmethod
    def for_array1d(steps, tstart, texp, interstep_delay=0.):
        """Alternative constructor.

        Input:
            steps               number of time steps.
            tstart              start time in seconds TDB.
            texp                exposure duration in second for each sample.
            interstep_delay     time delay in seconds between the end of one
                                integration and the beginning of the next, in
                                seconds. Default is 0.
        """

        return Metronome(tstart, texp + interstep_delay, texp, steps)

    #===========================================================================
    @staticmethod
    def for_array0d(tstart, texp):
        """Alternative constructor for a product with no time-axis.

        Input:
            tstart              start time in seconds TDB.
            texp                exposure duration in seconds.
        """

        return Metronome(tstart, texp, texp, 1)

################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy as np
from polymath import Boolean

# Tests are defined here as separate functions so they can also be used for
# testing Sequences that are defined to simulate the behavior of Metronomes.

class Test_Metronome(unittest.TestCase):

  def runTest(self):

    np.random.seed(4182)

    ############################################
    # Tests for continuous case
    # 100-110, 110-120, 120-130, 130-140
    ############################################

    cadence = Metronome(100., 10., 10., 4)
    case_continuous(self, cadence)

    # tstride_at_tstep
    tstep = Scalar(7 * np.random.rand(100) - 1.)
    tstride = cadence.tstride_at_tstep(tstep, remask=False)
    self.assertEqual(tstride, cadence.tstride)

    tstride = cadence.tstride_at_tstep(tstep, remask=True)
    outside = (tstep < 0.) | (tstep > 4.)
    self.assertEqual(tstride[~outside], cadence.tstride)
    self.assertEqual(tstride[outside], Scalar.MASKED)

    # Unclipped Metronome tests
    cadence = Metronome(100., 10., 10., 4, clip=False)
    self.assertEqual(cadence.time_at_tstep(-0.5, remask=False),  95.)
    self.assertEqual(cadence.time_at_tstep( 4.5, remask=False), 145.)

    self.assertEqual(cadence.tstep_at_time( 95., remask=False), -0.5)
    self.assertEqual(cadence.tstep_at_time(145., remask=False),  4.5)

    ############################################
    # Discontinuous case
    # 100-107.5, 110-117.5, 120-127.5, 130-137.5
    ############################################

    cadence = Metronome(100., 10., 7.5, 4)
    case_discontinuous(self, cadence)

    # Unclipped Metronome tests
    cadence = Metronome(100., 10., 8., 4, clip=False)
    self.assertEqual(cadence.time_at_tstep(-0.5, remask=False),  94.)
    self.assertEqual(cadence.time_at_tstep( 4.5, remask=False), 144.)
    self.assertEqual(cadence.time_at_tstep(3.5), 134.)
    self.assertEqual(cadence.time_at_tstep(4), 138.)
    self.assertEqual(cadence.time_at_tstep((3,4)), (130.,138.))

    self.assertEqual(cadence.tstep_at_time(139., remask=False), 4.)
    self.assertEqual(cadence.tstep_at_time(140., remask=False), 4.)
    self.assertEqual(cadence.tstep_at_time(144., remask=False), 4.5)
    self.assertEqual(cadence.tstep_at_time(154., remask=False), 5.5)
    self.assertEqual(cadence.tstep_at_time( 90., remask=False), -1.)
    self.assertEqual(cadence.tstep_at_time( 94., remask=False), -0.5)

    ############################################
    # Non-unique case
    # 100-140, 110-150, 120-160, 130-170
    ############################################

    cadence = Metronome(100., 10., 40., 4)
    case_non_unique(self, cadence)

    # Unclipped Metronome tests
    cadence = Metronome(100., 10., 40., 4, clip=False)
    self.assertEqual(cadence.time_at_tstep(-0.5,remask=False), 110.)
    self.assertEqual(cadence.time_at_tstep( 4.5,remask=False), 160.)
    self.assertEqual(cadence.tstep_at_time(170., inclusive=False), 7.)
    self.assertEqual(cadence.tstep_at_time(171., remask=False), 7.025)
    self.assertEqual(cadence.tstep_range_at_time(170., inclusive=False), (3,3))

    ############################################
    # Partial overlap case
    # 100-140, 130-170, 160-200, 190-230
    ############################################

    cadence = Metronome(100., 30., 40., 4)
    case_partial_overlap(self, cadence)

    # Unclipped Metronome tests
    cadence = Metronome(100., 30., 40., 4, clip=False)
    self.assertEqual(cadence.time_at_tstep(-0.5,remask=False),  90.)
    self.assertEqual(cadence.time_at_tstep( 4.5,remask=False), 240.)
    self.assertEqual(cadence.tstep_at_time(230., inclusive=False), 4.25)
    self.assertEqual(cadence.tstep_at_time(235., remask=False), 4.375)

    ############################################
    # One time step, 100-110
    ############################################

    cadence = Metronome(100., 22., 10., 1)
    one_time_step(self, cadence)

############################################
# Tests for continuous case
# 100-110, 110-120, 120-130, 130-140
############################################

def case_continuous(self, cadence):

    self.assertTrue(cadence.is_continuous)
    self.assertTrue(cadence.is_unique)

    # time_at_tstep()
    self.assertEqual(cadence.time_at_tstep(0, remask=True ), 100.)
    self.assertEqual(cadence.time_at_tstep(0, remask=False), 100.)
    self.assertEqual(cadence.time_at_tstep(1, remask=True ), 110.)
    self.assertEqual(cadence.time_at_tstep(1, remask=False), 110.)
    self.assertEqual(cadence.time_at_tstep(4, remask=True ), 140.)
    self.assertEqual(cadence.time_at_tstep(4, remask=False), 140.)
    self.assertEqual(cadence.time_at_tstep((3,4), remask=True ), (130.,140.))
    self.assertEqual(cadence.time_at_tstep((3,4), remask=False), (130.,140.))
    self.assertEqual(cadence.time_at_tstep(0.5, remask=True ), 105.)
    self.assertEqual(cadence.time_at_tstep(0.5, remask=False), 105.)
    self.assertEqual(cadence.time_at_tstep(3.5, remask=True ), 135.)
    self.assertEqual(cadence.time_at_tstep(3.5, remask=False), 135.)

    tstep = Scalar((0.,1.,2.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=True).mask),
                     [False,True,False])
    self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=False).mask),
                     [False,True,False])

    tstep = ([0,1],[2,3],[3,4])
    time  = ([100,110],[120,130],[130,140])
    self.assertEqual(cadence.time_at_tstep(tstep, remask=True), time)
    self.assertEqual(cadence.time_at_tstep(tstep, remask=False), time)

    tstep = ([-1,0],[2,4],[4.5,5])
    test = cadence.time_at_tstep(tstep, remask=False)
    self.assertEqual(test.masked(), 0)
    self.assertEqual(test, [[100,100],[120,140],[140,140]])

    test = cadence.time_at_tstep(tstep, remask=True)
    self.assertTrue(Boolean(test.mask) ==
                    [[True,False],[False,False],[True,True]])

    test = cadence.time_at_tstep(tstep, remask=True, inclusive=False)
    self.assertTrue(Boolean(test.mask) ==
                    [[True,False],[False,True],[True,True]])

    # time_at_tstep(), derivs
    tstep = Scalar((0.,1.,2.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=True).mask),
                     [False,True,False])
    self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=False).mask),
                     [False,True,False])

    tstep.insert_deriv('t', Scalar((2,3,4), tstep.mask))
    self.assertEqual(cadence.time_at_tstep(tstep, derivs=True).d_dt,
                     (20, Scalar.MASKED, 40))

    tstep = Scalar((0.,1.,2.))
    tstep.insert_deriv('t', Scalar((2,3,4)))
    tstep.insert_deriv('xy', Scalar(np.arange(6).reshape(3,2), drank=1))
    self.assertEqual(cadence.time_at_tstep(tstep, derivs=True).d_dxy,
                     [(0,10), (20,30), (40,50)])

    # time_is_inside()
    time = ([99,100],[120,140],[145,150])
    self.assertTrue(cadence.time_is_inside(time) ==
                     [[False,True],[True,True],[False,False]])
    self.assertTrue(Boolean(cadence.time_is_inside(time, inclusive=False)) ==
                     [[False,True],[True,False],[False,False]])

    time = Scalar((100.,110.,120.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_is_inside(time).mask), time.mask)

    # tstep_at_time()
    self.assertEqual(cadence.tstep_at_time(100., remask=True ), 0.)
    self.assertEqual(cadence.tstep_at_time(100., remask=False), 0.)
    self.assertEqual(cadence.tstep_at_time(105., remask=True ), 0.5)
    self.assertEqual(cadence.tstep_at_time(105., remask=False), 0.5)
    self.assertEqual(cadence.tstep_at_time(135., remask=True ), 3.5)
    self.assertEqual(cadence.tstep_at_time(135., remask=False), 3.5)
    self.assertEqual(cadence.tstep_at_time(140., remask=False), 4.0)
    self.assertEqual(cadence.tstep_at_time(140., remask=True ), 4.0)
    self.assertEqual(cadence.tstep_at_time(140., remask=True,
                                                 inclusive=False), Scalar.MASKED)

    tstep = [100.,105.,108.,109.,110]
    self.assertEqual(cadence.tstep_at_time(tstep, remask=True).count_masked(), 0)

    tstep = [95,100.,105.,110.,140.,145.]
    self.assertFalse(np.any(cadence.tstep_at_time(tstep, remask=False).mask))
    self.assertTrue(np.all(cadence.tstep_at_time(tstep,
                                                 remask=True).mask == (1,0,0,0,0,1)))
    self.assertTrue(np.all(cadence.tstep_at_time(tstep, remask=True,
                                                 inclusive=False).mask == (1,0,0,0,1,1)))
    self.assertTrue(np.all(cadence.tstep_at_time(tstep, remask=False,
                                                 inclusive=True).vals == [0,0,0.5,1,4,4]))

    time = Scalar((100.,110.,120.), [False,True,False])
    self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=True).mask),
                     time.mask)
    self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=False).mask),
                     time.mask)

    # tstep_at_time(), derivs
    time = Scalar((90,100,110,140), derivs={'t': Scalar((100,200,300,400))})
    self.assertEqual(cadence.tstep_at_time(time, remask=False, derivs=True).d_dt,
                     (0, 20, 30, 40))
    self.assertEqual(cadence.tstep_at_time(time, remask=False, derivs=True,
                                                 inclusive=False).d_dt,
                     (0, 20, 30, 0))

    # tstep_range_at_time()
    MASKED_TUPLE = (Scalar.MASKED, Scalar.MASKED)
    self.assertEqual(cadence.tstep_range_at_time(100.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(105.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(110.), (1,2))
    self.assertEqual(cadence.tstep_range_at_time(135.), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(140.), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(140., remask=True), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(140., remask=True,
                                                       inclusive=True), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(140., remask=False,
                                                       inclusive=False), (3,3))
    self.assertEqual(cadence.tstep_range_at_time(140., remask=True,
                                                       inclusive=False), MASKED_TUPLE)
    self.assertEqual(cadence.tstep_range_at_time(140., remask=False,
                                                       inclusive=True), (3,4))

    self.assertEqual(cadence.tstep_range_at_time(140.001), (3,3))
    self.assertEqual(cadence.tstep_range_at_time(140.001, remask=True), MASKED_TUPLE)

    self.assertEqual(cadence.tstep_range_at_time(100.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(100., remask=True), (0,1))

    self.assertEqual(cadence.tstep_range_at_time(99.999, remask=False), (0,0))
    self.assertEqual(cadence.tstep_range_at_time(99.999, remask=True), MASKED_TUPLE)

    tstep = [95.,100.,105.,110.,140.,145.]
    self.assertEqual(cadence.tstep_range_at_time(tstep, remask=False,
                                                 inclusive=True), ([0,0,0,1,3,3],
                                                                   [0,1,1,2,4,3]))
    self.assertEqual(cadence.tstep_range_at_time(tstep, remask=False,
                                                 inclusive=False), ([0,0,0,1,3,3],
                                                                    [0,1,1,2,3,3]))

    # Conversion and back
    tstep = Scalar(7 * np.random.rand(100,100) - 1.)
    time = cadence.time_at_tstep(tstep, remask=False)
    test = cadence.tstep_at_time(time, remask=False)
    mask = (tstep.vals < 0) | (tstep.vals > 4)
    self.assertTrue((abs(tstep - test)[~mask] < 1.e-14).all())
    self.assertTrue(np.all(time[tstep.vals < 0] == 100.))
    self.assertTrue(np.all(time[tstep.vals > 4] == 140.))
    self.assertEqual(test.masked(), 0)

    test = cadence.time_at_tstep(tstep, remask=True)
    self.assertTrue((abs(time - test).mvals < 1.e-14).all())
    self.assertTrue(np.all(test.mask == mask))
    self.assertTrue(cadence.time_is_inside(time).all_true_or_masked())

    time = Scalar(70 * np.random.rand(100,100) + 90.)
    tstep = cadence.tstep_at_time(time, remask=False)
    test = cadence.time_at_tstep(tstep, remask=False)
    mask = (time.vals < 100) | (time.vals > 140)
    self.assertTrue((abs(time - test)[~mask] < 1.e-14).all())
    self.assertEqual(tstep.masked(), 0)
    self.assertEqual(test.masked(), 0)

    # time_range_at_tstep()
    tstep = Scalar((0.,1.,2.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep)[0].mask),
                     [False,True,False])
    self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep)[1].mask),
                     [False,True,False])
    tstep = Scalar(7 * np.random.rand(100,100) - 1.)
    time = cadence.time_at_tstep(tstep, remask=False)
    (time0, time1) = cadence.time_range_at_tstep(tstep, remask=False)
    self.assertEqual(time0, 10*((time0/10).int()))
    self.assertEqual(time1, 10*((time1/10).int()))

    self.assertTrue((abs(time1 - time0 - 10.) < 1.e-14).all())

    mask = (tstep < 0) | (tstep > cadence.steps)
    unmasked = ~mask
    self.assertTrue((time0[unmasked] >= cadence.time[0]).all())
    self.assertTrue((time1[unmasked] >= cadence.time[0]).all())
    self.assertTrue((time0[unmasked] <= cadence.time[1]).all())
    self.assertTrue((time1[unmasked] <= cadence.time[1]).all())
    self.assertTrue((time0[unmasked] <= time[unmasked]).all())
    self.assertTrue((time1[unmasked] >= time[unmasked]).all())

    (time0, time1) = cadence.time_range_at_tstep(tstep, remask=True)
    self.assertTrue(Boolean(time0.mask) == mask)
    self.assertTrue(Boolean(time1.mask) == mask)

    # time_shift()
    shifted = cadence.time_shift(1.)
    time_shifted = shifted.time_at_tstep(tstep, remask=False)

    self.assertTrue((abs(time_shifted-time-1.) < 1.e-13).all())

    # tstride_at_tstep
    tstep = Scalar(7 * np.random.rand(100) - 1.)
    outside = (tstep < 0.) | (tstep > 4.)
    tstep = tstep.remask(50*[False] + 50*[True])

    tstride = cadence.tstride_at_tstep(tstep, remask=False)
    self.assertTrue(not np.any(tstride.mask[:50]))
    self.assertTrue(np.all(tstride.mask[50:]))

    tstride = cadence.tstride_at_tstep(tstep, remask=True)
    self.assertTrue(np.all(tstride.mask[:50] == outside[:50]))
    self.assertTrue(np.all(tstride.mask[50:]))

############################################
# Discontinuous case
# 100-107.5, 110-117.5, 120-127.5, 130-137.5
############################################

def case_discontinuous(self, cadence):

    self.assertFalse(cadence.is_continuous)
    self.assertTrue(cadence.is_unique)

    # time_at_tstep()
    self.assertEqual(cadence.time_at_tstep(0, remask=True ), 100.)
    self.assertEqual(cadence.time_at_tstep(0, remask=False), 100.)
    self.assertEqual(cadence.time_at_tstep(1, remask=True ), 110.)
    self.assertEqual(cadence.time_at_tstep(1, remask=False), 110.)
    self.assertEqual(cadence.time_at_tstep(4, remask=True ), 137.5)
    self.assertEqual(cadence.time_at_tstep(4, remask=False), 137.5)
    self.assertEqual(cadence.time_at_tstep((3,4), remask=True ), (130.,137.5))
    self.assertEqual(cadence.time_at_tstep((3,4), remask=False), (130.,137.5))
    self.assertEqual(cadence.time_at_tstep(0.5, remask=True ), 103.75)
    self.assertEqual(cadence.time_at_tstep(0.5, remask=False), 103.75)
    self.assertEqual(cadence.time_at_tstep(3.5, remask=True ), 133.75)
    self.assertEqual(cadence.time_at_tstep(3.5, remask=False), 133.75)

    tstep = ([0,1],[2,3],[3,4])
    time  = ([100,110],[120,130],[130,137.5])
    self.assertEqual(cadence.time_at_tstep(tstep, remask=True), time)
    self.assertEqual(cadence.time_at_tstep(tstep, remask=False), time)

    tstep = ([-1,0],[2,4],[4.5,5])
    test = cadence.time_at_tstep(tstep, remask=False)
    self.assertEqual(test.masked(), 0)
    test = cadence.time_at_tstep(tstep, remask=True)
    self.assertTrue(Boolean(test.mask) ==
                    [[True,False],[False,False],[True,True]])

    # time_at_tstep(), derivs
    tstep = Scalar((0.,1.,2.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=True).mask),
                     [False,True,False])
    self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=False).mask),
                     [False,True,False])

    tstep.insert_deriv('t', Scalar((2,3,4), tstep.mask))
    self.assertEqual(cadence.time_at_tstep(tstep, derivs=True).d_dt,
                     (15, Scalar.MASKED, 30))

    tstep = Scalar((0.,1.,2.))
    tstep.insert_deriv('t', Scalar((2,3,4)))
    tstep.insert_deriv('xy', Scalar(np.arange(6).reshape(3,2), drank=1))
    self.assertEqual(cadence.time_at_tstep(tstep, derivs=True).d_dxy,
                     [(0,7.5), (15,22.5), (30,37.5)])

    # time_is_inside()
    time  = ([99,100],[120,137.5],[145,150])
    self.assertTrue(cadence.time_is_inside(time) ==
                    [[False,True],[True,True],[False,False]])
    self.assertTrue(cadence.time_is_inside(time, inclusive=False) ==
                    [[False,True],[True,False],[False,False]])

    time = Scalar((100.,110.,120.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_is_inside(time).mask), time.mask)

    # tstep_at_time()
    self.assertEqual(cadence.tstep_at_time(100.  , remask=True ), 0.)
    self.assertEqual(cadence.tstep_at_time(100.  , remask=False), 0.)
    self.assertEqual(cadence.tstep_at_time(103.75, remask=True ), 0.5)
    self.assertEqual(cadence.tstep_at_time(103.75, remask=False), 0.5)
    self.assertEqual(cadence.tstep_at_time(110.  , remask=False), 1.)
    self.assertEqual(cadence.tstep_at_time(107.5 , remask=False), 1.)
    self.assertEqual(cadence.tstep_at_time(107.5 , remask=True, inclusive=False), Scalar.MASKED)
    self.assertEqual(cadence.tstep_at_time(107.5 , remask=True, inclusive=True) , Scalar.MASKED)
    self.assertEqual(cadence.tstep_at_time(107.5 , remask=False), 1.)
    self.assertEqual(cadence.tstep_at_time(107.5 , remask=True) , Scalar.MASKED)
    self.assertEqual(cadence.tstep_at_time(133.75, remask=True ), 3.5)
    self.assertEqual(cadence.tstep_at_time(133.75, remask=False), 3.5)
    self.assertEqual(cadence.tstep_at_time(137.5 , remask=False), 4.)
    self.assertEqual(cadence.tstep_at_time(137.5 , remask=False, inclusive=False), 4.)
    self.assertEqual(cadence.tstep_at_time(137.5 , remask=True , inclusive=True ), 4.)
    self.assertEqual(cadence.tstep_at_time(137.5 , remask=True , inclusive=False), Scalar.MASKED)
    self.assertEqual(cadence.tstep_at_time(138.  , remask=False, inclusive=False), 4.)
    self.assertEqual(cadence.tstep_at_time(138.  , remask=True), Scalar.MASKED)

    time = Scalar((100.,110.,120.), [False,True,False])
    self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=True).mask),
                     time.mask)
    self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=False).mask),
                     time.mask)

    time = [100.,103.75,107.5,109.,110.]
    self.assertTrue(cadence.tstep_at_time(time, remask=False) ==
                    [0., 0.5, 1., 1., 1.])
    self.assertTrue(Boolean(cadence.tstep_at_time(time, remask=True).mask) ==
                    [False,False,True,True,False])

    time = Scalar((100.,110.,120.), [False,True,False])
    self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=True).mask),
                     time.mask)
    self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=False).mask),
                     time.mask)

    # tstep_at_time(), derivs
    time = Scalar((90,100,113.75,137.5,140), derivs={'t': Scalar((15,30,45,60,75))})
    self.assertEqual(cadence.tstep_at_time(time, remask=False, derivs=True).d_dt,
                     (0,4,6,8,0))
    self.assertEqual(cadence.tstep_at_time(time, remask=False, derivs=True,
                                                 inclusive=False).d_dt,
                     (0,4,6,0,0))

    # tstep_range_at_time()
    MASKED_TUPLE = (Scalar.MASKED, Scalar.MASKED)
    self.assertEqual(cadence.tstep_range_at_time(100.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(105.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(135.), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(108.)[0],  # indicates empty range
                     cadence.tstep_range_at_time(108.)[1])
    self.assertEqual(cadence.tstep_range_at_time(108., remask=True), MASKED_TUPLE)
    self.assertEqual(cadence.tstep_range_at_time(107.5)[0],  # indicates empty range
                     cadence.tstep_range_at_time(107.5)[1])
    self.assertEqual(cadence.tstep_range_at_time(107.5, remask=True), MASKED_TUPLE)
    self.assertEqual(cadence.tstep_range_at_time(117.5)[0],  # indicates empty range
                     cadence.tstep_range_at_time(117.5)[1])
    self.assertEqual(cadence.tstep_range_at_time(117.5, remask=True), MASKED_TUPLE)

    self.assertEqual(cadence.tstep_range_at_time(140. ), (3,3))
    self.assertEqual(cadence.tstep_range_at_time(137.5), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(137.5, remask=True), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(137.5, remask=True,
                                                        inclusive=True), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(137.5, remask=False,
                                                        inclusive=False), (3,3))
    self.assertEqual(cadence.tstep_range_at_time(137.5, remask=True,
                                                        inclusive=False), MASKED_TUPLE)
    self.assertEqual(cadence.tstep_range_at_time(137.5, remask=False,
                                                        inclusive=True), (3,4))

    self.assertEqual(cadence.tstep_range_at_time(100.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(100., remask=True), (0,1))

    self.assertEqual(cadence.tstep_range_at_time(99.999, remask=False), (0,0))
    self.assertEqual(cadence.tstep_range_at_time(99.999, remask=True), MASKED_TUPLE)

    # Conversion and back
    tstep = Scalar(7 * np.random.rand(100,100) - 1.)
    time = cadence.time_at_tstep(tstep, remask=False)
    test = cadence.tstep_at_time(time, remask=False)
    mask = (tstep.vals < 0) | (tstep.vals > 4)
    self.assertTrue((abs(tstep - test)[~mask] < 1.e-14).all())
    self.assertTrue(np.all(time[tstep.vals < 0] == 100.))
    self.assertTrue(np.all(time[tstep.vals > 4] == 137.5))
    self.assertEqual(test.masked(), 0)

    mask = (tstep < 0) | (tstep > cadence.steps)
    test = cadence.time_at_tstep(tstep, remask=True)
    self.assertTrue((abs(time - test).mvals < 1.e-14).all())
    self.assertTrue(Boolean(test.mask) == mask)
    self.assertTrue(cadence.time_is_inside(time).all_true_or_masked())

    time = Scalar(70 * np.random.rand(100,100) + 90.)
    tstep = cadence.tstep_at_time(time, remask=True)
    test = cadence.time_at_tstep(tstep, remask=True)
    self.assertTrue((abs(time - test)[~test.mask] < 1.e-13).all())
    self.assertTrue(Boolean(test.mask) == tstep.mask)
    self.assertTrue(cadence.time_is_inside(time[~test.mask]).all())
    self.assertTrue(cadence.time_is_outside(time.vals[test.mask]).all())

    # time_range_at_tstep()
    tstep = Scalar((0.,1.,2.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep,
                                                         remask=True)[0].mask),
                     [False,True,False])
    self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep,
                                                         remask=True)[1].mask),
                     [False,True,False])
    tstep = Scalar(7 * np.random.rand(100,100) - 1.)
    tstep = tstep.int() # time_range_at_tstep requires an int input
    time = cadence.time_at_tstep(tstep, remask=False)
    (time0, time1) = cadence.time_range_at_tstep(tstep, remask=False)
    self.assertEqual(time0, 10*((time0/10).int()))
    self.assertEqual(time1, 10*((time1/10).int())+7.5)

    self.assertTrue((abs(time1 - time0 - 7.5) < 1.e-14).all())

    mask = (tstep < 0) | (tstep > cadence.steps)
    unmasked = ~mask
    self.assertTrue((time0[unmasked] >= cadence.time[0]).all())
    self.assertTrue((time1[unmasked] >= cadence.time[0]).all())
# These are not actually true with Metronome because we're happy to keep
# on computing time beyond the end of the time limits on both ends
#        self.assertTrue((time0[unmasked] <= cadence.time[1]).all())
#        self.assertTrue((time1[unmasked] <= cadence.time[1]).all())
#        self.assertTrue((time0[unmasked] <= time[unmasked]).all())
    self.assertTrue((time1[unmasked] >= time[unmasked]).all())

    (time0, time1) = cadence.time_range_at_tstep(tstep, remask=True)
    self.assertTrue(Boolean(time0.mask) == mask)
    self.assertTrue(Boolean(time1.mask) == mask)

    # time_shift()
    shifted = cadence.time_shift(1.)
    time_shifted = shifted.time_at_tstep(tstep, remask=False)

    self.assertTrue((abs(time_shifted-time-1.) < 1.e-13).all())

    ############################################
    # Converted-to-continuous case
    # We just do spot-checking here
    ############################################

    cadence = cadence.as_continuous()
    self.assertTrue(cadence.is_continuous)

    # time_at_tstep()
    self.assertEqual(cadence.time_at_tstep(0), 100.)
    self.assertEqual(cadence.time_at_tstep(1), 110.)

    tstep = ([0,1],[2,3],[3,3])
    time  = ([100,110],[120,130],[130,130])
    self.assertEqual(cadence.time_at_tstep(tstep), time)

    tstep = ([-1,0],[2,4],[4.5,5])
    test = cadence.time_at_tstep(tstep, remask=True)
    self.assertTrue(Boolean(test.mask) ==
                    [[True,False],[False,False],[True,True]])

    self.assertEqual(cadence.time_at_tstep(0.5), 105.)

############################################
# Non-unique case
# 100-140, 110-150, 120-160, 130-170
############################################

def case_non_unique(self, cadence):

    self.assertTrue(cadence.is_continuous)
    self.assertFalse(cadence.is_unique)

    # time_at_tstep()
    self.assertEqual(cadence.time_at_tstep(0), 100.)
    self.assertEqual(cadence.time_at_tstep(1), 110.)
    self.assertEqual(cadence.time_at_tstep(1.025), 111.)
    self.assertEqual(cadence.time_at_tstep(1.975), 149.)
    self.assertEqual(cadence.time_at_tstep((3,4)), (130.,170.))
    self.assertEqual(cadence.time_at_tstep(3.5,), 150.)

    tstep = ([0,1],[2,3],[3,4])
    time  = ([100,110],[120,130],[130,170])
    self.assertEqual(cadence.time_at_tstep(tstep, remask=True), time)
    self.assertEqual(cadence.time_at_tstep(tstep, remask=False), time)

    tstep = ([-1,0],[2,4],[4.5,5])
    test = cadence.time_at_tstep(tstep, remask=False)
    self.assertEqual(test.masked(), 0)
    test = cadence.time_at_tstep(tstep, remask=True)
    self.assertTrue(Boolean(test.mask) ==
                    [[True,False],[False,False],[True,True]])

    # time_at_tstep(), derivs
    tstep = Scalar((0.,1.,2.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=True).mask),
                     [False,True,False])
    self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=False).mask),
                     [False,True,False])

    tstep.insert_deriv('t', Scalar((2,3,4), tstep.mask))
    self.assertEqual(cadence.time_at_tstep(tstep, derivs=True).d_dt,
                     (80, Scalar.MASKED, 160))

    tstep = Scalar((0.,1.,2.))
    tstep.insert_deriv('t', Scalar((2,3,4)))
    tstep.insert_deriv('xy', Scalar(np.arange(6).reshape(3,2), drank=1))
    self.assertEqual(cadence.time_at_tstep(tstep, derivs=True).d_dxy,
                     [(0,40), (80,120), (160,200)])

    # time_is_inside()
    time  = ([99,100],[150,170],[171,200])
    self.assertTrue(cadence.time_is_inside(time) ==
                    [[False,True],[True,True],[False,False]])
    self.assertTrue(cadence.time_is_inside(time, inclusive=False) ==
                    [[False,True],[True,False],[False,False]])

    # tstep_at_time()
    self.assertEqual(cadence.tstep_at_time(100.), 0.)
    self.assertEqual(cadence.tstep_at_time(105.), 0.125)
    self.assertEqual(cadence.tstep_at_time(110.), 1.)
    self.assertEqual(cadence.tstep_at_time(140.), 3.25)

    self.assertEqual(cadence.tstep_at_time(170., inclusive=True), 4.)
    self.assertEqual(cadence.tstep_at_time(170., remask=True,
                                                 inclusive=False), Scalar.MASKED)
    self.assertEqual(cadence.tstep_at_time(170., remask=False,
                                                 inclusive=False), 4.)
    self.assertEqual(cadence.tstep_at_time(171., remask=True), Scalar.MASKED)

    time = Scalar((100.,110.,120.), [False,True,False])
    self.assertEqual(cadence.tstep_at_time(time), (0., Scalar.MASKED, 2.))

    # tstep_range_at_time()
    MASKED_TUPLE = (Scalar.MASKED, Scalar.MASKED)
    self.assertEqual(cadence.tstep_range_at_time( 99.), (0,0))
    self.assertEqual(cadence.tstep_range_at_time( 99., remask=True), MASKED_TUPLE)
    self.assertEqual(cadence.tstep_range_at_time(100.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(105.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(135.), (0,4))
    self.assertEqual(cadence.tstep_range_at_time(140.), (1,4))
    self.assertEqual(cadence.tstep_range_at_time(159.), (2,4))
    self.assertEqual(cadence.tstep_range_at_time(160.), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(170., inclusive=True ), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(170., inclusive=False), (3,3))
    self.assertEqual(cadence.tstep_range_at_time(170., inclusive=False,
                                                       remask=True), MASKED_TUPLE)

    time = Scalar(90 + 90. * np.random.rand(100))   # 90 to 180
    (tstep_min, tstep_max) = cadence.tstep_range_at_time(time, remask=True)
    self.assertEqual(Boolean(tstep_min.mask), tstep_max.mask)
    outside = (time < 100.) | (time > 170.)
    self.assertEqual(Boolean(tstep_min.mask), outside)

    for t in time:
        tstep_min, tstep_max = cadence.tstep_range_at_time(t)
        for tstep in range(tstep_min.vals, tstep_max.vals):
            time0, time1 = cadence.time_range_at_tstep(tstep)
            self.assertTrue(time0 < t < time1)
        for tstep in range(0, tstep_min.vals):
            time0, time1 = cadence.time_range_at_tstep(tstep)
            self.assertFalse(time0 < t < time1)
        for tstep in range(tstep_max.vals, cadence.steps):
            time0, time1 = cadence.time_range_at_tstep(tstep)
            self.assertFalse(time0 < t < time1)

    # time_range_at_tstep()
    tstep = Scalar((0.,1.,2.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep,
                                                         remask=True)[0].mask),
                     [False,True,False])
    self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep,
                                                         remask=True)[1].mask),
                     [False,True,False])
    tstep = Scalar(7 * np.random.rand(100,100) - 1.)

    time = cadence.time_at_tstep(tstep, remask=True)
    (time0, time1) = cadence.time_range_at_tstep(tstep, remask=False)
    self.assertTrue((time - time0 >= 0.)[~time.mask].all())
    self.assertTrue((time1 - time >= 0.)[~time.mask].all())
    self.assertTrue(cadence.time_is_inside(time[~time.mask]).all())

    mask = (tstep.vals < 0) | (tstep.vals > cadence.steps)
    self.assertTrue(np.all(mask == time.mask))

    unmasked = ~mask
    self.assertTrue((time0[unmasked] >= cadence.time[0]).all())
    self.assertTrue((time1[unmasked] >= cadence.time[0]).all())

    (time0, time1) = cadence.time_range_at_tstep(tstep, remask=True)
    self.assertTrue(Boolean(time0.mask) == mask)
    self.assertTrue(Boolean(time1.mask) == mask)

    # time_shift()
    shifted = cadence.time_shift(1.)
    time_shifted = shifted.time_at_tstep(tstep, remask=False)

    self.assertTrue((abs(time_shifted-time-1.)[~time.mask] < 1.e-13).all())

############################################
# Partial overlap case
# 100-140, 130-170, 160-200, 190-230
############################################

def case_partial_overlap(self, cadence):

    self.assertTrue(cadence.is_continuous)
    self.assertFalse(cadence.is_unique)

    # time_at_tstep()
    self.assertEqual(cadence.time_at_tstep(0), 100.)
    self.assertEqual(cadence.time_at_tstep(1), 130.)
    self.assertEqual(cadence.time_at_tstep(1.025), 131.)
    self.assertEqual(cadence.time_at_tstep(1.975), 169.)
    self.assertEqual(cadence.time_at_tstep((3,4)), (190.,230.))
    self.assertEqual(cadence.time_at_tstep(3.5,), 210.)

    # time_at_tstep(), derivs
    tstep = Scalar((0.,1.,2.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=True).mask),
                     [False,True,False])
    self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=False).mask),
                     [False,True,False])

    tstep.insert_deriv('t', Scalar((2,3,4), tstep.mask))
    self.assertEqual(cadence.time_at_tstep(tstep, derivs=True).d_dt,
                     (80, Scalar.MASKED, 160))

    tstep = Scalar((0.,1.,2.))
    tstep.insert_deriv('t', Scalar((2,3,4)))
    tstep.insert_deriv('xy', Scalar(np.arange(6).reshape(3,2), drank=1))
    self.assertEqual(cadence.time_at_tstep(tstep, derivs=True).d_dxy,
                     [(0,40), (80,120), (160,200)])

    # time_range_at_tstep()
    tstep = Scalar((0.,1.,2.), [False,True,False])
    self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep,
                                                         remask=True)[0].mask),
                     [False,True,False])
    self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep,
                                                         remask=True)[1].mask),
                     [False,True,False])
    tstep = Scalar(7 * np.random.rand(100,100) - 1.)
    time = cadence.time_at_tstep(tstep, remask=True)
    (time0, time1) = cadence.time_range_at_tstep(tstep, remask=False)
    self.assertTrue((time - time0 >= 0.)[~time.mask].all())
    self.assertTrue((time1 - time >= 0.)[~time.mask].all())

    # time_is_inside()
    time  = ([99,100],[150,230],[241,260])
    self.assertTrue(cadence.time_is_inside(time) ==
                    [[False,True],[True,True],[False,False]])
    self.assertTrue(cadence.time_is_inside(time, inclusive=False) ==
                    [[False,True],[True,False],[False,False]])

    # tstep_at_time()
    self.assertEqual(cadence.tstep_at_time(100.), 0.)
    self.assertEqual(cadence.tstep_at_time(110.), 0.25)
    self.assertEqual(cadence.tstep_at_time(135.), 1.125)

    self.assertEqual(cadence.tstep_at_time(230., inclusive=True), 4.)
    self.assertEqual(cadence.tstep_at_time(230., remask=True,
                                                 inclusive=False), Scalar.MASKED)
    self.assertEqual(cadence.tstep_at_time(231., remask=True), Scalar.MASKED)

    time = Scalar((100.,130.,160.), [False,True,False])
    self.assertEqual(cadence.tstep_at_time(time), (0., Scalar.MASKED, 2.))

    # tstep_range_at_time()
    MASKED_TUPLE = (Scalar.MASKED, Scalar.MASKED)
    self.assertEqual(cadence.tstep_range_at_time( 99.), (0,0))
    self.assertEqual(cadence.tstep_range_at_time( 99., remask=True), MASKED_TUPLE)
    self.assertEqual(cadence.tstep_range_at_time(100.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(105.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(136.), (0,2))
    self.assertEqual(cadence.tstep_range_at_time(170.), (2,3))
    self.assertEqual(cadence.tstep_range_at_time(230., inclusive=True ), (3,4))
    self.assertEqual(cadence.tstep_range_at_time(230., inclusive=False), (3,3))
    self.assertEqual(cadence.tstep_range_at_time(230., inclusive=False,
                                                        remask=True), MASKED_TUPLE)

    time = Scalar(90 + (240-90) * np.random.rand(100))  # 90 to 240
    (tstep_min, tstep_max) = cadence.tstep_range_at_time(time, remask=True)
    self.assertEqual(Boolean(tstep_min.mask), tstep_max.mask)
    outside = (time < 100.) | (time > 230.)
    self.assertEqual(Boolean(tstep_min.mask), outside)

    for t in time:
        tstep_min, tstep_max = cadence.tstep_range_at_time(t)
        for tstep in range(tstep_min.vals, tstep_max.vals):
            time0, time1 = cadence.time_range_at_tstep(tstep)
            self.assertTrue(time0 < t < time1)
        for tstep in range(0, tstep_min.vals):
            time0, time1 = cadence.time_range_at_tstep(tstep)
            self.assertFalse(time0 < t < time1)
        for tstep in range(tstep_max.vals, cadence.steps):
            time0, time1 = cadence.time_range_at_tstep(tstep)
            self.assertFalse(time0 < t < time1)

############################################
# One time step, 100-110
############################################

def one_time_step(self, cadence):

    self.assertTrue(cadence.is_continuous)
    self.assertTrue(cadence.is_unique)

    # time_at_tstep()
    self.assertEqual(cadence.time_at_tstep(-0.1), 100.)
    self.assertEqual(cadence.time_at_tstep(-0.1, remask=False), 100.)
    self.assertEqual(cadence.time_at_tstep(-0.1, remask=True ), Scalar.MASKED)
    self.assertEqual(cadence.time_at_tstep( 0  ), 100.)
    self.assertEqual(cadence.time_at_tstep( 0.5), 105.)
    self.assertEqual(cadence.time_at_tstep( 1, remask=False), 110.)
    self.assertEqual(cadence.time_at_tstep( 1, remask=True ), 110.)
    self.assertEqual(cadence.time_at_tstep(1, remask=True, inclusive=False), Scalar.MASKED)

    # time_at_tstep(), derivs
    tstep = Scalar((0., 0.5, 1., 2.))
    tstep.insert_deriv('t', Scalar((2,3,4,5)))

    self.assertEqual(cadence.time_at_tstep(tstep, remask=False), (100,105,110,110))
    self.assertEqual(cadence.time_at_tstep(tstep, remask=True), (100,105,110,Scalar.MASKED))
    self.assertEqual(cadence.time_at_tstep(tstep, derivs=True).d_dt, (20,30,40,0))
    self.assertEqual(cadence.time_at_tstep(tstep, derivs=True, inclusive=False).d_dt, (20,30,0,0))

    # time_is_inside()
    time = ([99,100],[120,140],[145,150])
    self.assertFalse(cadence.time_is_inside(90))
    self.assertTrue (cadence.time_is_inside(100))
    self.assertTrue (cadence.time_is_inside(110))
    self.assertFalse(cadence.time_is_inside(110, inclusive=False))
    self.assertFalse(cadence.time_is_inside(111))

    # tstep_at_time()
    self.assertEqual(cadence.tstep_at_time( 99), 0.)
    self.assertEqual(cadence.tstep_at_time( 99, remask=True), Scalar.MASKED)
    self.assertEqual(cadence.tstep_at_time(100), 0.)
    self.assertEqual(cadence.tstep_at_time(105), 0.5)
    self.assertEqual(cadence.tstep_at_time(110), 1.)
    self.assertEqual(cadence.tstep_at_time(110, remask=True), 1.)
    self.assertEqual(cadence.tstep_at_time(110, remask=True, inclusive=False), Scalar.MASKED)
    self.assertEqual(cadence.tstep_at_time(111), 1.)
    self.assertEqual(cadence.tstep_at_time(111, remask=True), Scalar.MASKED)

    # tstep_at_time(), derivs
    time = Scalar((90,100,110,140), derivs={'t': Scalar((100,200,300,400))})
    self.assertEqual(cadence.tstep_at_time(time, remask=False, derivs=True).d_dt, (0, 20, 30, 0))
    self.assertEqual(cadence.tstep_at_time(time, remask=False, derivs=True,
                                                 inclusive=False).d_dt, (0, 20, 0, 0))

    # tstep_range_at_time()
    MASKED_TUPLE = (Scalar.MASKED, Scalar.MASKED)
    self.assertEqual(cadence.tstep_range_at_time( 99.), (0,0))
    self.assertEqual(cadence.tstep_range_at_time( 99., remask=True), MASKED_TUPLE)
    self.assertEqual(cadence.tstep_range_at_time(100.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(105.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(110.), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(110., remask=True), (0,1))
    self.assertEqual(cadence.tstep_range_at_time(110., remask=True, inclusive=False), MASKED_TUPLE)
    self.assertEqual(cadence.tstep_range_at_time(135., remask=True), MASKED_TUPLE)

    tstep0, tstep1 = cadence.tstep_range_at_time(110., inclusive=False)
    self.assertEqual(tstep0, tstep1)    # indicates zero range

    tstep0, tstep1 = cadence.tstep_range_at_time(135.)
    self.assertEqual(tstep0, tstep1)

    # time_range_at_tstep()
    tstep = Scalar((-1,0,0.5,1,2))
    self.assertEqual(cadence.time_range_at_tstep(tstep)[0], 5*[100])
    self.assertEqual(cadence.time_range_at_tstep(tstep)[1], 5*[110])

    self.assertEqual(cadence.time_range_at_tstep(tstep[0], remask=True), MASKED_TUPLE)
    self.assertEqual(cadence.time_range_at_tstep(tstep[1:4], remask=True)[0], 3*[100])
    self.assertEqual(cadence.time_range_at_tstep(tstep[1:4], remask=True)[1], 3*[110])
    self.assertEqual(cadence.time_range_at_tstep(tstep[4], remask=True), MASKED_TUPLE)

    self.assertEqual(cadence.time_range_at_tstep(tstep[1:3], remask=True, inclusive=False)[0], 2*[100])
    self.assertEqual(cadence.time_range_at_tstep(tstep[1:3], remask=True, inclusive=False)[1], 2*[110])

    self.assertEqual(cadence.time_range_at_tstep(tstep[3], remask=True, inclusive=False), MASKED_TUPLE)

    # tstride_at_tstep
    self.assertEqual(cadence.tstride_at_tstep(0), 10)
    self.assertEqual(cadence.tstride_at_tstep(0.5), 10)
    self.assertEqual(cadence.tstride_at_tstep(1), 10)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
