################################################################################
# oops/cadence/sequence.py: Sequence subclass of class Cadence
################################################################################

import numpy as np
from polymath import Qube, Boolean, Scalar, Pair, Vector

from . import Cadence

class Sequence(Cadence):
    """Cadence subclass in which time steps are defined by a list."""

    PACKRAT_ARGS = ['tlist', 'texp']

    #===========================================================================
    def __init__(self, tlist, texp):
        """Constructor for a Sequence.

        Input:
            tlist       a Scalar, list or 1-D array of times in seconds TDB.
            texp        the exposure time in seconds associated with each step.
                        This can be shorter than the time interval due to
                        readout times, etc. This can be:
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

        # Interpret texp
        if np.shape(texp):          # texp is an array
            texp = np.asfarray(texp)
            assert texp.shape == tlist.shape
            assert np.all(texp > 0.)

            self.min_tstride = np.min(tstrides)
            self.max_tstride = np.max(tstrides)
            self.is_continuous = np.all(texp[:-1] >= tstrides)
            self.is_unique = np.all(texp[:-1] <= tstrides)

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

        else:                       # use diffs to define texp
            texp = tstrides
            tlist = tlist[:-1]      # last time is not a time step
            assert np.all(texp > 0.)

            tstrides = tstrides[:-1]
            self.min_tstride = np.min(tstrides)
            self.max_tstride = np.max(tstrides)
            self.is_continuous = True
            self.is_unique = True

        # Convert back to Scalar and save
        # as_readonly() ensures that these inputs cannot be modified by
        # something external to the object.
        self.tlist = Scalar(tlist).as_readonly()
        self.texp = Scalar(texp).as_readonly()

        self._steps = self.tlist.size
        self._max_tstep = self._steps - 1

        # Used for the inverse conversion
        self._interp_y = np.arange(self._steps, dtype='float')
        self._is_gapless = self.is_continuous and self.is_unique

        # Fill in required attributes
        self.lasttime = self.tlist.vals[-1]
        self.time = (self.tlist.vals[0],
                     self.tlist.vals[-1] + self.texp.vals[-1])
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.shape = self.tlist.shape

        return

    #===========================================================================
    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

        Input:
            tstep       a Pair of time step index values.
            remask      True to mask values outside the time limits.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the maximum index of the cadence as inside
                        the cadence; False to treat it as outside.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep, recursive=derivs)
        tstep_int = tstep.int(top=self._steps, remask=remask, clip=True,
                              inclusive=inclusive)
        tstep_frac = tstep - tstep_int

        return (self.tlist[tstep_int.vals] + tstep_frac *
                                             self.texp[tstep_int.vals])

    #===========================================================================
    def time_range_at_tstep(self, tstep, remask=False, inclusive=True,
                                         shift=True):
        """The range of times for the given time step.

        Input:
            tstep       a Pair of time step index values.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the maximum index of the cadence as inside
                        the cadence; False to treat it as outside.
            shift       True to identify the end moment of the cadence as being
                        part of the last time step.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        tstep = Scalar.as_scalar(tstep, recursive=False)
        tstep_int = tstep.int(top=self._steps, remask=remask, clip=True,
                              inclusive=inclusive, shift=shift)

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
            inclusive   True to treat the end time of an interval as inside the
                        cadence; False to treat it as outside. The start time of
                        an interval is always treated as inside.

        Return:         a Scalar of time step indices.
        """

        time = Scalar.as_scalar(time, recursive=derivs)

        # np.interp converts each time to a float whose integer part is the
        # index of the time step at or below this time. Times outside the valid
        # range get mapped to the nearest valid index. As a result, any time
        # before the start time gets mapped to 0. and any time during or after
        # the last time step returns the last index, self._steps-1.
        #
        # Note that, if the Sequence integration times overlap and therefore
        # tstep_at_time does not have a unique solution, this will return the
        # last tstep that contains the time, which is probably what we want.

        interp = np.interp(time.vals, self.tlist.vals, self._interp_y)
        tstep_int = interp.astype('int')

        # tstep_frac is 0 at the beginning of each integration and 1 and the
        # end. It is negative before the first time step and > 1 after the end
        # of the last; these cases are harmless can can be addressed with remask
        # if desired.

        tstep_frac = (time - self.tlist[tstep_int]) / self.texp[tstep_int]
        tstep = tstep_int + tstep_frac

        # However, it is also possible for tstep_frac >= 1 during intermediate
        # time steps if there is deadtime between samples. We need to clip these
        # cases, because otherwise (tstep_int + tstep_frac) will appear to
        # refer to a valid, later time step. We do not clip tstep_frac in the
        # final time interval, so after the end time, tstep is just a linear
        # extrapolation of the last time step.

        if not self.is_continuous:
            mask = (tstep_frac.vals >= 1.) & (tstep_int < self._max_tstep)
            tstep = tstep.mask_where(mask, replace=(tstep_int + 1.),
                                           remask=remask)

        if remask:
            if inclusive:
                tstep = tstep.mask_where((time < self.time[0]) |
                                         (time > self.time[1]))
            else:
                tstep = tstep.mask_where((time < self.time[0]) |
                                         (time >= self.time[1]))

        return tstep

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of an interval as inside the
                        cadence; False to treat it as outside. The start time of
                        an interval is always treated as inside.

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

class Test_Sequence(unittest.TestCase):

    def runTest(self):

        import numpy.random as random

        # These are the tests for subclass Metronome. We define the Sequence so
        # that behavior should be identical, except in the out-of-bound cases

        ####################################
        # Continuous case
        # cadence = Metronome(100., 10., 10., 4)
        # 100-110, 110-120, 120-130, 130-140
        ####################################

        cadence = Sequence([100.,110.,120.,130.,140.], 0.)
        self.assertTrue(cadence.is_continuous)
        self.assertTrue(cadence.is_unique)

        # time_at_tstep()
        self.assertEqual(cadence.time_at_tstep(0, remask=True), 100.)
        self.assertEqual(cadence.time_at_tstep(0, remask=False), 100.)
        self.assertEqual(cadence.time_at_tstep(1, remask=True), 110.)
        self.assertEqual(cadence.time_at_tstep(1, remask=False), 110.)
        self.assertEqual(cadence.time_at_tstep(4, remask=True), 140.)
        self.assertEqual(cadence.time_at_tstep(4, remask=False), 140.)
        self.assertEqual(cadence.time_at_tstep((3,4), remask=True), (130.,140.))
        self.assertEqual(cadence.time_at_tstep((3,4), remask=False), (130.,140.))
        self.assertEqual(cadence.time_at_tstep(0.5, remask=True), 105.)
        self.assertEqual(cadence.time_at_tstep(0.5, remask=False), 105.)
        self.assertEqual(cadence.time_at_tstep(3.5, remask=True), 135.)
        self.assertEqual(cadence.time_at_tstep(3.5, remask=False), 135.)
        self.assertEqual(cadence.time_at_tstep(-0.5, remask=False), 95.) # out of range
        self.assertEqual(cadence.time_at_tstep(4.5, remask=False), 145.) # out of range
        self.assertEqual(Boolean(cadence.tstep_at_time(Scalar((100.,110.,120.),
                                            [False,True,False])).mask),
                         [False,True,False])

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,140])
        self.assertEqual(cadence.time_at_tstep(tstep, remask=True), time)
        self.assertEqual(cadence.time_at_tstep(tstep, remask=False), time)

        tstep = ([-1,0],[2,4],[4.5,5])
        test = cadence.time_at_tstep(tstep, remask=False)
        self.assertEqual(test.count_masked(), 0)
        test = cadence.time_at_tstep(tstep, remask=True)
        self.assertTrue(Boolean(test.mask) ==
                        [[True,False],[False,False],[True,True]])

        # time_is_inside()
        time  = ([99,100],[120,140],[145,150])
        self.assertTrue(Boolean(cadence.time_is_inside(time)) ==
                        [[False,True],[True,True],[False,False]])
        self.assertTrue(Boolean(cadence.time_is_inside(time, inclusive=False)) ==
                        [[False,True],[True,False],[False,False]])

        time = Scalar((100.,110.,120.), [False,True,False])
        self.assertEqual(Boolean(cadence.time_is_inside(Scalar(time)).mask),
                         time.mask)

     # tstep_at_time()
        self.assertEqual(cadence.tstep_at_time(100., remask=True), 0.)
        self.assertEqual(cadence.tstep_at_time(100., remask=False), 0.)
        self.assertEqual(cadence.tstep_at_time(105., remask=True), 0.5)
        self.assertEqual(cadence.tstep_at_time(105., remask=False), 0.5)
        self.assertEqual(cadence.tstep_at_time(135., remask=True), 3.5)
        self.assertEqual(cadence.tstep_at_time(135., remask=False), 3.5)
        self.assertEqual(cadence.tstep_at_time([100.,105.,108.,109.,110],
                                               remask=True).count_masked(), 0)
        # out of range...
        self.assertTrue(cadence.tstep_at_time(95., remask=True).mask)
        self.assertFalse(cadence.tstep_at_time(95., remask=False).mask)
        self.assertTrue(cadence.tstep_at_time(145., remask=True).mask)
        self.assertFalse(cadence.tstep_at_time(145., remask=False).mask)

        # Conversion and back (and tstride_at_tstep)
#         random.seed(0)
        tstep = Scalar(4*random.rand(100,100))
        time = cadence.time_at_tstep(tstep, remask=False)
        test = cadence.tstep_at_time(time, remask=False)
        self.assertTrue((abs(tstep - test) < 1.e-14).all())
        self.assertEqual(time.count_masked(), 0)
        self.assertEqual(test.count_masked(), 0)

        mask  = (tstep < 0) | (tstep > cadence._steps)
        mask1 = (tstep < 0) | (tstep > cadence._steps-1)

        self.assertTrue((abs(cadence.tstride_at_tstep(tstep,
                                                      remask=False) - 10.) <
                         1.e-13).all())
        self.assertEqual(cadence.tstride_at_tstep(tstep,
                                                  remask=False).count_masked(), 0)
        self.assertTrue((abs(cadence.tstride_at_tstep(tstep, remask=True) -
                             10.).mvals < 1.e-13).all())
        self.assertTrue(Boolean(cadence.tstride_at_tstep(tstep,
                                                         remask=True).mask) ==
                        mask1)

        test = cadence.time_at_tstep(tstep, remask=True)
        self.assertTrue((abs(time - test).mvals < 1.e-14).all())
        self.assertTrue(Boolean(test.mask) == mask)
        self.assertTrue(cadence.time_is_inside(time) == ~mask)

        time = Scalar(40*random.rand(100,100) + 100.)
        tstep = cadence.tstep_at_time(time, remask=False)
        test = cadence.time_at_tstep(tstep, remask=False)
        self.assertTrue((abs(time - test) < 1.e-14).all())
        self.assertEqual(tstep.count_masked(), 0)
        self.assertEqual(test.count_masked(), 0)

        mask2 = (time < 100.) | (time > 140.)
        test = cadence.tstep_at_time(time, remask=True)
        self.assertTrue((abs(tstep - test).mvals < 1.e-14).all())
        self.assertTrue(Boolean(test.mask) == mask2)
        self.assertTrue(cadence.time_is_inside(time) == ~mask2)

        # time_range_at_tstep()
        self.assertEqual(Boolean(cadence.time_range_at_tstep(
                            Scalar((0.,1.,2.), [False,True,False]),
                            remask=True)[0].mask),
                         [False,True,False])
        self.assertEqual(Boolean(cadence.time_range_at_tstep(
                            Scalar((0.,1.,2.), [False,True,False]),
                            remask=True)[1].mask),
                         [False,True,False])
        tstep = Scalar(7*random.rand(100,100) - 1.)
        tstep = tstep.int() # time_range_at_tstep requires an int input
        time = cadence.time_at_tstep(tstep, remask=False)
        (time0, time1) = cadence.time_range_at_tstep(tstep, remask=False)
        self.assertEqual(time0, 10*((time0/10).int()))
        self.assertEqual(time1, 10*((time1/10).int()))

        self.assertTrue((abs(time1 - time0 - 10.) < 1.e-14).all())

        mask = (tstep < 0) | (tstep > cadence._steps)
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

        ####################################
        # Discontinuous case
        # texp = 8.
        # cadence = Metronome(100., 10., texp, 4)
        # 100-108, 110-118, 120-128, 130-138
        ####################################

        texp = 8.
        cadence = Sequence([100.,110.,120.,130.], texp)
        self.assertFalse(cadence.is_continuous)

        # time_at_tstep()
        self.assertEqual(cadence.time_at_tstep(0, remask=True), 100.)
        self.assertEqual(cadence.time_at_tstep(0, remask=False), 100.)
        self.assertEqual(cadence.time_at_tstep(1, remask=True), 110.)
        self.assertEqual(cadence.time_at_tstep(1, remask=False), 110.)
        self.assertEqual(cadence.time_at_tstep(4, remask=True), 138.)
        self.assertEqual(cadence.time_at_tstep(4, remask=False), 138.)
        self.assertEqual(cadence.time_at_tstep((3,4), remask=True), (130.,138.))
        self.assertEqual(cadence.time_at_tstep((3,4), remask=False), (130.,138.))
        self.assertEqual(cadence.time_at_tstep(0.5, remask=True), 104.)
        self.assertEqual(cadence.time_at_tstep(0.5, remask=False), 104.)
        self.assertEqual(cadence.time_at_tstep(3.5, remask=True), 134.)
        self.assertEqual(cadence.time_at_tstep(3.5, remask=False), 134.)

        # These cases are different than Metronome because we don't have a
        # regular stride to rely on - the last entry is texp long instead
        # of tstride
        self.assertEqual(cadence.time_at_tstep(-0.5, remask=False), 96.) # out of range
        self.assertEqual(cadence.time_at_tstep(4.5, remask=False), 142.) # out of range
        self.assertEqual(Boolean(cadence.tstep_at_time(Scalar((100.,110.,120.),
                                            [False,True,False])).mask),
                         [False,True,False])

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,138])
        self.assertEqual(cadence.time_at_tstep(tstep, remask=True), time)
        self.assertEqual(cadence.time_at_tstep(tstep, remask=False), time)

        tstep = ([-1,0],[2,4],[4.5,5])
        test = cadence.time_at_tstep(tstep, remask=False)
        self.assertEqual(test.count_masked(), 0)
        test = cadence.time_at_tstep(tstep, remask=True)
        self.assertTrue(Boolean(test.mask) ==
                        [[True,False],[False,False],[True,True]])

        # time_is_inside()
        time  = ([99,100],[120,138],[145,150])
        self.assertTrue(cadence.time_is_inside(time) ==
                        [[False,True],[True,True],[False,False]])
        self.assertTrue(cadence.time_is_inside(time, inclusive=False) ==
                        [[False,True],[True,False],[False,False]])

        time = Scalar((100.,108.,109.,110.,120.,138.))
        self.assertEqual(cadence.time_is_inside(time),
                         [True,True,False,True,True,True])
        self.assertEqual(cadence.time_is_inside(time, inclusive=False),
                         [True,False,False,True,True,False])

        time = Scalar((100.,108.,109.,110.,120.,138.),
                      (True,False,True,False,True,False))
        self.assertEqual(Boolean(cadence.time_is_inside(time).mask), time.mask)

        # tstep_at_time()
        self.assertEqual(cadence.tstep_at_time(100., remask=True), 0.)
        self.assertEqual(cadence.tstep_at_time(100., remask=False), 0.)
        self.assertEqual(cadence.tstep_at_time(105., remask=True), 0.625)
        self.assertEqual(cadence.tstep_at_time(105., remask=False), 0.625)
        self.assertEqual(cadence.tstep_at_time(135., remask=True), 3.625)
        self.assertEqual(cadence.tstep_at_time(135., remask=False), 3.625)

        test = cadence.tstep_at_time(109., remask=False)
        self.assertEqual(cadence.tstep_at_time(109., remask=False), 1.) # clipped value

        time = [100.,105.,108.,109.,110]
        self.assertTrue(Boolean(cadence.tstep_at_time(time, remask=True).mask) ==
                        [False,False,True,True,False])

        self.assertEqual(cadence.tstep_at_time(92., remask=False), -1.) # extrapolations
        self.assertEqual(cadence.tstep_at_time(146, remask=False), 5.)
        self.assertEqual(cadence.tstep_at_time(92., remask=True), Scalar.MASKED)
        self.assertEqual(cadence.tstep_at_time(144, remask=True), Scalar.MASKED)

        time = Scalar((100.,110.,120.), [False,True,False])
        self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=True).mask),
                         time.mask)

        # Conversion and back (and tstride_at_tstep)
        random.seed(0)
        tstep = Scalar(4*random.rand(100,100))
        time = cadence.time_at_tstep(tstep, remask=False)
        test = cadence.tstep_at_time(time, remask=False)
        self.assertTrue((abs(tstep - test) < 1.e-14).all())
        self.assertEqual(time.count_masked(), 0)
        self.assertEqual(test.count_masked(), 0)

        mask = (tstep < 0) | (tstep > cadence._steps)

        test = cadence.time_at_tstep(tstep, remask=True)
        self.assertTrue((abs(time - test).mvals < 1.e-14).all())
        self.assertTrue(Boolean(test.mask) == mask)
        self.assertTrue(cadence.time_is_inside(time) == ~mask)

        tstep = Scalar(3*random.rand(100,100))
        mask1 = (tstep < 0) | (tstep > cadence._steps-1)
        self.assertTrue((abs(cadence.tstride_at_tstep(tstep,
                                                      remask=False) - 10.) <
                         1.e-13).all())
        self.assertEqual(cadence.tstride_at_tstep(tstep,
                                                remask=False).count_masked(), 0)
        self.assertTrue((abs(cadence.tstride_at_tstep(tstep) -
                             10.).mvals < 1.e-13).all())
        self.assertTrue(Boolean(cadence.tstride_at_tstep(tstep).mask) == mask1)

        # We can't recompute "time" for the discontinuous case because not
        # all times are valid
        tstep = cadence.tstep_at_time(time, remask=False)
        test = cadence.time_at_tstep(tstep, remask=False)
        self.assertTrue((abs(time - test) < 1.e-14).all())
        self.assertEqual(tstep.count_masked(), 0)
        self.assertEqual(test.count_masked(), 0)

        mask2 = (time < 100.) | (time > 140.)
        test = cadence.tstep_at_time(time)
        self.assertTrue(((abs(tstep - test) < 1.e-14) | mask2).all())
        self.assertTrue(Boolean(test.mask) == mask2)
        self.assertTrue(cadence.time_is_inside(time) == ~mask2)

        # time_range_at_tstep()
        self.assertEqual(Boolean(cadence.time_range_at_tstep(Scalar((0.,1.,2.),
                                            [False,True,False]))[0].mask),
                         [False,True,False])
        self.assertEqual(Boolean(cadence.time_range_at_tstep(Scalar((0.,1.,2.),
                                            [False,True,False]))[1].mask),
                         [False,True,False])
        tstep = Scalar(7*random.rand(100,100) - 1.)
        tstep = tstep.int() # time_range_at_tstep requires an int input
        time = cadence.time_at_tstep(tstep, remask=False)
        (time0, time1) = cadence.time_range_at_tstep(tstep, remask=False)
        self.assertEqual(time0, 10*((time0/10).int()))
        self.assertEqual(time1, 10*((time1/10).int())+8)

        self.assertTrue((abs(time1 - time0 - 8.) < 1.e-14).all())

        mask = (tstep < 0) | (tstep > cadence._steps)
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

        ####################################
        # Converted-to-continuous case
        # We just do spot-checking here
        ####################################

        cadence = cadence.as_continuous()
        self.assertTrue(cadence.is_continuous)

        # time_at_tstep()
        self.assertEqual(cadence.time_at_tstep(0, remask=True), 100.)
        self.assertEqual(cadence.time_at_tstep(1, remask=True), 110.)
        self.assertEqual(cadence.time_at_tstep(4, remask=True), 138.)
        self.assertEqual(cadence.time_at_tstep((3,4), remask=True), (130.,138.))

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,138])
        self.assertEqual(cadence.time_at_tstep(tstep, remask=True), time)

        tstep = ([-1,0],[2,4],[4.5,5])
        test = cadence.time_at_tstep(tstep, remask=True)
        self.assertTrue(Boolean(test.mask) ==
                        [[True,False],[False,False],[True,True]])

        self.assertEqual(cadence.time_at_tstep(0.5, remask=True), 105.)
        self.assertEqual(cadence.time_at_tstep(3.5, remask=True), 134.)

        ####################################
        # Other cases
        ####################################

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
