################################################################################
# oops/cadence/metronome.py: Metronome subclass of class Cadence
################################################################################

from polymath import Scalar

from . import Cadence

class Metronome(Cadence):
    """A Cadence subclass where time steps occur at uniform intervals."""

    #===========================================================================
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

    def __getstate__(self):
        return (self.tstart, self.tstride, self.texp, self.steps)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

        Input:
            tstep       a Scalar of time step index values.
            remask      True to mask values outside the time limits.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the maximum index of the cadence as inside
                        the cadence; False to treat it as outside.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep, recursive=derivs)

        if self._gapless:
            time = self.time[0] + self.tstride * tstep
            if remask:
                if inclusive:
                    mask = (tstep.vals < 0) | (tstep.vals > self.steps)
                else:
                    mask = (tstep.vals < 0) | (tstep.vals >= self.steps)
                time = time.mask_where(mask)

        else:
            tstep_int = tstep.int(top=self.steps,
                                  remask=remask,
                                  inclusive=inclusive,
                                  shift=inclusive)
            tstep_frac = tstep - tstep_int
            time = (self.time[0] + tstep_int * self.tstride
                                 + tstep_frac * self.texp)

        return time

    #===========================================================================
    def time_range_at_tstep(self, tstep, remask=False, inclusive=True,
                                         shift=True):
        """The range of times for the given integer time step.

        Input:
            tstep       a Scalar of time step index values.
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
        tstep_int = tstep.int(top=self.steps, remask=remask,
                              inclusive=inclusive, shift=shift)
        time_min = self.time[0] + tstep_int * self.tstride

        return (time_min, time_min + self.texp)

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
        tstep = (time - self.time[0]) / self.tstride

        if self._gapless:
            if remask:
                mask_endpoints = (False, not inclusive)
                tstep = tstep.mask_where_outside(0, self.steps, remask=True,
                                                 mask_endpoints=mask_endpoints)

        else:
            # When time steps overlap, this returns the latest tstep, which is
            # the one yielding the smallest tstep_frac
            tstep_int = tstep.int(top=self.steps, remask=remask,
                                  inclusive=inclusive, shift=inclusive)
            tstep_frac = (tstep - tstep_int) * self._tscale

            # Don't let an interior fractional part exceed unity, which can
            # happen if there are gaps.
            if inclusive:
                tstep_frac = Scalar.mask_where_gt(tstep_frac, 1., replace=1.,
                                                  remask=remask)
            else:
                tstep_frac = Scalar.mask_where_ge(tstep_frac, 1., replace=1.,
                                                  remask=remask)

            tstep = tstep_int + tstep_frac

        return tstep

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Masked time values return masked results.

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

class Test_Metronome(unittest.TestCase):

    def runTest(self):

        import numpy.random as random
        from polymath import Boolean

        ####################################
        # Continuous case
        # 100-110, 110-120, 120-130, 130-140
        ####################################
        cadence = Metronome(100., 10., 10., 4)
        self.assertTrue(cadence.is_continuous)

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

        tstep = Scalar((0.,1.,2.), [False,True,False])
        self.assertEqual(Boolean(cadence.time_at_tstep(tstep, remask=True).mask),
                         [False,True,False])

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,140])
        self.assertEqual(cadence.time_at_tstep(tstep, remask=True), time)
        self.assertEqual(cadence.time_at_tstep(tstep, remask=False), time)

        tstep = ([-1,0],[2,4],[4.5,5])
        test = cadence.time_at_tstep(tstep, remask=False)
        self.assertEqual(test.masked(), 0)
        test = cadence.time_at_tstep(tstep, remask=True)
        self.assertTrue(Boolean(test.mask) ==
                        [[True,False],[False,False],[True,True]])

        # time_is_inside()
        time  = ([99,100],[120,140],[145,150])
        self.assertTrue(cadence.time_is_inside(time) ==
                         [[False,True],[True,True],[False,False]])
        self.assertTrue(Boolean(cadence.time_is_inside(time, inclusive=False)) ==
                         [[False,True],[True,False],[False,False]])

        time = Scalar((100.,110.,120.), [False,True,False])
        self.assertEqual(Boolean(cadence.time_is_inside(time).mask), time.mask)

        # tstep_at_time()
        self.assertEqual(cadence.tstep_at_time(100., remask=True), 0.)
        self.assertEqual(cadence.tstep_at_time(100., remask=False), 0.)
        self.assertEqual(cadence.tstep_at_time(105., remask=True), 0.5)
        self.assertEqual(cadence.tstep_at_time(105., remask=False), 0.5)
        self.assertEqual(cadence.tstep_at_time(135., remask=True), 3.5)
        self.assertEqual(cadence.tstep_at_time(135., remask=False), 3.5)

        tstep = [100.,105., 108.,109.,110]
        self.assertEqual(cadence.tstep_at_time(tstep, remask=True).count_masked(), 0)
        self.assertEqual(cadence.tstep_at_time(95., remask=False), -0.5) # out of range
        self.assertEqual(cadence.tstep_at_time(145., remask=False), 4.5) # out of range

        time = Scalar((100.,110.,120.), [False,True,False])
        self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=True).mask),
                         time.mask)
        self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=False).mask),
                         time.mask)

        # Conversion and back (and tstride_at_tstep)
        random.seed(0)
        tstep = Scalar(7*random.rand(100,100) - 1.)
        time = cadence.time_at_tstep(tstep, remask=False)
        test = cadence.tstep_at_time(time, remask=False)
        self.assertTrue((abs(tstep - test) < 1.e-14).all())
        self.assertEqual(time.masked(), 0)
        self.assertEqual(test.masked(), 0)

        mask = (tstep < 0) | (tstep > cadence.steps)
        mask1 = (tstep < 0) | (tstep > cadence.steps-1)

        self.assertTrue((abs(cadence.tstride_at_tstep(tstep,
                                                      remask=False) - 10.) <
                         1.e-13).all())
        self.assertEqual(cadence.tstride_at_tstep(tstep,
                                                  remask=False).masked(), 0)
        self.assertTrue((abs(cadence.tstride_at_tstep(tstep) -
                             10.).mvals < 1.e-13).all())
        self.assertTrue(cadence.tstride_at_tstep(tstep,
                                                 remask=True).mask == mask1)

        test = cadence.time_at_tstep(tstep, remask=True)
        self.assertTrue((abs(time - test).mvals < 1.e-14).all())
        self.assertTrue(test.mask == mask)
        self.assertTrue(cadence.time_is_inside(time) == ~mask)

        time = Scalar(70*random.rand(100,100) + 90.)
        tstep = cadence.tstep_at_time(time, remask=False)
        test = cadence.time_at_tstep(tstep, remask=False)
        self.assertTrue((abs(time - test) < 1.e-14).all())
        self.assertEqual(tstep.masked(), 0)
        self.assertEqual(test.masked(), 0)

        mask2 = (time < 100.) | (time > 140.)
        test = cadence.tstep_at_time(time, remask=True)
        self.assertTrue((abs(tstep - test).mvals < 1.e-14).all())
        self.assertTrue(Boolean(test.mask) == mask2)
        self.assertTrue(cadence.time_is_inside(time) == ~mask2)

        # time_range_at_tstep()
        tstep = Scalar((0.,1.,2.), [False,True,False])
        self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep)[0].mask),
                         [False,True,False])
        self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep)[1].mask),
                         [False,True,False])
        tstep = Scalar(7*random.rand(100,100) - 1.)
        tstep = Scalar(7*random.rand(3,3) - 1.)     ############### delete me
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

        ####################################
        # Discontinuous case
        # 100-108, 110-118, 120-128, 130-138
        ####################################

        texp = 8.
        cadence = Metronome(100., 10., texp, 4)
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
        self.assertEqual(cadence.time_at_tstep(-0.5, remask=False), 94.) # out of range
        self.assertEqual(cadence.time_at_tstep(4.5, remask=False), 144.) # out of range

        time = Scalar((100.,110.,120.), [False,True,False])
        self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=True).mask),
                         time.mask)
        self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=False).mask),
                         time.mask)

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,138])
        self.assertEqual(cadence.time_at_tstep(tstep, remask=True), time)
        self.assertEqual(cadence.time_at_tstep(tstep, remask=False), time)

        tstep = ([-1,0],[2,4],[4.5,5])
        test = cadence.time_at_tstep(tstep, remask=False)
        self.assertEqual(test.masked(), 0)
        test = cadence.time_at_tstep(tstep, remask=True)
        self.assertTrue(Boolean(test.mask) ==
                        [[True,False],[False,False],[True,True]])

        # time_is_inside()
        time  = ([99,100],[120,138],[145,150])
        self.assertTrue(cadence.time_is_inside(time) ==
                        [[False,True],[True,True],[False,False]])
        self.assertTrue(cadence.time_is_inside(time, inclusive=False) ==
                        [[False,True],[True,False],[False,False]])

        time = Scalar((100.,110.,120.), [False,True,False])
        self.assertEqual(Boolean(cadence.time_is_inside(time).mask), time.mask)

        # tstep_at_time()
        self.assertEqual(cadence.tstep_at_time(100., remask=True), 0.)
        self.assertEqual(cadence.tstep_at_time(100., remask=False), 0.)
        self.assertEqual(cadence.tstep_at_time(105., remask=True), 0.625)
        self.assertEqual(cadence.tstep_at_time(105., remask=False), 0.625)
        self.assertEqual(cadence.tstep_at_time(135., remask=True), 3.625)
        self.assertEqual(cadence.tstep_at_time(135., remask=False), 3.625)
        self.assertEqual(cadence.tstep_at_time(109., remask=False), 1.) # internal clip

        self.assertEqual(cadence.tstep_at_time(138., remask=False), 4.) # extrapolations
        self.assertEqual(cadence.tstep_at_time(140., remask=False), 4.)
        self.assertEqual(cadence.tstep_at_time(154., remask=False), 5.5)
        self.assertEqual(cadence.tstep_at_time( 90., remask=False), -1.)
        self.assertEqual(cadence.tstep_at_time( 94., remask=False), -0.5)

        time = [100.,105.,108.,109.,110]
        self.assertTrue(Boolean(cadence.tstep_at_time(time, remask=True).mask) ==
                        [False,False,False,True,False])

        time = Scalar((100.,110.,120.), [False,True,False])
        self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=True).mask),
                         time.mask)
        self.assertEqual(Boolean(cadence.tstep_at_time(time, remask=False).mask),
                         time.mask)

        # Conversion and back (and tstride_at_tstep)
        random.seed(0)
        tstep = Scalar(7*random.rand(100,100) - 1.)
        time = cadence.time_at_tstep(tstep, remask=False)
        test = cadence.tstep_at_time(time, remask=False)
        self.assertTrue((abs(tstep - test) < 1.e-14).all())
        self.assertEqual(time.masked(), 0)
        self.assertEqual(test.masked(), 0)

        mask = (tstep < 0) | (tstep > cadence.steps)
        test = cadence.time_at_tstep(tstep, remask=True)
        self.assertTrue((abs(time - test).mvals < 1.e-14).all())
        self.assertTrue(Boolean(test.mask) == mask)
        self.assertTrue(cadence.time_is_inside(time) == ~mask)
        mask1 = (tstep < 0) | (tstep > cadence.steps-1)

        self.assertTrue((abs(cadence.tstride_at_tstep(tstep,
                                                      remask=False) - 10.) <
                         1.e-13).all())
        self.assertEqual(cadence.tstride_at_tstep(tstep,
                                                  remask=False).masked(), 0)
        self.assertTrue((abs(cadence.tstride_at_tstep(tstep, remask=True) -
                             10.).mvals < 1.e-13).all())
        self.assertTrue(cadence.tstride_at_tstep(tstep, remask=True).mask == mask1)

        # We can't recompute "time" for the discontinuous case because not
        # all times are valid

        tstep = cadence.tstep_at_time(time, remask=False)
        test = cadence.time_at_tstep(tstep, remask=False)
        self.assertTrue((abs(time - test) < 1.e-14).all())
        self.assertEqual(tstep.masked(), 0)
        self.assertEqual(test.masked(), 0)

        mask2 = (time < 100.) | (time > 140.)
        test = cadence.tstep_at_time(time, remask=True)
        self.assertTrue(((abs(tstep - test) < 1.e-14) | mask2).all())
        self.assertTrue(Boolean(test.mask) == mask2)
        self.assertTrue(cadence.time_is_inside(time) == ~mask2)

        # time_range_at_tstep()
        tstep = Scalar((0.,1.,2.), [False,True,False])
        self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep,
                                                             remask=True)[0].mask),
                         [False,True,False])
        self.assertEqual(Boolean(cadence.time_range_at_tstep(tstep,
                                                             remask=True)[1].mask),
                         [False,True,False])
        tstep = Scalar(7*random.rand(100,100) - 1.)
        tstep = tstep.int() # time_range_at_tstep requires an int input
        time = cadence.time_at_tstep(tstep, remask=False)
        (time0, time1) = cadence.time_range_at_tstep(tstep, remask=False)
        self.assertEqual(time0, 10*((time0/10).int()))
        self.assertEqual(time1, 10*((time1/10).int())+8)

        self.assertTrue((abs(time1 - time0 - 8.) < 1.e-14).all())

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

        ####################################
        # Converted-to-continuous case
        # We just do spot-checking here
        ####################################

        cadence = cadence.as_continuous()
        self.assertTrue(cadence.is_continuous)

        # time_at_tstep()
        self.assertEqual(cadence.time_at_tstep(0), 100.)
        self.assertEqual(cadence.time_at_tstep(1), 110.)
        self.assertEqual(cadence.time_at_tstep(4), 140.)
        self.assertEqual(cadence.time_at_tstep((3,4)), (130.,140.))

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,140])
        self.assertEqual(cadence.time_at_tstep(tstep), time)

        tstep = ([-1,0],[2,4],[4.5,5])
        test = cadence.time_at_tstep(tstep, remask=True)
        self.assertTrue(Boolean(test.mask) ==
                        [[True,False],[False,False],[True,True]])

        self.assertEqual(cadence.time_at_tstep(0.5), 105.)
        self.assertEqual(cadence.time_at_tstep(3.5), 135.)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
