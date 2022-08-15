################################################################################
# oops/cadence_/metronome.py: Metronome subclass of class Cadence
################################################################################

from polymath import *
from oops.cadence_.cadence import Cadence

#*****************************************************************************
# Metronome
#*****************************************************************************
class Metronome(Cadence):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A Cadence subclass where time steps occur at uniform intervals.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    PACKRAT_ARGS = ['tstart', 'tstride', 'texp', 'steps']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, tstart, tstride, texp, steps):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Metronome.

        Input:
            tstart      the start time of the observation in seconds TDB.
            tstride     the interval in seconds from the start of one time step
                        to the start of the next.
            texp        the exposure time in seconds associated with each step.
                        This may be shorter than tstride due to readout times,
                        etc.
            steps       the number of time steps.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    #===========================================================================



    #===========================================================================
    # time_at_tstep
    #===========================================================================
    def time_at_tstep(self, tstep, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the min time(s) associated with the given time step(s).


        This method supports non-integer step values.

        Input:
            tstep       a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar of times in seconds TDB.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        tstep = Scalar.as_scalar(tstep)

        if self.is_continuous:
            time = self.time[0] + self.tstride * tstep
        else:
            tstep_int = tstep.int()
            time = (self.time[0] + self.tstride * tstep_int
                                 + (tstep - tstep_int) * self.texp)
            # The maximum tstep is still inside the domain
            time = time.mask_where(tstep == self.steps, replace=self.time[1],
                                   remask=False)

        if mask:
            time = time.mask_where((tstep < 0) | (tstep > self.steps))

        return time
    #===========================================================================



    #===========================================================================
    # time_range_at_tstep
    #===========================================================================
    def time_range_at_tstep(self, tstep, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the range of time(s) for the given integer time step(s).

        Input:
            tstep       a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        tstep_int = Scalar.as_scalar(tstep).int()

        time_min = self.time[0] + tstep_int * self.tstride
        time_max = time_min + self.texp

        if mask:
            is_outside = (tstep < 0) | (tstep > self.steps)
            time_min = time_min.mask_where(is_outside)
            time_max = time_max.mask_where(is_outside)

        return (time_min, time_max)
    #===========================================================================



    #===========================================================================
    # tstep_at_time
    #===========================================================================
    def tstep_at_time(self, time, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the time step(s) for given time(s).

        This method supports non-integer time values.

        Input:
            time        a Scalar of times in seconds TDB.
            mask        True to mask time values not sampled within the cadence.

        Return:         a Scalar or Pair of time step indices.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        time = Scalar.as_scalar(time)

        tstep = (time - self.time[0]) / self.tstride

        if not self.is_continuous:
            tstep_int = tstep.int()
            tstep_frac = (tstep - tstep_int) * self.tscale
            tstep_frac = tstep_frac.clip(0,1,False)
            if mask:
                tstep_frac = tstep_frac.mask_where_ge(1.)

            tstep = tstep_int + tstep_frac

        if mask:
            tstep = tstep.mask_where_outside(0, self.steps)

        return tstep
    #===========================================================================



    #===========================================================================
    # time_is_inside
    #===========================================================================
    def time_is_inside(self, time, inclusive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return which time(s) fall inside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to include the end moment of a time interval;
                        False to exclude.

        Return:         a Boolean array indicating which time values are
                        sampled by the cadence. A masked time results in a
                        value of False, not a masked Boolean.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        time = Scalar.as_scalar(time)

        if self.is_continuous:
            return Cadence.time_is_inside(self, time, inclusive=inclusive)
        else:
            time_mod_vals = (time - self.time[0]) % self.tstride

            if inclusive:
                return ((time_mod_vals <= self.texp) &
                        (time >= self.time[0]) &
                        (time <= self.time[1]))
            else:
                return ((time_mod_vals < self.texp) &
                        (time >= self.time[0]) &
                        (time <  self.time[1]))
    #===========================================================================



    #===========================================================================
    # time_shift
    #===========================================================================
    def time_shift(self, secs):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a duplicate with all times shifted by given amount."

        Input:
            secs        the number of seconds to shift the time later.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Metronome(self.tstart + secs,
                         self.tstride, self.texp, self.steps)
    #===========================================================================



    #===========================================================================
    # as_continuous
    #===========================================================================
    def as_continuous(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a shallow copy forced to be continuous.

        For Metronome this is accomplished by forcing the exposure times to
        be equal to the stride.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Metronome(self.tstart, self.tstride, self.tstride, self.steps)
    #===========================================================================


#*****************************************************************************



################################################################################
# UNIT TESTS
################################################################################

import unittest

#*****************************************************************************
# Test_Metronome
#*****************************************************************************
class Test_Metronome(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        import numpy.random as random

        ####################################
        # Continuous case
        # 100-110, 110-120, 120-130, 130-140
        ####################################
        cadence = Metronome(100., 10., 10., 4)
        self.assertTrue(cadence.is_continuous)

        #---------------------
        # time_at_tstep()
        #---------------------
        self.assertEqual(cadence.time_at_tstep(0), 100.)
        self.assertEqual(cadence.time_at_tstep(0, mask=False), 100.)
        self.assertEqual(cadence.time_at_tstep(1), 110.)
        self.assertEqual(cadence.time_at_tstep(1, mask=False), 110.)
        self.assertEqual(cadence.time_at_tstep(4), 140.)
        self.assertEqual(cadence.time_at_tstep(4, mask=False), 140.)
        self.assertEqual(cadence.time_at_tstep((3,4)), (130.,140.))
        self.assertEqual(cadence.time_at_tstep((3,4), mask=False), (130.,140.))
        self.assertEqual(cadence.time_at_tstep(0.5), 105.)
        self.assertEqual(cadence.time_at_tstep(0.5, mask=False), 105.)
        self.assertEqual(cadence.time_at_tstep(3.5), 135.)
        self.assertEqual(cadence.time_at_tstep(3.5, mask=False), 135.)
        self.assertEqual(cadence.time_at_tstep(-0.5, mask=False), 95.) # out of range
        self.assertEqual(cadence.time_at_tstep(4.5, mask=False), 145.) # out of range
        self.assertEqual(Boolean(cadence.time_at_tstep(Scalar((0.,1.,2.),
                                            [False,True,False])).mask),
                         [False,True,False])

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,140])
        self.assertEqual(cadence.time_at_tstep(tstep), time)
        self.assertEqual(cadence.time_at_tstep(tstep, mask=False), time)

        tstep = ([-1,0],[2,4],[4.5,5])
        test = cadence.time_at_tstep(tstep, mask=False)
        self.assertEqual(test.masked(), 0)
        test = cadence.time_at_tstep(tstep)
        self.assertTrue(Boolean(test.mask) ==
                        [[True,False],[False,False],[True,True]])

        #---------------------
        # time_is_inside()
        #---------------------
        time  = ([99,100],[120,140],[145,150])
        self.assertTrue(Boolean(cadence.time_is_inside(time)) ==
                         [[False,True],[True,True],[False,False]])
        self.assertTrue(Boolean(cadence.time_is_inside(time, inclusive=False)) ==
                         [[False,True],[True,False],[False,False]])
        self.assertEqual(cadence.time_is_inside(Scalar((100.,110.,120.),
                                                       [False,True,False])),
                         [True,False,True])

        #---------------------
        # tstep_at_time()
        #---------------------
        self.assertEqual(cadence.tstep_at_time(100.), 0.)
        self.assertEqual(cadence.tstep_at_time(100., mask=False), 0.)
        self.assertEqual(cadence.tstep_at_time(105.), 0.5)
        self.assertEqual(cadence.tstep_at_time(105., mask=False), 0.5)
        self.assertEqual(cadence.tstep_at_time(135.), 3.5)
        self.assertEqual(cadence.tstep_at_time(135., mask=False), 3.5)
        self.assertEqual(cadence.tstep_at_time([100.,105.,
                                                108.,109.,110]).masked(), 0)
        self.assertEqual(cadence.tstep_at_time(95., mask=False), -0.5) # out of range
        self.assertEqual(cadence.tstep_at_time(145., mask=False), 4.5) # out of range
        self.assertEqual(Boolean(cadence.tstep_at_time(Scalar((100.,110.,120.),
                                            [False,True,False])).mask),
                         [False,True,False])

        #-----------------------------------------------
        # Conversion and back (and tstride_at_tstep)
        #-----------------------------------------------
        random.seed(0)
        tstep = Scalar(7*random.rand(100,100) - 1.)
        time = cadence.time_at_tstep(tstep, mask=False)
        test = cadence.tstep_at_time(time, mask=False)
        self.assertTrue((abs(tstep - test) < 1.e-14).all())
        self.assertEqual(time.masked(), 0)
        self.assertEqual(test.masked(), 0)

        mask = (tstep < 0) | (tstep > cadence.steps)
        mask1 = (tstep < 0) | (tstep > cadence.steps-1)

        self.assertTrue((abs(cadence.tstride_at_tstep(tstep, mask=False) - 10.) <
                         1.e-13).all())
        self.assertEqual(cadence.tstride_at_tstep(tstep, mask=False).masked(), 0)
        self.assertTrue((abs(cadence.tstride_at_tstep(tstep) -
                             10.).mvals < 1.e-13).all())
        self.assertTrue(Boolean(cadence.tstride_at_tstep(tstep).mask) == mask1)

        test = cadence.time_at_tstep(tstep)
        self.assertTrue((abs(time - test).mvals < 1.e-14).all())
        self.assertTrue(Boolean(test.mask) == mask)
        self.assertTrue(cadence.time_is_inside(time) == ~mask)

        time = Scalar(70*random.rand(100,100) + 90.)
        tstep = cadence.tstep_at_time(time, mask=False)
        test = cadence.time_at_tstep(tstep, mask=False)
        self.assertTrue((abs(time - test) < 1.e-14).all())
        self.assertEqual(tstep.masked(), 0)
        self.assertEqual(test.masked(), 0)

        mask2 = (time < 100.) | (time > 140.)
        test = cadence.tstep_at_time(time)
        self.assertTrue((abs(tstep - test).mvals < 1.e-14).all())
        self.assertTrue(Boolean(test.mask) == mask2)
        self.assertTrue(cadence.time_is_inside(time) == ~mask2)

        #--------------------------
        # time_range_at_tstep()
        #--------------------------
        self.assertEqual(Boolean(cadence.time_range_at_tstep(Scalar((0.,1.,2.),
                                            [False,True,False]))[0].mask),
                         [False,True,False])
        self.assertEqual(Boolean(cadence.time_range_at_tstep(Scalar((0.,1.,2.),
                                            [False,True,False]))[1].mask),
                         [False,True,False])
        tstep = Scalar(7*random.rand(100,100) - 1.)
        time = cadence.time_at_tstep(tstep, mask=False)
        (time0, time1) = cadence.time_range_at_tstep(tstep, mask=False)
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

        (time0, time1) = cadence.time_range_at_tstep(tstep)
        self.assertTrue(Boolean(time0.mask) == mask)
        self.assertTrue(Boolean(time1.mask) == mask)

        #------------------
        # time_shift()
        #------------------
        shifted = cadence.time_shift(1.)
        time_shifted = shifted.time_at_tstep(tstep, mask=False)

        self.assertTrue((abs(time_shifted-time-1.) < 1.e-13).all())

        ####################################
        # Discontinuous case
        # 100-108, 110-118, 120-128, 130-138
        ####################################

        texp = 8.
        cadence = Metronome(100., 10., texp, 4)
        self.assertFalse(cadence.is_continuous)

        #---------------------
        # time_at_tstep()
        #---------------------
        self.assertEqual(cadence.time_at_tstep(0), 100.)
        self.assertEqual(cadence.time_at_tstep(0, mask=False), 100.)
        self.assertEqual(cadence.time_at_tstep(1), 110.)
        self.assertEqual(cadence.time_at_tstep(1, mask=False), 110.)
        self.assertEqual(cadence.time_at_tstep(4), 138.)
        self.assertEqual(cadence.time_at_tstep(4, mask=False), 138.)
        self.assertEqual(cadence.time_at_tstep((3,4)), (130.,138.))
        self.assertEqual(cadence.time_at_tstep((3,4), mask=False), (130.,138.))
        self.assertEqual(cadence.time_at_tstep(0.5), 104.)
        self.assertEqual(cadence.time_at_tstep(0.5, mask=False), 104.)
        self.assertEqual(cadence.time_at_tstep(3.5), 134.)
        self.assertEqual(cadence.time_at_tstep(3.5, mask=False), 134.)
        self.assertEqual(cadence.time_at_tstep(-0.5, mask=False), 94.) # out of range
        self.assertEqual(cadence.time_at_tstep(4.5, mask=False), 144.) # out of range
        self.assertEqual(Boolean(cadence.tstep_at_time(Scalar((100.,110.,120.),
                                            [False,True,False])).mask),
                         [False,True,False])

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,138])
        self.assertEqual(cadence.time_at_tstep(tstep), time)
        self.assertEqual(cadence.time_at_tstep(tstep, mask=False), time)

        tstep = ([-1,0],[2,4],[4.5,5])
        test = cadence.time_at_tstep(tstep, mask=False)
        self.assertEqual(test.masked(), 0)
        test = cadence.time_at_tstep(tstep, mask=True)
        self.assertTrue(Boolean(test.mask) ==
                        [[True,False],[False,False],[True,True]])

        #--------------------
        # time_is_inside()
        #--------------------
        time  = ([99,100],[120,138],[145,150])
        self.assertTrue(Boolean(cadence.time_is_inside(time)) ==
                        [[False,True],[True,True],[False,False]])
        self.assertTrue(Boolean(cadence.time_is_inside(time, inclusive=False)) ==
                        [[False,True],[True,False],[False,False]])
        self.assertEqual(cadence.time_is_inside(Scalar((100.,110.,120.),
                                                       [False,True,False])),
                         [True,False,True])

        #--------------------
        # tstep_at_time()
        #--------------------
        self.assertEqual(cadence.tstep_at_time(100.), 0.)
        self.assertEqual(cadence.tstep_at_time(100., mask=False), 0.)
        self.assertEqual(cadence.tstep_at_time(105.), 0.625)
        self.assertEqual(cadence.tstep_at_time(105., mask=False), 0.625)
        self.assertEqual(cadence.tstep_at_time(135.), 3.625)
        self.assertEqual(cadence.tstep_at_time(135., mask=False), 3.625)
        self.assertEqual(cadence.tstep_at_time(109., mask=False), 1.) # illegal value clips
        self.assertTrue(Boolean(cadence.tstep_at_time([100.,105.,108.,109.,110],
                                                      mask=True).mask) ==
                        [False,False,True,True,False])
        self.assertEqual(cadence.tstep_at_time(95., mask=False), -0.375) # out of range
        self.assertEqual(cadence.tstep_at_time(145., mask=False), 4.625) # out of range
        self.assertEqual(Boolean(cadence.tstep_at_time(Scalar((100.,110.,120.),
                                            [False,True,False])).mask),
                         [False,True,False])

        #----------------------------------------------
        # Conversion and back (and tstride_at_tstep)
        #----------------------------------------------
        random.seed(0)
        tstep = Scalar(7*random.rand(100,100) - 1.)
        time = cadence.time_at_tstep(tstep, mask=False)
        test = cadence.tstep_at_time(time, mask=False)
        self.assertTrue((abs(tstep - test) < 1.e-14).all())
        self.assertEqual(time.masked(), 0)
        self.assertEqual(test.masked(), 0)

        mask = (tstep < 0) | (tstep > cadence.steps)
        test = cadence.time_at_tstep(tstep)
        self.assertTrue((abs(time - test).mvals < 1.e-14).all())
        self.assertTrue(Boolean(test.mask) == mask)
        self.assertTrue(cadence.time_is_inside(time) == ~mask)
        mask1 = (tstep < 0) | (tstep > cadence.steps-1)

        self.assertTrue((abs(cadence.tstride_at_tstep(tstep, mask=False) - 10.) <
                         1.e-13).all())
        self.assertEqual(cadence.tstride_at_tstep(tstep, mask=False).masked(), 0)
        self.assertTrue((abs(cadence.tstride_at_tstep(tstep) -
                             10.).mvals < 1.e-13).all())
        self.assertTrue(Boolean(cadence.tstride_at_tstep(tstep).mask) == mask1)


        #------------------------------------------------------------------
        # We can't recompute "time" for the discontinuous case because not
        # all times are valid
        #------------------------------------------------------------------
        tstep = cadence.tstep_at_time(time, mask=False)
        test = cadence.time_at_tstep(tstep, mask=False)
        self.assertTrue((abs(time - test) < 1.e-14).all())
        self.assertEqual(tstep.masked(), 0)
        self.assertEqual(test.masked(), 0)

        mask2 = (time < 100.) | (time > 140.)
        test = cadence.tstep_at_time(time)
        self.assertTrue(((abs(tstep - test) < 1.e-14) | mask2).all())
        self.assertTrue(Boolean(test.mask) == mask2)
        self.assertTrue(cadence.time_is_inside(time) == ~mask2)

        #---------------------------
        # time_range_at_tstep()
        #---------------------------
        self.assertEqual(Boolean(cadence.time_range_at_tstep(Scalar((0.,1.,2.),
                                            [False,True,False]))[0].mask),
                         [False,True,False])
        self.assertEqual(Boolean(cadence.time_range_at_tstep(Scalar((0.,1.,2.),
                                            [False,True,False]))[1].mask),
                         [False,True,False])
        tstep = Scalar(7*random.rand(100,100) - 1.)
        tstep = tstep.int() # time_range_at_tstep requires an int input
        time = cadence.time_at_tstep(tstep, mask=False)
        (time0, time1) = cadence.time_range_at_tstep(tstep, mask=False)
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

        (time0, time1) = cadence.time_range_at_tstep(tstep)
        self.assertTrue(Boolean(time0.mask) == mask)
        self.assertTrue(Boolean(time1.mask) == mask)

        #------------------
        # time_shift()
        #------------------
        shifted = cadence.time_shift(1.)
        time_shifted = shifted.time_at_tstep(tstep, mask=False)

        self.assertTrue((abs(time_shifted-time-1.) < 1.e-13).all())


        ####################################
        # Converted-to-continuous case
        # We just do spot-checking here
        ####################################
        
        cadence = cadence.as_continuous()
        self.assertTrue(cadence.is_continuous)

        #---------------------
        # time_at_tstep()
        #---------------------
        self.assertEqual(cadence.time_at_tstep(0), 100.)
        self.assertEqual(cadence.time_at_tstep(1), 110.)
        self.assertEqual(cadence.time_at_tstep(4), 140.)
        self.assertEqual(cadence.time_at_tstep((3,4)), (130.,140.))

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,140])
        self.assertEqual(cadence.time_at_tstep(tstep), time)

        tstep = ([-1,0],[2,4],[4.5,5])
        test = cadence.time_at_tstep(tstep)
        self.assertTrue(Boolean(test.mask) ==
                        [[True,False],[False,False],[True,True]])

        self.assertEqual(cadence.time_at_tstep(0.5), 105.)
        self.assertEqual(cadence.time_at_tstep(3.5), 135.)
    #===========================================================================

#*****************************************************************************



########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

