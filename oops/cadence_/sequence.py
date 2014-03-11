################################################################################
# oops/cadence_/sequence.py: Sequence subclass of class Cadence
################################################################################

import numpy as np
from polymath import *
from oops.cadence_.cadence import Cadence

class Sequence(Cadence):
    """Sequence is a Cadence subclass in which time steps are defined by a list."""

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

        self.tlist = Scalar(times)
        assert len(self.tlist.shape) == 1
        texp = Scalar(texp)

        # Used for the inverse conversion; filled in only if needed
        self.padded_indices = None
        self.padded_tlist = None

        if len(texp.shape) > 0:
            self.texp = Scalar(texp)
            assert self.texp.shape == self.tlist.shape
            self.is_continuous = (self.texp[:-1] == np.diff(self.tlist.vals)).all()
        elif texp == 0:
            self.texp = Scalar(np.diff(self.tlist.vals))
            self.tlist = self.tlist[:-1]
            self.is_continuous = True
        else:
            (ignore,self.texp) = Qube.broadcast(self.tlist,texp.as_float())
            self.is_continuous = (np.diff(self.tlist.vals) == texp).all()

        self.steps = self.tlist.size

        self.lasttime = self.tlist[-1]
        self.time = (self.tlist[0], self.tlist[-1] + self.texp[-1])
        self.midtime = (self.time[0] + self.time[1]) * 0.5

        self.shape = (self.steps,)

        return

    def time_at_tstep(self, tstep, mask=True):
        """Return the time(s) associated with the given time step(s).
        
        This method supports non-integer step values.

        Input:
            tstep       a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep)
        tstep_int = tstep.int().clip(0,self.steps-1,False)

        time = ((tstep - tstep_int) * self.texp[tstep_int] +
                self.tlist[tstep_int])     # Scalar + ndarray

        if mask:
            time = time.mask_where((tstep < 0) | (tstep > self.steps))

        return time

    def time_range_at_tstep(self, tstep, mask=True):
        """Return the range of time(s) for the given integer time step(s).

        Input:
            indices     a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        tstep = Scalar.as_int(tstep)
        tstep_clipped = tstep.clip(0,self.steps-1,False)
        time_min = Scalar(self.tlist[tstep_clipped]) # , tstep.mask) XXX
        time_max = time_min + self.texp[tstep_clipped]

        if mask:
            is_outside = (tstep < 0) | (tstep > self.steps)
            time_min = time_min.mask_where(is_outside)
            time_max = time_max.mask_where(is_outside)

        return (time_min, time_max)

    def tstep_at_time(self, time, mask=True):
        """Return the time step(s) for given time(s).

        This method supports non-integer time values.

        Input:
            time        a Scalar of times in seconds TDB.
            mask        True to mask time values not sampled within the cadence.

        Return:         a Scalar or Pair of time step indices.
        """

        # Fill in the internals if they are still empty
        if self.padded_indices is None:
            self.padded_indices = np.arange(self.steps+1)
        if self.padded_tlist is None:
            self.padded_tlist = np.array(list(self.tlist.vals)+
                                         [(self.tlist[-1]+self.texp[-1]).vals])

        time = Scalar.as_scalar(time)
        tstep = Scalar(np.interp(time.vals, self.padded_tlist,
                                 self.padded_indices))
        tstep_int = tstep.int().clip(0,self.steps-1,False)
        tstep_frac = ((time - self.tlist[tstep_int]) /
                      self.texp[tstep_int])
        if mask:
            tstep_frac = tstep_frac.mask_where((tstep_frac >= 1.) &
                                               (time != self.time[1]))
        tstep_frac = tstep_frac.clip(0,1,False)

        tstep = tstep_int + tstep_frac

        if mask:
            tstep = tstep.mask_where((time < self.time[0]) | (time > self.time[1]))

        return tstep

    def time_is_inside(self, time, inclusive=True):
        """Return which time(s) fall inside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to include the end moment of a time interval;
                        False to exclude.

        Return:         a Boolean array indicating which time values are
                        sampled by the cadence. A masked time results in a
                        value of False, not a masked Boolean.
        """

        # Fill in the internals if they are still empty
        if self.padded_indices is None:
            self.padded_indices = np.arange(self.steps+1)
        if self.padded_tlist is None:
            self.padded_tlist = np.array(list(self.tlist.vals)+
                                         [(self.tlist[-1]+self.texp[-1]).vals])

        time = Scalar.as_scalar(time)
        tstep = Scalar(np.interp(time.vals, self.padded_tlist,
                                 self.padded_indices))
        tstep_int = tstep.int().clip(0,self.steps-1,False)
        time_frac = time - self.tlist[tstep_int]

        if inclusive:
            return ((time_frac >= 0) &
                    (time_frac <= self.texp[tstep_int]))
        else:
            return ((time_frac >= 0) &
                    (time_frac <  self.texp[tstep_int]))

    def time_shift(self, secs):
        """Return a duplicate with all times shifted by given amount."

        Input:
            secs        the number of seconds to shift the time later.
        """

        result = Sequence(self.tlist + secs, self.texp)

        result.padded_indices = self.padded_indices
        result.padded_tlist = None

        return result

    def as_continuous(self):
        """Return a shallow copy forced to be continuous.
        
        For Sequence this is accomplished by forcing the exposure times to
        be equal to the stride for each step.
        """

        if self.is_continuous: return self

        texp = np.empty(self.tlist.shape)
        texp[:-1] = np.diff(self.tlist.vals)
        texp[ -1] = self.texp[-1].vals

        result = Sequence(self.tlist, texp)
        result.is_continuous = True
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

        # time_at_tstep()
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
        self.assertEqual(Boolean(cadence.tstep_at_time(Scalar((100.,110.,120.),
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

        # time_is_inside()
        time  = ([99,100],[120,140],[145,150])
        self.assertTrue(Boolean(cadence.time_is_inside(time)) ==
                        [[False,True],[True,True],[False,False]])
        self.assertTrue(Boolean(cadence.time_is_inside(time, inclusive=False)) ==
                        [[False,True],[True,False],[False,False]])
        self.assertEqual(cadence.time_is_inside(Scalar((100.,110.,120.),
                                                       [False,True,False])),
                         [True,False,True])

        # tstep_at_time()
        self.assertEqual(cadence.tstep_at_time(100.), 0.)
        self.assertEqual(cadence.tstep_at_time(100., mask=False), 0.)
        self.assertEqual(cadence.tstep_at_time(105.), 0.5)
        self.assertEqual(cadence.tstep_at_time(105., mask=False), 0.5)
        self.assertEqual(cadence.tstep_at_time(135.), 3.5)
        self.assertEqual(cadence.tstep_at_time(135., mask=False), 3.5)
        self.assertEqual(cadence.tstep_at_time([100.,105.,108.,109.,110],
                                               mask=True).masked(), 0)
        self.assertEqual(cadence.tstep_at_time(95., mask=False), 0.) # out of range
        self.assertEqual(cadence.tstep_at_time(145., mask=False), 4.) # out of range
        self.assertEqual(Boolean(cadence.tstep_at_time(Scalar((100.,110.,120.),
                                            [False,True,False])).mask),
                         [False,True,False])

        # Conversion and back (and tstride_at_tstep)
        random.seed(0)
        tstep = Scalar(4*random.rand(100,100))
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

        time = Scalar(40*random.rand(100,100) + 100.)
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
        
        # time_range_at_tstep()
#        self.assertEqual(Boolean(cadence.time_range_at_tstep(Scalar((0.,1.,2.),
#                                            [False,True,False]))[0].mask),
#                         [False,True,False])
#        self.assertEqual(Boolean(cadence.time_range_at_tstep(Scalar((0.,1.,2.),
#                                            [False,True,False]))[1].mask),
#                         [False,True,False])
        tstep = Scalar(7*random.rand(100,100) - 1.)
        tstep = tstep.int() # time_range_at_tstep requires an int input
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
        
        # time_shift()
        shifted = cadence.time_shift(1.)
        time_shifted = shifted.time_at_tstep(tstep, mask=False)

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
        # These cases are different than Metronome because we don't have a
        # regular stride to rely on - the last entry is texp long instead
        # of tstride
        self.assertEqual(cadence.time_at_tstep(-0.5, mask=False), 96.) # out of range
        self.assertEqual(cadence.time_at_tstep(4.5, mask=False), 142.) # out of range
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

        # time_is_inside()
        time  = ([99,100],[120,138],[145,150])
        self.assertTrue(Boolean(cadence.time_is_inside(time)) ==
                        [[False,True],[True,True],[False,False]])
        self.assertTrue(Boolean(cadence.time_is_inside(time, inclusive=False)) ==
                        [[False,True],[True,False],[False,False]])
        self.assertEqual(cadence.time_is_inside(Scalar((100.,110.,120.),
                                                       [False,True,False])),
                         [True,False,True])

        # tstep_at_time()
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
        self.assertEqual(cadence.tstep_at_time(95., mask=False), 0.) # out of range
        self.assertEqual(cadence.tstep_at_time(145., mask=False), 4.) # out of range
        self.assertEqual(Boolean(cadence.tstep_at_time(Scalar((100.,110.,120.),
                                            [False,True,False])).mask),
                         [False,True,False])

        # Conversion and back (and tstride_at_tstep)
        random.seed(0)
        tstep = Scalar(4*random.rand(100,100))
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

        tstep = Scalar(3*random.rand(100,100))
        mask1 = (tstep < 0) | (tstep > cadence.steps-1)
        self.assertTrue((abs(cadence.tstride_at_tstep(tstep, mask=False) - 10.) <
                         1.e-13).all())
        self.assertEqual(cadence.tstride_at_tstep(tstep, mask=False).masked(), 0)
        self.assertTrue((abs(cadence.tstride_at_tstep(tstep) -
                             10.).mvals < 1.e-13).all())
        self.assertTrue(Boolean(cadence.tstride_at_tstep(tstep).mask) == mask1)
        
        # We can't recompute "time" for the discontinuous case because not
        # all times are valid
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

        # time_range_at_tstep()
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
        self.assertTrue((time0[unmasked] <= cadence.time[1]).all())
        self.assertTrue((time1[unmasked] <= cadence.time[1]).all())
        self.assertTrue((time0[unmasked] <= time[unmasked]).all())
        self.assertTrue((time1[unmasked] >= time[unmasked]).all())

        (time0, time1) = cadence.time_range_at_tstep(tstep)
        self.assertTrue(Boolean(time0.mask) == mask)
        self.assertTrue(Boolean(time1.mask) == mask)
        
        # time_shift()
        shifted = cadence.time_shift(1.)
        time_shifted = shifted.time_at_tstep(tstep, mask=False)

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
        self.assertEqual(cadence.time_at_tstep(4), 138.)
        self.assertEqual(cadence.time_at_tstep((3,4)), (130.,138.))

        tstep = ([0,1],[2,3],[3,4])
        time  = ([100,110],[120,130],[130,138])
        self.assertEqual(cadence.time_at_tstep(tstep), time)

        tstep = ([-1,0],[2,4],[4.5,5])
        test = cadence.time_at_tstep(tstep)
        self.assertTrue(Boolean(test.mask) ==
                        [[True,False],[False,False],[True,True]])

        self.assertEqual(cadence.time_at_tstep(0.5), 105.)
        self.assertEqual(cadence.time_at_tstep(3.5), 134.)

        ####################################
        # Other cases
        ####################################

        cadence = Sequence([100.,110.,120.,130.], [10.,10.,5.,10.])
        self.assertFalse(cadence.is_continuous)
        cadence = Sequence([100.,110.,125.,130.], [10.,15.,5.,10.])
        self.assertTrue(cadence.is_continuous)

        self.assertEqual(cadence.tstep_at_time(105.), 0.5)
        self.assertEqual(cadence.tstep_at_time(115.), 4./3.)
        self.assertEqual(cadence.tstep_at_time(127.), 2.4)
        self.assertEqual(cadence.time_at_tstep(0.5), 105.)
        self.assertEqual(cadence.time_at_tstep(4./3.), 115.)
        self.assertEqual(cadence.time_at_tstep(2.4), 127.)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

