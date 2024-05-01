################################################################################
# oops/cadence/metronome.py: Metronome subclass of class Cadence
################################################################################

import numpy as np
import unittest

from polymath import Boolean, Scalar
import oops

# Tests are defined here as separate functions so they can also be used for
# testing Sequences that are defined to simulate the behavior of Metronomes.

class Test_Metronome(unittest.TestCase):

  def runTest(self):

    np.random.seed(4182)

    ############################################
    # Tests for continuous case
    # 100-110, 110-120, 120-130, 130-140
    ############################################

    cadence = oops.cadence.Metronome(100., 10., 10., 4)
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
    cadence = oops.cadence.Metronome(100., 10., 10., 4, clip=False)
    self.assertEqual(cadence.time_at_tstep(-0.5, remask=False),  95.)
    self.assertEqual(cadence.time_at_tstep( 4.5, remask=False), 145.)

    self.assertEqual(cadence.tstep_at_time( 95., remask=False), -0.5)
    self.assertEqual(cadence.tstep_at_time(145., remask=False),  4.5)

    ############################################
    # Discontinuous case
    # 100-107.5, 110-117.5, 120-127.5, 130-137.5
    ############################################

    cadence = oops.cadence.Metronome(100., 10., 7.5, 4)
    case_discontinuous(self, cadence)

    # Unclipped Metronome tests
    cadence = oops.cadence.Metronome(100., 10., 8., 4, clip=False)
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

    cadence = oops.cadence.Metronome(100., 10., 40., 4)
    case_non_unique(self, cadence)

    # Unclipped Metronome tests
    cadence = oops.cadence.Metronome(100., 10., 40., 4, clip=False)
    self.assertEqual(cadence.time_at_tstep(-0.5,remask=False), 110.)
    self.assertEqual(cadence.time_at_tstep( 4.5,remask=False), 160.)
    self.assertEqual(cadence.tstep_at_time(170., inclusive=False), 7.)
    self.assertEqual(cadence.tstep_at_time(171., remask=False), 7.025)
    self.assertEqual(cadence.tstep_range_at_time(170., inclusive=False), (3,3))

    ############################################
    # Partial overlap case
    # 100-140, 130-170, 160-200, 190-230
    ############################################

    cadence = oops.cadence.Metronome(100., 30., 40., 4)
    case_partial_overlap(self, cadence)

    # Unclipped Metronome tests
    cadence = oops.cadence.Metronome(100., 30., 40., 4, clip=False)
    self.assertEqual(cadence.time_at_tstep(-0.5,remask=False),  90.)
    self.assertEqual(cadence.time_at_tstep( 4.5,remask=False), 240.)
    self.assertEqual(cadence.tstep_at_time(230., inclusive=False), 4.25)
    self.assertEqual(cadence.tstep_at_time(235., remask=False), 4.375)

    ############################################
    # One time step, 100-110
    ############################################

    cadence = oops.cadence.Metronome(100., 22., 10., 1)
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
