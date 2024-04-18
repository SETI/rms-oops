################################################################################
# oops/cadence/tdicadence.py: TDICadence subclass of class Cadence
################################################################################

import numpy as np
import unittest

from polymath import Scalar
import oops


class Test_TDICadence(unittest.TestCase):

    def runTest(self):

        ########################################
        # 10 lines, 2 stages, TDI downward, 100-120
        ########################################

        cad = oops.cadence.TDICadence(10, 100., 10., 2)
        case_tdicadence_10_100_10_2_down(self, cad)

        ########################################
        # 10 lines, 2 stages, TDI upward
        ########################################

        cad = oops.cadence.TDICadence(10, 100., 10., 2, tdi_sign=1)
        case_tdicadence_10_100_10_2_up(self, cad)

        ########################################
        # 100 lines, 100 stages, TDI downward
        ########################################

        cad = oops.cadence.TDICadence(100, 1000., 10., 100)
        case_tdicadence_100_1000_10_100_down(self, cad)

        ########################################
        # 10 lines, one stage
        ########################################

        cad = oops.cadence.TDICadence(10, 100., 10., 1)
#         print(cad.time_at_tstep(10))
        case_tdicadence_10_100_10_1(self, cad)

def case_tdicadence_10_100_10_2_down(self, cad):

    # time_range_at_tstep
    self.assertEqual(cad.time_range_at_tstep(-1), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(-1, remask=True), (Scalar.MASKED, Scalar.MASKED))
    self.assertEqual(cad.time_range_at_tstep(0), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(8), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(9), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(9.5),(110., 120.))
    self.assertEqual(cad.time_range_at_tstep(10), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(10, inclusive=False,
                                                 remask=False), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(10, inclusive=False,
                                                 remask=True), (Scalar.MASKED, Scalar.MASKED))
    self.assertEqual(cad.time_range_at_tstep(11), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(11, inclusive=False,
                                                 remask=False), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(11, inclusive=False,
                                                 remask=True), (Scalar.MASKED, Scalar.MASKED))

    tstep = Scalar(([0,1],[2,9]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[0].vals ==
                           [[100,100],[100,110]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].vals ==
                           [[100,100],[100,110]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].vals ==
                           [[100,100],[100,110]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[0].vals == [[100,100],[100,110]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[0].mask == [[0,0],[1,1]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[1].mask == [[0,0],[1,1]]))

    # time_at_tstep
    self.assertEqual(cad.time_at_tstep(-1.), 100.)
    self.assertEqual(cad.time_at_tstep(-1., remask=True), Scalar.MASKED)
    self.assertEqual(cad.time_at_tstep(0. ), 100.)
    self.assertEqual(cad.time_at_tstep(0.5), 110.)
    self.assertEqual(cad.time_at_tstep(0.9), 118.)
    self.assertEqual(cad.time_at_tstep(1. ), 100.)
    self.assertEqual(cad.time_at_tstep(1.5), 110.)
    self.assertEqual(cad.time_at_tstep(1.9), 118.)
    self.assertEqual(cad.time_at_tstep(9. ), 110.)
    self.assertEqual(cad.time_at_tstep(9.5), 115.)
    self.assertEqual(cad.time_at_tstep(10.), 120.)
    self.assertEqual(cad.time_at_tstep(10., remask=True,
                                            inclusive=True), 120.)
    self.assertEqual(cad.time_at_tstep(10., remask=True,
                                            inclusive=False), Scalar.MASKED)

    tstep = Scalar(([0,1],[9,10]),([False,True],[False,False]))
    self.assertTrue(np.all(cad.time_at_tstep(tstep).vals == [[100,100],[110,120]]))
    self.assertTrue(np.all(cad.time_at_tstep(tstep).mask == tstep.mask))
    self.assertTrue(np.all(cad.time_at_tstep(tstep, remask=True,
                                             inclusive=False).mask == [[0,1],[0,1]]))

    # time_at_tstep, derivs
    tstep = Scalar([-1, 0, 0.5, 0.9, 1.,1.5, 1.9, 9, 9.5, 10])
    tstep.insert_deriv('t', Scalar(np.arange(10.)))
    self.assertEqual(cad.time_at_tstep(tstep),
                    [100,100,110,118,100,110,118,110,115,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True),
                    [100,100,110,118,100,110,118,110,115,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True),
                    [Scalar.MASKED,100,110,118,100,110,118,110,115,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False),
                    [Scalar.MASKED,100,110,118,100,110,118,110,115,Scalar.MASKED])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True).d_dt,
                    [0,20,40,60,80,100,120,70,80,90])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True).d_dt,
                    [Scalar.MASKED,20,40,60,80,100,120,70,80,90])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False).d_dt,
                    [Scalar.MASKED,20,40,60,80,100,120,70,80,Scalar.MASKED])

    # tstep_range_at_time
    self.assertEqual(cad.tstep_range_at_time(100.), (0, 9))
    self.assertEqual(cad.tstep_range_at_time(109.), (0, 9))
    self.assertEqual(cad.tstep_range_at_time(110.), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120.), (0, 10))

    # self.assertEqual(cad.tstep_range_at_time(120., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(120., inclusive=False)
    self.assertEqual(test0, test1)
    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=True), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=False), (Scalar.MASKED, Scalar.MASKED))

    time = Scalar([100,110,120],[False,False,False])
    self.assertTrue(np.all(cad.tstep_range_at_time(time)[0].vals == (0,0,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time)[1].vals == (9,10,10)))

    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (0,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (9,10,0)))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    self.assertTrue(np.all(test0.vals[:2] == (0,0)))
    self.assertTrue(np.all(test1.vals[:2] == (9,10)))
    self.assertEqual(test0.vals[2], test1.vals[2])  # zero range is required, specific values are not

    self.assertTrue(not np.any(cad.tstep_range_at_time(time, inclusive=False)[0].mask))
    self.assertTrue(not np.any(cad.tstep_range_at_time(time, inclusive=False)[1].mask))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == (0,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == (9,10,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == test0.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == test1.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask == (0,0,1)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask == (0,0,1)))

    time = Scalar([100,110,120],[True,False,False])
    # self.assertTrue(np.all(cad.tstep_range_at_time(time)[0].vals == (0,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time)[1].vals == (0,10,10)))
    (test0, test1) = cad.tstep_range_at_time(time)
    self.assertEqual(test0.vals[0], test1.vals[0])  # zero range is required, specific values are not
    self.assertTrue(np.all(test0.vals[1:] == (0,0)))
    self.assertTrue(np.all(test1.vals[1:] == (10,10)))

    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (0,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (0,10,0)))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    self.assertEqual(test0.vals[0], test1.vals[0])  # zero range is required, specific values are not
    self.assertEqual(test0.vals[1], 0)
    self.assertEqual(test1.vals[1], 10)
    self.assertEqual(test0.vals[2], test1.vals[2])  # zero range is required, specific values are not

    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].mask == (1,0,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].mask == (1,0,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == test0.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == test1.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask == (1,0,1)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask == (1,0,1)))

    # tstride_at_tstep
    self.assertEqual(cad.tstride_at_tstep(0), 0)
    self.assertEqual(cad.tstride_at_tstep(8), 10)
    self.assertEqual(cad.tstride_at_tstep(8, sign=-1), 0)
    self.assertEqual(cad.tstride_at_tstep(9), 10)
    self.assertEqual(cad.tstride_at_tstep(9, sign=-1), 10)
    self.assertEqual(cad.tstride_at_tstep(10), 10)

def case_tdicadence_10_100_10_2_up(self, cad):

    # time_range_at_tstep
    self.assertEqual(cad.time_range_at_tstep(-1), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(-1, remask=True), (Scalar.MASKED, Scalar.MASKED))
    self.assertEqual(cad.time_range_at_tstep(0), (110., 120.))
    self.assertEqual(cad.time_range_at_tstep(8), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(9), (100., 120.))

    self.assertEqual(cad.time_range_at_tstep(10), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(10, inclusive=False,
                                                 remask=False), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(10, inclusive=False,
                                                 remask=True), (Scalar.MASKED, Scalar.MASKED))
    self.assertEqual(cad.time_range_at_tstep(11), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(11, inclusive=False,
                                                 remask=False), (100., 120.))
    self.assertEqual(cad.time_range_at_tstep(11, inclusive=False,
                                                 remask=True), (Scalar.MASKED, Scalar.MASKED))

    tstep = Scalar(([0,1],[2,9]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[0].vals ==
                           [[110,100],[100,100]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].vals ==
                           [[110,100],[100,100]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].vals ==
                           [[110,100],[100,100]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].mask == tstep.mask))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].mask == tstep.mask))

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[0].vals == [[110,100],[100,100]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[1].vals == 120))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[0].mask == [[0,0],[1,1]]))
    self.assertTrue(np.all(cad.time_range_at_tstep(tstep, inclusive=False,
                                                   remask=True)[1].mask == [[0,0],[1,1]]))

    # time_at_tstep
    self.assertEqual(cad.time_at_tstep(-1.), 110.)
    self.assertEqual(cad.time_at_tstep(-1., remask=True), Scalar.MASKED)
    self.assertEqual(cad.time_at_tstep(0. ), 110.)
    self.assertEqual(cad.time_at_tstep(0.5), 115.)
    self.assertEqual(cad.time_at_tstep(0.9), 119.)
    self.assertEqual(cad.time_at_tstep(1. ), 100.)
    self.assertEqual(cad.time_at_tstep(1.5), 110.)
    self.assertEqual(cad.time_at_tstep(1.9), 118.)
    self.assertEqual(cad.time_at_tstep(9. ), 100.)
    self.assertEqual(cad.time_at_tstep(9.5), 110.)
    self.assertEqual(cad.time_at_tstep(10.), 120.)
    self.assertEqual(cad.time_at_tstep(10., remask=True,
                                            inclusive=True), 120.)
    self.assertEqual(cad.time_at_tstep(10., remask=True,
                                            inclusive=False), Scalar.MASKED)

    self.assertEqual(cad.tstep_range_at_time(100.), (1, 10))
    self.assertEqual(cad.tstep_range_at_time(109.), (1, 10))
    self.assertEqual(cad.tstep_range_at_time(110.), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120.), (0, 10))

    # self.assertEqual(cad.tstep_range_at_time(120., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(120., inclusive=False)
    self.assertEqual(test0, test1)  # zero range is required; specific values are not

    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=True), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=False), (Scalar.MASKED, Scalar.MASKED))

    # time_at_tstep, derivs
    tstep = Scalar([-1, 0, 0.5, 0.9, 1.,1.5, 1.9, 9, 9.5, 10])
    tstep.insert_deriv('t', Scalar(np.arange(10.)))
    self.assertEqual(cad.time_at_tstep(tstep),
                    [110,110,115,119,100,110,118,100,110,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True),
                    [110,110,115,119,100,110,118,100,110,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True),
                    [Scalar.MASKED,110,115,119,100,110,118,100,110,120])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False),
                    [Scalar.MASKED,110,115,119,100,110,118,100,110,Scalar.MASKED])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True).d_dt,
                    [0,10,20,30,80,100,120,140,160,180])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True).d_dt,
                    [Scalar.MASKED,10,20,30,80,100,120,140,160,180])
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False).d_dt,
                    [Scalar.MASKED,10,20,30,80,100,120,140,160,Scalar.MASKED])

    # tstep_range_at_time
    self.assertEqual(cad.tstep_range_at_time(100.), (1, 10))
    self.assertEqual(cad.tstep_range_at_time(109.), (1, 10))
    self.assertEqual(cad.tstep_range_at_time(110.), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120.), (0, 10))

    # self.assertEqual(cad.tstep_range_at_time(120., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(120., inclusive=False)
    self.assertEqual(test0, test1)  # zero range is required; specific values are not

    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=True), (0, 10))
    self.assertEqual(cad.tstep_range_at_time(120., remask=True,
                                                   inclusive=False), (Scalar.MASKED, Scalar.MASKED))

    time = Scalar([100,110,120],[False,False,False])
    self.assertTrue(np.all(cad.tstep_range_at_time(time)[0].vals == (1,0,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time)[1].vals == (10,10,10)))

    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (1,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (10,10,0)))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    self.assertTrue(np.all(test0.vals[:2] == (1,0)))
    self.assertTrue(np.all(test1.vals[:2] == (10,10)))
    self.assertEqual(test0.vals[2], test1.vals[2])  # zero range is required, specific values are not

    self.assertTrue(not np.any(cad.tstep_range_at_time(time, inclusive=False)[0].mask))
    self.assertTrue(not np.any(cad.tstep_range_at_time(time, inclusive=False)[1].mask))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == (1,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == (10,10,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == test0.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == test1.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask == (0,0,1)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask == (0,0,1)))

    time = Scalar([100,110,120],[False,True,False])
    # self.assertTrue(np.all(cad.tstep_range_at_time(time)[0].vals == (1,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time)[1].vals == (10,0,10)))
    (test0, test1) = cad.tstep_range_at_time(time)
    self.assertEqual(test0.vals[1], test1.vals[1])  # zero range is required, specific values are not
    self.assertTrue(np.all(test0.vals[0::2] == (1,0)))
    self.assertTrue(np.all(test1.vals[0::2] == (10,10)))

    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (1,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (10,0,0)))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    self.assertEqual(test0.vals[0], 1)
    self.assertEqual(test1.vals[0], 10)
    self.assertEqual(test0.vals[1], test1.vals[1])  # zero range is required, specific values are not
    self.assertEqual(test0.vals[2], test1.vals[2])

    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[0].mask == (0,1,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False)[1].mask == (0,1,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == (1,0,0)))
    # self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == (10,0,0)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals == test0.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals == test1.vals))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask == (0,1,1)))
    self.assertTrue(np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask == (0,1,1)))

def case_tdicadence_100_1000_10_100_down(self, cad):

    tstep = Scalar(np.arange(100))
    (time0, time1) = cad.time_range_at_tstep(tstep)
    self.assertEqual(time1, 2000.)
    self.assertEqual(time0, 1000. + 10. * tstep)

    self.assertEqual(cad.tstep_range_at_time(1000.), (0, 1))
    self.assertEqual(cad.tstep_range_at_time(1010.), (0, 2))
    self.assertEqual(cad.tstep_range_at_time(1990.), (0, 100))
    # self.assertEqual(cad.tstep_range_at_time(2000., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(2000., inclusive=False)
    self.assertEqual(test0, test1)
    self.assertEqual(cad.tstep_range_at_time(2000., remask=True,
                                                    inclusive=True), (0, 100))
    self.assertEqual(cad.tstep_range_at_time(2000., remask=True,
                                                    inclusive=False), (Scalar.MASKED, Scalar.MASKED))

    # time_is_inside()
    self.assertEqual(cad.time_is_inside([1000,2000], inclusive=True ), [1,1])
    self.assertEqual(cad.time_is_inside([1000,2000], inclusive=False), [1,0])

def case_tdicadence_10_100_10_1(self, cad):

    self.assertTrue(cad.is_continuous)
    self.assertTrue(cad.is_unique)

    # time_at_tstep()
    self.assertEqual(cad.time_at_tstep(-0.1), 100.)
    self.assertEqual(cad.time_at_tstep(-0.1, remask=False), 100.)
    self.assertEqual(cad.time_at_tstep(-0.1, remask=True ), Scalar.MASKED)
    self.assertEqual(cad.time_at_tstep( 0  ), 100.)
    self.assertEqual(cad.time_at_tstep( 9.5), 105.)
    self.assertEqual(cad.time_at_tstep(10, remask=False), 110.)
    self.assertEqual(cad.time_at_tstep(10, remask=True ), 110.)
    self.assertEqual(cad.time_at_tstep(10, remask=True, inclusive=False), Scalar.MASKED)

    # time_at_tstep(), derivs
    tstep = Scalar((0., 0.5, 10., 20.))
    tstep.insert_deriv('t', Scalar((2,3,4,5)))

    self.assertEqual(cad.time_at_tstep(tstep, remask=False), (100,105,110,110))
    self.assertEqual(cad.time_at_tstep(tstep, remask=True), (100,105,110,Scalar.MASKED))
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True).d_dt, (20,30,40,0))
    self.assertEqual(cad.time_at_tstep(tstep, derivs=True, inclusive=False).d_dt, (20,30,0,0))

    # time_is_inside()
    time = ([99,100],[120,140],[145,150])
    self.assertFalse(cad.time_is_inside(90))
    self.assertTrue (cad.time_is_inside(100))
    self.assertTrue (cad.time_is_inside(110))
    self.assertFalse(cad.time_is_inside(110, inclusive=False))
    self.assertFalse(cad.time_is_inside(111))

    # tstep_at_time()
    self.assertEqual(cad.tstep_at_time( 99), 0.)
    self.assertEqual(cad.tstep_at_time( 99, remask=True), Scalar.MASKED)
    self.assertEqual(cad.tstep_at_time(100), 0.)
    self.assertEqual(cad.tstep_at_time(105), 0.5)
    self.assertEqual(cad.tstep_at_time(110), 1.)
    self.assertEqual(cad.tstep_at_time(110, remask=True), 1.)
    self.assertEqual(cad.tstep_at_time(110, remask=True, inclusive=False), Scalar.MASKED)
    self.assertEqual(cad.tstep_at_time(111), 1.)
    self.assertEqual(cad.tstep_at_time(111, remask=True), Scalar.MASKED)

    # tstep_at_time(), derivs
    time = Scalar((90,100,110,140), derivs={'t': Scalar((100,200,300,400))})
    self.assertEqual(cad.tstep_at_time(time, remask=False, derivs=True).d_dt, (0, 20, 30, 0))
    self.assertEqual(cad.tstep_at_time(time, remask=False, derivs=True,
                                                 inclusive=False).d_dt, (0, 20, 0, 0))

    # tstep_range_at_time()
    MASKED_TUPLE = (Scalar.MASKED, Scalar.MASKED)
    self.assertEqual(cad.tstep_range_at_time( 99.), (0,0))
    self.assertEqual(cad.tstep_range_at_time( 99., remask=True), MASKED_TUPLE)
    self.assertEqual(cad.tstep_range_at_time(100.), (0,10))
    self.assertEqual(cad.tstep_range_at_time(105.), (0,10))
    self.assertEqual(cad.tstep_range_at_time(110.), (0,10))
    self.assertEqual(cad.tstep_range_at_time(110., remask=True), (0,10))
    self.assertEqual(cad.tstep_range_at_time(110., remask=True, inclusive=False), MASKED_TUPLE)
    self.assertEqual(cad.tstep_range_at_time(135., remask=True), MASKED_TUPLE)

    tstep0, tstep1 = cad.tstep_range_at_time(110., inclusive=False)
    self.assertEqual(tstep0, tstep1)    # indicates zero range

    tstep0, tstep1 = cad.tstep_range_at_time(135.)
    self.assertEqual(tstep0, tstep1)

    # time_range_at_tstep()
    tstep = Scalar((-1,0,0.5,10,12))
    self.assertEqual(cad.time_range_at_tstep(tstep)[0], 5*[100])
    self.assertEqual(cad.time_range_at_tstep(tstep)[1], 5*[110])

    self.assertEqual(cad.time_range_at_tstep(tstep[0], remask=True), MASKED_TUPLE)
    self.assertEqual(cad.time_range_at_tstep(tstep[1:4], remask=True)[0], 3*[100])
    self.assertEqual(cad.time_range_at_tstep(tstep[1:4], remask=True)[1], 3*[110])
    self.assertEqual(cad.time_range_at_tstep(tstep[4], remask=True), MASKED_TUPLE)

    self.assertEqual(cad.time_range_at_tstep(tstep[1:3], remask=True, inclusive=False)[0], 2*[100])
    self.assertEqual(cad.time_range_at_tstep(tstep[1:3], remask=True, inclusive=False)[1], 2*[110])

    self.assertEqual(cad.time_range_at_tstep(tstep[3], remask=True, inclusive=False), MASKED_TUPLE)

    # tstride_at_tstep
    self.assertEqual(cad.tstride_at_tstep(0), 0)
    self.assertEqual(cad.tstride_at_tstep(0.5), 0)
    self.assertEqual(cad.tstride_at_tstep(9), 10)
    self.assertEqual(cad.tstride_at_tstep(9, sign=-1), 0)
    self.assertEqual(cad.tstride_at_tstep(10), 10)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
