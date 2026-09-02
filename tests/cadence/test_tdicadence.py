##########################################################################################
# oops/cadence/tdicadence.py: TDICadence subclass of class Cadence
##########################################################################################

import numpy as np
import pytest

from polymath import Scalar
import oops


def test_tdicadence():
    ######################################################################################
    # 10 lines, 2 stages, TDI downward, 100-120
    ######################################################################################

    cad = oops.cadence.TDICadence(10, 100., 10., 2)
    case_tdicadence_10_100_10_2_down(cad)

    ######################################################################################
    # 10 lines, 2 stages, TDI upward
    ######################################################################################

    cad = oops.cadence.TDICadence(10, 100., 10., 2, tdi_sign=1)
    case_tdicadence_10_100_10_2_up(cad)

    ######################################################################################
    # 100 lines, 100 stages, TDI downward
    ######################################################################################

    cad = oops.cadence.TDICadence(100, 1000., 10., 100)
    case_tdicadence_100_1000_10_100_down(cad)

    ######################################################################################
    # 10 lines, one stage
    ######################################################################################

    cad = oops.cadence.TDICadence(10, 100., 10., 1)
#         print(cad.time_at_tstep(10))
    case_tdicadence_10_100_10_1(cad)

def case_tdicadence_10_100_10_2_down(cad):

    # time_range_at_tstep
    assert cad.time_range_at_tstep(-1) == (100., 120.)
    assert cad.time_range_at_tstep(-1, remask=True) == (Scalar.MASKED, Scalar.MASKED)
    assert cad.time_range_at_tstep(0) == (100., 120.)
    assert cad.time_range_at_tstep(8) == (100., 120.)
    assert cad.time_range_at_tstep(9) == (110., 120.)
    assert cad.time_range_at_tstep(9.5) == (110., 120.)
    assert cad.time_range_at_tstep(10) == (110., 120.)
    assert cad.time_range_at_tstep(10, inclusive=False, remask=False) == (110., 120.)
    assert (cad.time_range_at_tstep(10, inclusive=False, remask=True)
            == (Scalar.MASKED, Scalar.MASKED))
    assert cad.time_range_at_tstep(11) == (110., 120.)
    assert cad.time_range_at_tstep(11, inclusive=False, remask=False) == (110., 120.)
    assert (cad.time_range_at_tstep(11, inclusive=False, remask=True)
            == (Scalar.MASKED, Scalar.MASKED))

    tstep = Scalar(([0,1],[2,9]),([False,False],[True,False]))
    assert np.all(cad.time_range_at_tstep(tstep)[0].vals == [[100,100],[100,110]])
    assert np.all(cad.time_range_at_tstep(tstep)[1].vals == 120)
    assert np.all(cad.time_range_at_tstep(tstep)[0].mask == tstep.mask)
    assert np.all(cad.time_range_at_tstep(tstep)[1].mask == tstep.mask)

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].vals
                  == [[100,100],[100,110]])
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].vals == 120)
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].mask == tstep.mask)
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].mask == tstep.mask)

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].vals
                  == [[100,100],[100,110]])
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].vals == 120)
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].mask == tstep.mask)
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].mask == tstep.mask)

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False, remask=True)[0].vals
                  == [[100,100],[100,110]])
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False, remask=True)[1].vals
                  == 120)
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False, remask=True)[0].mask
                  == [[0,0],[1,1]])
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False, remask=True)[1].mask
                  == [[0,0],[1,1]])

    # time_at_tstep
    assert cad.time_at_tstep(-1.) == 100.
    assert cad.time_at_tstep(-1., remask=True) == Scalar.MASKED
    assert cad.time_at_tstep(0. ) == 100.
    assert cad.time_at_tstep(0.5) == 110.
    assert cad.time_at_tstep(0.9) == 118.
    assert cad.time_at_tstep(1. ) == 100.
    assert cad.time_at_tstep(1.5) == 110.
    assert cad.time_at_tstep(1.9) == 118.
    assert cad.time_at_tstep(9. ) == 110.
    assert cad.time_at_tstep(9.5) == 115.
    assert cad.time_at_tstep(10.) == 120.
    assert cad.time_at_tstep(10., remask=True, inclusive=True) == 120.
    assert cad.time_at_tstep(10., remask=True, inclusive=False) == Scalar.MASKED

    tstep = Scalar(([0,1],[9,10]),([False,True],[False,False]))
    assert np.all(cad.time_at_tstep(tstep).vals == [[100,100],[110,120]])
    assert np.all(cad.time_at_tstep(tstep).mask == tstep.mask)
    assert np.all(cad.time_at_tstep(tstep, remask=True, inclusive=False).mask
                  == [[0,1],[0,1]])

    # time_at_tstep, derivs
    tstep = Scalar([-1, 0, 0.5, 0.9, 1.,1.5, 1.9, 9, 9.5, 10])
    tstep.insert_deriv('t', Scalar(np.arange(10.)))
    assert cad.time_at_tstep(tstep) == [100,100,110,118,100,110,118,110,115,120]
    assert (cad.time_at_tstep(tstep, derivs=True)
            == [100,100,110,118,100,110,118,110,115,120])
    assert (cad.time_at_tstep(tstep, derivs=True, remask=True)
            == [Scalar.MASKED,100,110,118,100,110,118,110,115,120])
    assert (cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False)
            == [Scalar.MASKED,100,110,118,100,110,118,110,115,Scalar.MASKED])
    assert cad.time_at_tstep(tstep, derivs=True).d_dt == [0,20,40,60,80,100,120,70,80,90]
    assert (cad.time_at_tstep(tstep, derivs=True, remask=True).d_dt
            == [Scalar.MASKED,20,40,60,80,100,120,70,80,90])
    assert (cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False).d_dt
            == [Scalar.MASKED,20,40,60,80,100,120,70,80,Scalar.MASKED])

    # tstep_range_at_time
    assert cad.tstep_range_at_time(100.) == (0, 9)
    assert cad.tstep_range_at_time(109.) == (0, 9)
    assert cad.tstep_range_at_time(110.) == (0, 10)
    assert cad.tstep_range_at_time(120.) == (0, 10)

    # self.assertEqual(cad.tstep_range_at_time(120., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(120., inclusive=False)
    assert test0 == test1
    assert cad.tstep_range_at_time(120., remask=True, inclusive=True) == (0, 10)
    assert (cad.tstep_range_at_time(120., remask=True, inclusive=False)
            == (Scalar.MASKED, Scalar.MASKED))

    time = Scalar([100,110,120],[False,False,False])
    assert np.all(cad.tstep_range_at_time(time)[0].vals == (0,0,0))
    assert np.all(cad.tstep_range_at_time(time)[1].vals == (9,10,10))

    # assert np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (0,0,0))
    # assert np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (9,10,0))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    assert np.all(test0.vals[:2] == (0,0))
    assert np.all(test1.vals[:2] == (9,10))
    assert test0.vals[2] == test1.vals[2]  # zero range required, not specific values

    assert not np.any(cad.tstep_range_at_time(time, inclusive=False)[0].mask)
    assert not np.any(cad.tstep_range_at_time(time, inclusive=False)[1].mask)
    # assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals
    #               == (0,0,0))
    # assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals
    #               == (9,10,0))
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals
                  == test0.vals)
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals
                  == test1.vals)
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask
                  == (0,0,1))
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask
                  == (0,0,1))

    time = Scalar([100,110,120],[True,False,False])
    # assert np.all(cad.tstep_range_at_time(time)[0].vals == (0,0,0))
    # assert np.all(cad.tstep_range_at_time(time)[1].vals == (0,10,10))
    (test0, test1) = cad.tstep_range_at_time(time)
    assert test0.vals[0] == test1.vals[0]  # zero range required, not specific values
    assert np.all(test0.vals[1:] == (0,0))
    assert np.all(test1.vals[1:] == (10,10))

    # assert np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (0,0,0))
    # assert np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (0,10,0))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    assert test0.vals[0] == test1.vals[0]  # zero range required, not specific values
    assert test0.vals[1] == 0
    assert test1.vals[1] == 10
    assert test0.vals[2] == test1.vals[2]  # zero range required, not specific values

    assert np.all(cad.tstep_range_at_time(time, inclusive=False)[0].mask == (1,0,0))
    assert np.all(cad.tstep_range_at_time(time, inclusive=False)[1].mask == (1,0,0))
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals
                  == test0.vals)
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals
                  == test1.vals)
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask
                  == (1,0,1))
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask
                  == (1,0,1))

    # tstride_at_tstep
    assert cad.tstride_at_tstep(0) == 0
    assert cad.tstride_at_tstep(8) == 10
    assert cad.tstride_at_tstep(8, sign=-1) == 0
    assert cad.tstride_at_tstep(9) == 10
    assert cad.tstride_at_tstep(9, sign=-1) == 10
    assert cad.tstride_at_tstep(10) == 10

def case_tdicadence_10_100_10_2_up(cad):

    # time_range_at_tstep
    assert cad.time_range_at_tstep(-1) == (110., 120.)
    assert cad.time_range_at_tstep(-1, remask=True) == (Scalar.MASKED, Scalar.MASKED)
    assert cad.time_range_at_tstep(0) == (110., 120.)
    assert cad.time_range_at_tstep(8) == (100., 120.)
    assert cad.time_range_at_tstep(9) == (100., 120.)

    assert cad.time_range_at_tstep(10) == (100., 120.)
    assert cad.time_range_at_tstep(10, inclusive=False, remask=False) == (100., 120.)
    assert (cad.time_range_at_tstep(10, inclusive=False, remask=True)
            == (Scalar.MASKED, Scalar.MASKED))
    assert cad.time_range_at_tstep(11) == (100., 120.)
    assert cad.time_range_at_tstep(11, inclusive=False, remask=False) == (100., 120.)
    assert (cad.time_range_at_tstep(11, inclusive=False, remask=True)
            == (Scalar.MASKED, Scalar.MASKED))

    tstep = Scalar(([0,1],[2,9]),([False,False],[True,False]))
    assert np.all(cad.time_range_at_tstep(tstep)[0].vals == [[110,100],[100,100]])
    assert np.all(cad.time_range_at_tstep(tstep)[1].vals == 120)
    assert np.all(cad.time_range_at_tstep(tstep)[0].mask == tstep.mask)
    assert np.all(cad.time_range_at_tstep(tstep)[1].mask == tstep.mask)

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].vals
                  == [[110,100],[100,100]])
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].vals == 120)
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=True)[0].mask == tstep.mask)
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=True)[1].mask == tstep.mask)

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].vals
                  == [[110,100],[100,100]])
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].vals == 120)
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False)[0].mask == tstep.mask)
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False)[1].mask == tstep.mask)

    tstep = Scalar(([0,1],[2,10]),([False,False],[True,False]))
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False, remask=True)[0].vals
                  == [[110,100],[100,100]])
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False, remask=True)[1].vals
                  == 120)
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False, remask=True)[0].mask
                  == [[0,0],[1,1]])
    assert np.all(cad.time_range_at_tstep(tstep, inclusive=False, remask=True)[1].mask
                  == [[0,0],[1,1]])

    # time_at_tstep
    assert cad.time_at_tstep(-1.) == 110.
    assert cad.time_at_tstep(-1., remask=True) == Scalar.MASKED
    assert cad.time_at_tstep(0. ) == 110.
    assert cad.time_at_tstep(0.5) == 115.
    assert cad.time_at_tstep(0.9) == 119.
    assert cad.time_at_tstep(1. ) == 100.
    assert cad.time_at_tstep(1.5) == 110.
    assert cad.time_at_tstep(1.9) == 118.
    assert cad.time_at_tstep(9. ) == 100.
    assert cad.time_at_tstep(9.5) == 110.
    assert cad.time_at_tstep(10.) == 120.
    assert cad.time_at_tstep(10., remask=True, inclusive=True) == 120.
    assert cad.time_at_tstep(10., remask=True, inclusive=False) == Scalar.MASKED

    assert cad.tstep_range_at_time(100.) == (1, 10)
    assert cad.tstep_range_at_time(109.) == (1, 10)
    assert cad.tstep_range_at_time(110.) == (0, 10)
    assert cad.tstep_range_at_time(120.) == (0, 10)

    # self.assertEqual(cad.tstep_range_at_time(120., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(120., inclusive=False)
    assert test0 == test1  # zero range required, not specific values

    assert cad.tstep_range_at_time(120., remask=True, inclusive=True) == (0, 10)
    assert (cad.tstep_range_at_time(120., remask=True, inclusive=False)
            == (Scalar.MASKED, Scalar.MASKED))

    # time_at_tstep, derivs
    tstep = Scalar([-1, 0, 0.5, 0.9, 1.,1.5, 1.9, 9, 9.5, 10])
    tstep.insert_deriv('t', Scalar(np.arange(10.)))
    assert cad.time_at_tstep(tstep) == [110,110,115,119,100,110,118,100,110,120]
    assert (cad.time_at_tstep(tstep, derivs=True)
            == [110,110,115,119,100,110,118,100,110,120])
    assert (cad.time_at_tstep(tstep, derivs=True, remask=True)
            == [Scalar.MASKED,110,115,119,100,110,118,100,110,120])
    assert (cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False)
            == [Scalar.MASKED,110,115,119,100,110,118,100,110,Scalar.MASKED])
    assert (cad.time_at_tstep(tstep, derivs=True).d_dt
            == [0,10,20,30,80,100,120,140,160,180])
    assert (cad.time_at_tstep(tstep, derivs=True, remask=True).d_dt
            == [Scalar.MASKED,10,20,30,80,100,120,140,160,180])
    assert (cad.time_at_tstep(tstep, derivs=True, remask=True, inclusive=False).d_dt
            == [Scalar.MASKED,10,20,30,80,100,120,140,160,Scalar.MASKED])

    # tstep_range_at_time
    assert cad.tstep_range_at_time(100.) == (1, 10)
    assert cad.tstep_range_at_time(109.) == (1, 10)
    assert cad.tstep_range_at_time(110.) == (0, 10)
    assert cad.tstep_range_at_time(120.) == (0, 10)

    # self.assertEqual(cad.tstep_range_at_time(120., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(120., inclusive=False)
    assert test0 == test1  # zero range required, not specific values

    assert cad.tstep_range_at_time(120., remask=True, inclusive=True) == (0, 10)
    assert (cad.tstep_range_at_time(120., remask=True, inclusive=False)
            == (Scalar.MASKED, Scalar.MASKED))

    time = Scalar([100,110,120],[False,False,False])
    assert np.all(cad.tstep_range_at_time(time)[0].vals == (1,0,0))
    assert np.all(cad.tstep_range_at_time(time)[1].vals == (10,10,10))

    # assert np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (1,0,0))
    # assert np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (10,10,0))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    assert np.all(test0.vals[:2] == (1,0))
    assert np.all(test1.vals[:2] == (10,10))
    assert test0.vals[2] == test1.vals[2]  # zero range required, not specific values

    assert not np.any(cad.tstep_range_at_time(time, inclusive=False)[0].mask)
    assert not np.any(cad.tstep_range_at_time(time, inclusive=False)[1].mask)
    # assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals
    #               == (1,0,0))
    # assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals
    #               == (10,10,0))
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals
                  == test0.vals)
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals
                  == test1.vals)
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask
                  == (0,0,1))
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask
                  == (0,0,1))

    time = Scalar([100,110,120],[False,True,False])
    # assert np.all(cad.tstep_range_at_time(time)[0].vals == (1,0,0))
    # assert np.all(cad.tstep_range_at_time(time)[1].vals == (10,0,10))
    (test0, test1) = cad.tstep_range_at_time(time)
    assert test0.vals[1] == test1.vals[1]  # zero range required, not specific values
    assert np.all(test0.vals[0::2] == (1,0))
    assert np.all(test1.vals[0::2] == (10,10))

    # assert np.all(cad.tstep_range_at_time(time, inclusive=False)[0].vals == (1,0,0))
    # assert np.all(cad.tstep_range_at_time(time, inclusive=False)[1].vals == (10,0,0))
    (test0, test1) = cad.tstep_range_at_time(time, inclusive=False)
    assert test0.vals[0] == 1
    assert test1.vals[0] == 10
    assert test0.vals[1] == test1.vals[1]  # zero range required, not specific values
    assert test0.vals[2] == test1.vals[2]

    assert np.all(cad.tstep_range_at_time(time, inclusive=False)[0].mask == (0,1,0))
    assert np.all(cad.tstep_range_at_time(time, inclusive=False)[1].mask == (0,1,0))
    # assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals
    #               == (1,0,0))
    # assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals
    #               == (10,0,0))
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].vals
                  == test0.vals)
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].vals
                  == test1.vals)
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[0].mask
                  == (0,1,1))
    assert np.all(cad.tstep_range_at_time(time, inclusive=False, remask=True)[1].mask
                  == (0,1,1))

def case_tdicadence_100_1000_10_100_down(cad):

    tstep = Scalar(np.arange(100))
    (time0, time1) = cad.time_range_at_tstep(tstep)
    assert time1 == 2000.
    assert time0 == 1000. + 10. * tstep

    assert cad.tstep_range_at_time(1000.) == (0, 1)
    assert cad.tstep_range_at_time(1010.) == (0, 2)
    assert cad.tstep_range_at_time(1990.) == (0, 100)
    # self.assertEqual(cad.tstep_range_at_time(2000., inclusive=False), (0, 0))
    (test0, test1) = cad.tstep_range_at_time(2000., inclusive=False)
    assert test0 == test1
    assert cad.tstep_range_at_time(2000., remask=True, inclusive=True) == (0, 100)
    assert (cad.tstep_range_at_time(2000., remask=True, inclusive=False)
            == (Scalar.MASKED, Scalar.MASKED))

    # time_is_inside()
    assert cad.time_is_inside([1000,2000], inclusive=True ) == [1,1]
    assert cad.time_is_inside([1000,2000], inclusive=False) == [1,0]

def case_tdicadence_10_100_10_1(cad):

    assert cad.is_continuous
    assert not cad.is_unique        # all ten lines span the one interval

    # time_at_tstep()
    assert cad.time_at_tstep(-0.1) == 100.
    assert cad.time_at_tstep(-0.1, remask=False) == 100.
    assert cad.time_at_tstep(-0.1, remask=True ) == Scalar.MASKED
    assert cad.time_at_tstep( 0  ) == 100.
    assert cad.time_at_tstep( 9.5) == 105.
    assert cad.time_at_tstep(10, remask=False) == 110.
    assert cad.time_at_tstep(10, remask=True ) == 110.
    assert cad.time_at_tstep(10, remask=True, inclusive=False) == Scalar.MASKED

    # time_at_tstep(), derivs
    tstep = Scalar((0., 0.5, 10., 20.))
    tstep.insert_deriv('t', Scalar((2,3,4,5)))

    assert cad.time_at_tstep(tstep, remask=False) == (100,105,110,110)
    assert cad.time_at_tstep(tstep, remask=True) == (100,105,110,Scalar.MASKED)
    assert cad.time_at_tstep(tstep, derivs=True).d_dt == (20,30,40,0)
    assert cad.time_at_tstep(tstep, derivs=True, inclusive=False).d_dt == (20,30,0,0)

    # time_is_inside()
    time = ([99,100],[120,140],[145,150])
    assert not cad.time_is_inside(90)
    assert cad.time_is_inside(100)
    assert cad.time_is_inside(110)
    assert not cad.time_is_inside(110, inclusive=False)
    assert not cad.time_is_inside(111)

    # tstep_at_time()
    assert cad.tstep_at_time( 99) == 0.
    assert cad.tstep_at_time( 99, remask=True) == Scalar.MASKED
    assert cad.tstep_at_time(100) == 0.
    assert cad.tstep_at_time(105) == 0.5
    assert cad.tstep_at_time(110) == 1.
    assert cad.tstep_at_time(110, remask=True) == 1.
    assert cad.tstep_at_time(110, remask=True, inclusive=False) == Scalar.MASKED
    assert cad.tstep_at_time(111) == 1.
    assert cad.tstep_at_time(111, remask=True) == Scalar.MASKED

    # tstep_at_time(), derivs
    time = Scalar((90,100,110,140), derivs={'t': Scalar((100,200,300,400))})
    assert cad.tstep_at_time(time, remask=False, derivs=True).d_dt == (0, 20, 30, 0)
    assert (cad.tstep_at_time(time, remask=False, derivs=True, inclusive=False).d_dt
            == (0, 20, 0, 0))

    # tstep_range_at_time()
    MASKED_TUPLE = (Scalar.MASKED, Scalar.MASKED)
    assert cad.tstep_range_at_time( 99.) == (0,0)
    assert cad.tstep_range_at_time( 99., remask=True) == MASKED_TUPLE
    assert cad.tstep_range_at_time(100.) == (0,10)
    assert cad.tstep_range_at_time(105.) == (0,10)
    assert cad.tstep_range_at_time(110.) == (0,10)
    assert cad.tstep_range_at_time(110., remask=True) == (0,10)
    assert cad.tstep_range_at_time(110., remask=True, inclusive=False) == MASKED_TUPLE
    assert cad.tstep_range_at_time(135., remask=True) == MASKED_TUPLE

    tstep0, tstep1 = cad.tstep_range_at_time(110., inclusive=False)
    assert tstep0 == tstep1    # indicates zero range

    tstep0, tstep1 = cad.tstep_range_at_time(135.)
    assert tstep0 == tstep1

    # time_range_at_tstep()
    tstep = Scalar((-1,0,0.5,10,12))
    assert cad.time_range_at_tstep(tstep)[0] == 5*[100]
    assert cad.time_range_at_tstep(tstep)[1] == 5*[110]

    assert cad.time_range_at_tstep(tstep[0], remask=True) == MASKED_TUPLE
    assert cad.time_range_at_tstep(tstep[1:4], remask=True)[0] == 3*[100]
    assert cad.time_range_at_tstep(tstep[1:4], remask=True)[1] == 3*[110]
    assert cad.time_range_at_tstep(tstep[4], remask=True) == MASKED_TUPLE

    assert cad.time_range_at_tstep(tstep[1:3], remask=True, inclusive=False)[0] == 2*[100]
    assert cad.time_range_at_tstep(tstep[1:3], remask=True, inclusive=False)[1] == 2*[110]

    assert cad.time_range_at_tstep(tstep[3], remask=True, inclusive=False) == MASKED_TUPLE

    # tstride_at_tstep
    assert cad.tstride_at_tstep(0) == 0
    assert cad.tstride_at_tstep(0.5) == 0
    assert cad.tstride_at_tstep(9) == 10
    assert cad.tstride_at_tstep(9, sign=-1) == 0
    assert cad.tstride_at_tstep(10) == 10


def test_tdicadence_time_shift() -> None:
    """A shifted TDICadence keeps every parameter and offsets all of its times."""

    cad = oops.cadence.TDICadence(10, 100., 10., 2, tdi_sign=1)
    shifted = cad.time_shift(5.)

    assert shifted._lines == cad._lines
    assert shifted._tstart == cad._tstart + 5.
    assert shifted._tdi_texp == cad._tdi_texp
    assert shifted._tdi_stages == cad._tdi_stages
    assert shifted._tdi_sign == cad._tdi_sign
    assert shifted.time[0] == cad.time[0] + 5.
    assert shifted.time[1] == cad.time[1] + 5.


@pytest.mark.parametrize('stages', [1, 2, 4, 10])
def test_tdi_shifts_after_time_never_exceeds_the_shift_count(stages: int) -> None:
    """A cadence of N stages performs N-1 shifts, so no time has more than that to come.

    A time before the exposure begins still has the whole sequence ahead of it, which is
    one fewer than the number of stages.
    """

    cad = oops.cadence.TDICadence(10, 100., 10., stages)

    for time in (-1.e6, 0., 99., 100.):
        assert cad.tdi_shifts_after_time(time) == stages - 1


def test_tdi_shifts_after_time_counts_down_once_per_stage() -> None:
    """Each elapsed TDI interval completes one shift, leaving one fewer still to come."""

    cad = oops.cadence.TDICadence(10, 100., 10., 4)

    assert cad.tdi_shifts_after_time(100.) == 3
    assert cad.tdi_shifts_after_time(105.) == 3      # partway through the first stage
    assert cad.tdi_shifts_after_time(110.) == 2
    assert cad.tdi_shifts_after_time(120.) == 1
    assert cad.tdi_shifts_after_time(130.) == 0
    assert cad.tdi_shifts_after_time(140.) == 0      # the end time
    assert cad.tdi_shifts_after_time(1.e6) == 0


def test_tdi_shifts_after_time_ignores_the_shift_direction() -> None:
    """The shifts remaining depend on the time alone, not on which way the DNs move."""

    down = oops.cadence.TDICadence(10, 100., 10., 4, tdi_sign=-1)
    up = oops.cadence.TDICadence(10, 100., 10., 4, tdi_sign=1)

    times = Scalar([0., 100., 115., 140., 1.e6])
    assert down.tdi_shifts_after_time(times) == up.tdi_shifts_after_time(times)


def test_tdi_shifts_after_time_masks_times_outside_the_cadence() -> None:
    """With remask, a time outside the exposure is masked instead of being clipped."""

    cad = oops.cadence.TDICadence(10, 100., 10., 4)
    shifts = cad.tdi_shifts_after_time(Scalar([99., 100., 140., 141.]), remask=True)

    assert list(shifts.mask) == [True, False, False, True]


def test_tdi_shifts_at_line_and_after_time_share_an_upper_bound() -> None:
    """Both counts describe the same shifts, so neither can exceed the shift count."""

    cad = oops.cadence.TDICadence(10, 100., 10., 4)

    lines = Scalar(np.arange(-3, 14))
    times = Scalar(np.arange(80., 161., 5.))

    assert cad.tdi_shifts_at_line(lines).max() == cad._max_shifts
    assert cad.tdi_shifts_after_time(times).max() == cad._max_shifts


def test_tdicadence_is_unique_only_with_one_line() -> None:
    """A lone line has the cadence to itself; any more and they overlap in time."""

    assert oops.cadence.TDICadence(1, 100., 10., 1).is_unique is True
    assert oops.cadence.TDICadence(2, 100., 10., 1).is_unique is False
    assert oops.cadence.TDICadence(10, 100., 10., 1).is_unique is False
    assert oops.cadence.TDICadence(10, 100., 10., 4).is_unique is False


@pytest.mark.parametrize('lines,stages', [(1, 1), (2, 1), (2, 2), (10, 1), (10, 4),
                                          (10, 10)])
@pytest.mark.parametrize('sign', [-1, 1])
def test_tdicadence_is_unique_agrees_with_the_active_lines(lines: int, stages: int,
                                                           sign: int) -> None:
    """The flag means what the base class says: no time is shared by two time steps."""

    cad = oops.cadence.TDICadence(lines, 100., 10., stages, tdi_sign=sign)

    times = np.linspace(cad.time[0], cad.time[1], 101)
    spans = [int(cad.tstep_range_at_time(float(t))[1].vals)
             - int(cad.tstep_range_at_time(float(t))[0].vals) for t in times]

    assert cad.is_unique == (max(spans) <= 1)


def test_tdicadence_tstep_at_time_requires_a_single_stage() -> None:
    """With more stages each line spans a different interval, so no one step will do."""

    cad = oops.cadence.TDICadence(10, 100., 10., 4)

    with pytest.raises(NotImplementedError, match='single TDI stage'):
        cad.tstep_at_time(105.)


def test_tdicadence_tstep_at_time_reports_the_first_line() -> None:
    """Every line of a one-stage cadence gives the same time, so the first stands in."""

    cad = oops.cadence.TDICadence(10, 100., 10., 1)
    tstep = cad.tstep_at_time(105.)

    assert tstep == 0.5
    assert cad.time_at_tstep(tstep) == 105.
    assert cad.time_at_tstep(tstep + 5.) == 105.    # and so does every other line


##########################################################################################
