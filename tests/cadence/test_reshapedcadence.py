################################################################################
# tests/cadence/test_reshapedcadence.py
################################################################################

import numpy as np
import pytest

from polymath import Scalar, Pair, Vector
import oops

from tests.cadence.test_dualcadence import case_dual_metronome


def case_reshape_roundtrip(oldshape, newshape, arg):
    """A complete there-and-back test of ReshapedCadence._reshape_tstep()."""

    oldstride = np.cumprod((oldshape + (1,))[::-1])[-2::-1]
    newstride = np.cumprod((newshape + (1,))[::-1])[-2::-1]
    oldrank = len(oldshape)
    newrank = len(newshape)

    arg1 = oops.cadence.ReshapedCadence._reshape_tstep(arg,
                                          oldshape, oldstride, oldrank,
                                          newshape, newstride, newrank,
                                          np.prod(oldshape))
    arg2 = oops.cadence.ReshapedCadence._reshape_tstep(arg1,
                                          newshape, newstride, newrank,
                                          oldshape, oldstride, oldrank,
                                          np.prod(oldshape))

    assert arg == arg2

    assert type(arg) is type(arg2)

    if arg.is_int():
        assert arg2.is_int()
    else:
        assert arg2.is_float()

def test_reshapedcadence():
    from oops.cadence.metronome import Metronome

    case_reshape_roundtrip((10,), (10,), Scalar(1))
    case_reshape_roundtrip((10,), (2,5), Scalar(1))
    case_reshape_roundtrip((10,), (2,5), Scalar(1.5))
    case_reshape_roundtrip((10,), (2,5), Scalar(np.arange(10)))
    case_reshape_roundtrip((10,), (2,5), Scalar(np.arange(20)/2.))
    case_reshape_roundtrip((10,), (2,5), Scalar(np.arange(10).reshape(5,2)))
    case_reshape_roundtrip((10,), (2,5), Scalar((np.arange(20)/2.).reshape(2,5,2)))

    case_reshape_roundtrip((2,3,4), (24,), Vector((1,2,3)))
    case_reshape_roundtrip((2,3,4), (24,), Vector((1,2,3.5)))
    case_reshape_roundtrip((2,3,4), (24,), Vector([(1,2,3),(1,2,3.5),(0,0,0.25)]))

    case_reshape_roundtrip((2,3,4), (4,6), Vector((1,2,3)))
    case_reshape_roundtrip((2,3,4), (4,6), Vector((1,2,3.5)))
    case_reshape_roundtrip((2,3,4), (4,6), Vector([(1,2,3),(1,2,3.5),(0,0,0.25)]))

    ########################################################################
    # Compare a Metronome reshaped to 2-D to an equivalent DualCadence
    # cad1d: 100-101, 102-103, 104-105, ... 198-199.

    cad1d = Metronome(100., 2., 1., 50)

#         long = oops.cadence.Metronome(100., 10., 1., 10)
#         short = oops.cadence.Metronome(0, 2., 1., 5)
#         cad2d = oops.cadence.DualCadence(long, short)
    cad2d = oops.cadence.ReshapedCadence(cad1d, (10,5))

    case_dual_metronome(cad1d, cad2d)

    ############################################
    # Weirdly reshaped case, 100 -> (25,4)
    # 100-110, 110-120, 120-130, ...
    ############################################

    cadence = oops.cadence.Metronome(100., 10., 10., 100)
    reshaped = oops.cadence.ReshapedCadence(cadence, (25,4))

    assert reshaped.is_continuous
    assert reshaped.is_unique

    assert reshaped.time_at_tstep((0,0)) == 100.
    assert reshaped.time_at_tstep((0,1)) == 110.
    assert reshaped.time_at_tstep((1,0)) == 140.
    assert reshaped.time_at_tstep((1,1)) == 150.
    assert reshaped.time_at_tstep((1,1.5)) == 155.

    tstep = Pair([[(0,0),(0,1)],[(1,0),(1,1)]])
    assert reshaped.time_at_tstep(tstep) == [[100,110],[140,150]]

    tstep = Pair([[(0,0),(0,1)],[(1,0),(1,1)]], [[1,0],[0,0]])
    time = reshaped.time_at_tstep(tstep)
    assert np.all(tstep.mask == time.mask)
    assert time[0,0] == Scalar.MASKED
    assert reshaped.time_at_tstep(tstep) == [[Scalar.MASKED,110],[140,150]]

    assert reshaped.tstep_at_time(100.) == (0,0)
    assert reshaped.tstep_at_time(110.) == (0,1)
    assert reshaped.tstep_at_time(140.) == (1,0)
    assert reshaped.tstep_at_time(150.) == (1,1)
    assert reshaped.tstep_at_time(155.) == (1,1.5)

    for i in np.arange(-2., 28., 0.5):
      for j in np.arange(-1., 5., 0.25):
        k = 4*np.floor(i) + j
        (time1a, time1b) = cadence.time_range_at_tstep(k)
        (time2a, time2b) = reshaped.time_range_at_tstep((i,j), remask=True,
                                                        inclusive=False)
        if not time2a.mask:
            assert time1a == time2a
            assert time1b == time2b

            time = reshaped.time_at_tstep((i,j), remask=True)
            tstep = reshaped.tstep_at_time(time)
            assert tstep == (np.floor(i),j)

    ############################################
    # Weirdly reshaped case, 100 -> (25,4), discontinuous
    # [100-108, 116-124, 132-140, 148-156], [164-172, ...], [..., 1684-1692]
    ############################################

    cadence = oops.cadence.Metronome(100., 16., 8., 100)
    reshaped = oops.cadence.ReshapedCadence(cadence, (25,4))
    assert not reshaped.is_continuous
    assert reshaped.is_unique
    assert reshaped.time_at_tstep((0,0)) == 100.
    assert reshaped.time_at_tstep((0,1)) == 116.
    assert reshaped.time_at_tstep((1,0)) == 164.
    assert reshaped.time_at_tstep((1,1)) == 180.
    assert reshaped.time_at_tstep((1,1.5)) == 184.
    assert reshaped.time_at_tstep((1.5,1.5)) == 184.

    new_cadence = reshaped.as_continuous()
    assert new_cadence.is_continuous
    assert new_cadence.time_at_tstep((0,0)) == 100.
    assert new_cadence.time_at_tstep((1,0)) == 164.
    assert new_cadence.time_at_tstep((1,1)) == 180.
    assert new_cadence.time_at_tstep((1,1.5)) == 188
    assert new_cadence.time_at_tstep((1.5,1.5)) == 188

    assert reshaped.tstep_at_time( 99.) == (0,0)
    assert reshaped.tstep_at_time( 99., remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(100.) == (0,0)
    assert reshaped.tstep_at_time(106.) == (0,0.75)
    assert reshaped.tstep_at_time(116.) == (0,1)
    assert reshaped.tstep_at_time(108.) == (0,1)
    assert reshaped.tstep_at_time(108., remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(132.) == (0,2)
    assert reshaped.tstep_at_time(148.) == (0,3)
    assert reshaped.tstep_at_time(140.) == (0,3)
    assert reshaped.tstep_at_time(140., remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(155.) == (0,3.875)
    assert reshaped.tstep_at_time(156.) == (0,4)
    assert reshaped.tstep_at_time(156., remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(163.99999) == (0,4)
    assert reshaped.tstep_at_time(163.99999, remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(164.) == (1,0)
    assert reshaped.tstep_at_time(1684) == (24,3)
    assert reshaped.tstep_at_time(1692) == (24,4)
    assert reshaped.tstep_at_time(1692, inclusive=False) == (25,4)
    assert reshaped.tstep_at_time(1692, inclusive=False, remask=True) == Pair.MASKED

    assert reshaped.tstep_range_at_time( 99.) == ((0,0), (1,0))
    assert reshaped.tstep_range_at_time( 99., remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(100.) == ((0,0), (1,1))
    assert reshaped.tstep_range_at_time(106.) == ((0,0), (1,1))
    assert reshaped.tstep_range_at_time(108.) == ((0,0), (1,0))
    assert reshaped.tstep_range_at_time(108., remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(115.999) == ((0,0), (1,0))
    assert reshaped.tstep_range_at_time(115.999, remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(116.) == ((0,1), (1,2))
    assert reshaped.tstep_range_at_time(148.) == ((0,3), (1,4))
    assert reshaped.tstep_range_at_time(140.) == ((0,2), (1,2))
    assert reshaped.tstep_range_at_time(140., remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(156.) == ((0,3), (1,3))
    assert reshaped.tstep_range_at_time(156., remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(163.999) == ((0,3), (1,3))
    assert reshaped.tstep_range_at_time(163.999, remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(164.) == ((1,0), (2,1))
    assert reshaped.tstep_range_at_time(1684) == ((24,3), (25,4))
    assert reshaped.tstep_range_at_time(1692) == ((24,3), (25,4))
    assert reshaped.tstep_range_at_time(1692, inclusive=False) == ((24,3), (25,3))
    assert (reshaped.tstep_range_at_time(1692, inclusive=False, remask=True)[0]
            == Pair.MASKED)

    for i in np.arange(-2., 28., 0.5):
      for j in np.arange(-1., 5., 0.25):
        k = 4*np.floor(i) + j
        (time1a, time1b) = cadence.time_range_at_tstep(k)
        (time2a, time2b) = reshaped.time_range_at_tstep((i,j), remask=True,
                                                        inclusive=False)
        if not time2a.mask:
            assert time1a == time2a
            assert time1b == time2b

            time = reshaped.time_at_tstep((i,j), remask=True)
            tstep = reshaped.tstep_at_time(time)
            assert tstep == (np.floor(i),j)

    ############################################
    # Weirdly reshaped case, 100 -> (25,4), overlapping
    # [100-116, 110-126, 120-136, 130-146], [140-156, ...], [..., 1090-1106]
    ############################################

    cadence = oops.cadence.Metronome(100., 10., 16., 100)
    reshaped = oops.cadence.ReshapedCadence(cadence, (25,4))
    assert reshaped.is_continuous
    assert not reshaped.is_unique
    assert reshaped.time_at_tstep((0,0)) == 100.
    assert reshaped.time_at_tstep((0,1)) == 110.
    assert reshaped.time_at_tstep((1,0)) == 140.
    assert reshaped.time_at_tstep((1,1)) == 150.
    assert reshaped.time_at_tstep((1,1.5)) == 158.
    assert reshaped.time_at_tstep((1.5,1.5)) == 158.

    tstep = Pair([(0,0),(0,1),(1,0),(1,1),(1,1.5),(1.5,1.5)],
                 [True] + 5*[False])
    assert reshaped.time_at_tstep(tstep) == (Scalar.MASKED, 110, 140, 150, 158, 158)
    tstep.insert_deriv('t' , Pair(np.arange(12).reshape(6,2)))
    tstep.insert_deriv('xy', Pair(np.ones((6,2,2)), drank=1))
    assert reshaped.time_at_tstep(tstep) == (Scalar.MASKED, 110, 140, 150, 158, 158)
    assert (reshaped.time_at_tstep(tstep, derivs=True).d_dt
            == (Scalar.MASKED, 48, 80, 112, 144, 176))
    assert (reshaped.time_at_tstep(tstep, derivs=True).d_dxy
            == Scalar(16 * np.ones((6,2)), [True] + 5*[False], drank=1))

    assert reshaped.time_range_at_tstep((0,0)) == (100, 116)
    assert reshaped.time_range_at_tstep((0,1)) == (110, 126)
    assert reshaped.time_range_at_tstep((1,0)) == (140, 156)
    assert reshaped.time_range_at_tstep((1,1)) == (150, 166)
    assert reshaped.time_range_at_tstep((1,1.5)) == (150, 166)
    assert reshaped.time_range_at_tstep((1.5,1.5)) == (150, 166)

    assert (reshaped.time_range_at_tstep(tstep)
            == ((Scalar.MASKED, 110, 140, 150, 150, 150), (Scalar.MASKED, 126, 156, 166, 166, 166)))

    assert reshaped.tstep_at_time( 99.) == (0,0)
    assert reshaped.tstep_at_time( 99., remask=True) == Pair.MASKED
    assert reshaped.tstep_at_time(100.) == (0,0)
    assert reshaped.tstep_at_time(108.) == (0,0.5)
    assert reshaped.tstep_at_time(110.) == (0,1)
    assert reshaped.tstep_at_time(140.) == (1,0)
    assert reshaped.tstep_at_time(1090) == (24,3)
    assert reshaped.tstep_at_time(1106) == (24,4)
    assert reshaped.tstep_at_time(1106, inclusive=False) == (25,4)
    assert reshaped.tstep_at_time(1106, inclusive=False, remask=True) == Pair.MASKED

    assert reshaped.tstep_range_at_time( 99.) == ((0,0), (1,0))
    assert reshaped.tstep_range_at_time( 99., remask=True)[0] == Pair.MASKED
    assert reshaped.tstep_range_at_time(100.) == ((0,0), (1,1))
    assert reshaped.tstep_range_at_time(106.) == ((0,0), (1,1))
    assert reshaped.tstep_range_at_time(110.) == ((0,0), (1,2))
    assert reshaped.tstep_range_at_time(139.999) == ((0,3), (1,4))
    with pytest.raises(ValueError):
        reshaped.tstep_range_at_time(140)

    assert reshaped.tstep_range_at_time(1090) == ((24,2), (25,4))
    assert reshaped.tstep_range_at_time(1095.999) == ((24,2), (25,4))
    assert reshaped.tstep_range_at_time(1096) == ((24,3), (25,4))
    assert reshaped.tstep_range_at_time(1106) == ((24,3), (25,4))
    assert reshaped.tstep_range_at_time(1106, inclusive=False) == ((24,3), (25,3))
    assert (reshaped.tstep_range_at_time(1106, inclusive=False, remask=True)[0]
            == Pair.MASKED)
################################################################################
